use anyhow::Result;
use clap::Parser;
use locall::config::{default_db_path, has_content, kind_label, load_config, Config};
use locall::embed::{load_index_image, Embedder, GpuUse};
use locall::extract::{chunk_text, extract_text, pdf_first_page_image, NotSupported};
use locall::scan::{walk_files, ScanReport};
use locall::store::{add_vector, bump_version, connect, drop_vectors, get_meta, remove_file, set_meta};
use rusqlite::params;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(name = "locall-index", about = "LocalL 고속 색인 도구 (Rust)")]
struct Args {
    #[arg(long, help = "설정의 roots 대신 이 폴더만 색인 (여러 번 지정 가능)")]
    root: Vec<String>,

    #[arg(long, help = "데이터베이스 경로")]
    db: Option<PathBuf>,

    #[arg(long, help = "파일 목록만 훑고 임베딩은 하지 않음")]
    scan_only: bool,

    #[arg(long, default_value_t = 0, help = "내용 임베딩할 파일 수 제한 (시험용)")]
    limit: usize,

    #[arg(long, help = "Rust로 못 읽는 파일을 실패로 적지 않고 대기로 남김 (뒤이어 파이썬 색인기를 돌릴 때)")]
    leave_unsupported: bool,
}

struct ExistingFile {
    id: i64,
    size: u64,
    mtime: f64,
}

fn format_num<T: std::fmt::Display>(n: T) -> String {
    let s = n.to_string();
    let mut out = String::new();
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    for (i, &c) in chars.iter().enumerate() {
        out.push(c);
        let rem = len - 1 - i;
        if rem > 0 && rem % 3 == 0 && c.is_ascii_digit() {
            out.push(',');
        }
    }
    out
}

/// `prune_outside`: 설정의 검색 폴더 전체로 도는 중이면 true. 그러면 어느 검색 폴더에도 속하지 않는
/// 항목(검색 폴더 목록에서 뺀 폴더의 파일)도 색인에서 지운다. `--root`로 일부만 돌 때는 false.
fn scan_phase(
    con: &mut rusqlite::Connection,
    cfg: &locall::config::Config,
    roots: &[String],
    prune_outside: bool,
) -> Result<()> {
    let t0 = Instant::now();
    let mut existing: HashMap<String, ExistingFile> = HashMap::new();

    {
        let mut stmt = con.prepare("SELECT id, path, size, mtime FROM files")?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(1)?,
                ExistingFile {
                    id: row.get(0)?,
                    size: row.get::<_, Option<i64>>(2)?.unwrap_or(0) as u64,
                    mtime: row.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
                },
            ))
        })?;
        for r in rows.flatten() {
            existing.insert(r.0, r.1);
        }
    }

    let mut seen = HashSet::new();
    let mut report = ScanReport::default();
    let mut kinds_count: HashMap<String, usize> = HashMap::new();
    let mut new_count = 0;
    let mut changed_count = 0;

    let tx = con.transaction()?;

    {
        let mut insert_stmt = tx.prepare_cached(
            "INSERT OR IGNORE INTO files(path, name, ext, kind, size, mtime, content_state) \
             VALUES(?, ?, ?, ?, ?, ?, ?)",
        )?;

        let mut update_stmt = tx.prepare_cached(
            "UPDATE files SET size=?, mtime=?, content_state=? WHERE id=?",
        )?;

        walk_files(roots, cfg, &mut report, |entry| {
            *kinds_count.entry(entry.kind.clone()).or_insert(0) += 1;
            let state = if has_content(&entry.kind, &entry.ext, cfg) { 0 } else { 2 };

            if let Some(row) = existing.get(&entry.path) {
                seen.insert(row.id);
                if !entry.is_dir && (row.size != entry.size || (row.mtime - entry.mtime).abs() > 1.0) {
                    // 내용이 바뀌었으니 예전 본문/이미지 벡터는 버린다 (안 버리면 다음 색인 때 중복으로 쌓인다)
                    let _ = drop_vectors(&tx, row.id, Some("text"));
                    let _ = drop_vectors(&tx, row.id, Some("image"));
                    let _ = update_stmt.execute(params![entry.size as i64, entry.mtime, state, row.id]);
                    changed_count += 1;
                }
            } else {
                let _ = insert_stmt.execute(params![
                    entry.path,
                    entry.name,
                    entry.ext,
                    entry.kind,
                    entry.size as i64,
                    entry.mtime,
                    state,
                ]);
                new_count += 1;
            }
        });
    }

    // 사라진 항목 정리
    let mut removed_count = 0;
    let mut outside_count = 0;
    let with_slash = |r: &String| {
        let mut s = r.replace('/', "\\").to_lowercase();
        if !s.ends_with('\\') {
            s.push('\\');
        }
        s
    };
    let live_roots: Vec<String> = roots
        .iter()
        .filter(|r| std::path::Path::new(r).is_dir() || report.cloud.contains(*r))
        .map(with_slash)
        .collect();
    // 지금 없는 검색 폴더(뽑힌 USB 등)도 포함: 그 파일은 다시 꽂을 때를 위해 남겨 둔다
    let all_roots: Vec<String> = roots.iter().map(with_slash).collect();
    let prune_outside = prune_outside && !all_roots.is_empty();

    for (path, row) in &existing {
        if !seen.contains(&row.id) {
            let path_norm = path.replace('/', "\\").to_lowercase();
            if live_roots.iter().any(|lr| path_norm.starts_with(lr)) {
                remove_file(&tx, row.id)?;
                removed_count += 1;
            } else if prune_outside && !all_roots.iter().any(|r| path_norm.starts_with(r)) {
                remove_file(&tx, row.id)?;
                outside_count += 1;
            }
        }
    }

    tx.commit()?;

    let total_scanned: usize = kinds_count.values().sum();
    let elapsed = t0.elapsed().as_secs_f64();

    println!(
        "[훑기] {}개 확인 ({:.1}초) · 새로 {} · 변경 {} · 삭제 {}{}",
        format_num(total_scanned),
        elapsed,
        format_num(new_count),
        format_num(changed_count),
        format_num(removed_count),
        if outside_count > 0 { format!(" · 검색 폴더 밖 정리 {}", format_num(outside_count)) } else { String::new() }
    );

    let mut sorted_kinds: Vec<_> = kinds_count.into_iter().collect();
    sorted_kinds.sort_by(|a, b| b.1.cmp(&a.1));
    let kinds_summary: Vec<String> = sorted_kinds
        .into_iter()
        .map(|(k, v)| format!("{} {}", kind_label(&k), format_num(v)))
        .collect();
    println!("       {}", kinds_summary.join(" · "));

    for p in &report.cloud {
        println!("       클라우드라서 건너뜀: {}", p);
    }
    if report.denied > 0 {
        println!("       접근 권한이 없어 건너뛴 폴더 {}개", format_num(report.denied));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// 임베딩 단계 (파이썬 locall/index.py의 name_phase / docs_phase와 같은 결과를 내도록)
// ---------------------------------------------------------------------------

const MB: u64 = 1024 * 1024;

/// 파이썬 `doc_text`
fn doc_text(title: &str, content: &str) -> String {
    format!("title: {} | text: {}", if title.is_empty() { "none" } else { title }, content)
}

/// 파이썬 `os.path.splitext(name)[0]` (앞쪽 점만 있는 이름은 확장자로 보지 않는다)
fn splitext_root(name: &str) -> &str {
    if let Some(dot) = name.rfind('.') {
        if name[..dot].chars().any(|c| c != '.') {
            return &name[..dot];
        }
    }
    name
}

/// 파이썬 `name_text`: 파일명과 위치만으로 만드는 설명문 (내용을 못 읽는 파일도 찾을 수 있게)
fn name_text(path: &str, name: &str, kind: &str) -> String {
    let stem = if kind == "dir" { name } else { splitext_root(name) };
    // re.sub(r"[_\-.]+", " ", stem)
    let mut s = String::with_capacity(stem.len());
    let mut in_run = false;
    for c in stem.chars() {
        if matches!(c, '_' | '-' | '.') {
            if !in_run {
                s.push(' ');
                in_run = true;
            }
        } else {
            s.push(c);
            in_run = false;
        }
    }
    // os.path.dirname(path).replace(":", "").split(os.sep) 중 빈 것 빼고 마지막 4개
    let dir = path.rfind(['\\', '/']).map_or("", |i| &path[..i]);
    let dir = dir.replace(':', "");
    let parents: Vec<&str> = dir.split('\\').filter(|p| !p.is_empty()).collect();
    let parents = &parents[parents.len().saturating_sub(4)..];
    let label = match kind {
        "dir" | "doc" | "image" | "video" | "audio" | "archive" | "app" | "code" => kind_label(kind),
        _ => "파일",
    };
    doc_text(name, &format!("{} {}. 위치: {}", label, s, parents.join(" > ")))
}

fn to_f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| half::f16::from_f32(x).to_le_bytes()).collect()
}

/// 진행 상황: 5초마다 한 줄, 60초마다 색인 버전을 올려 떠 있는 검색 앱이 새 내용을 읽게 한다
struct Progress {
    label: &'static str,
    total: usize,
    t0: Instant,
    last: Instant,
    last_bump: Instant,
}

impl Progress {
    fn new(label: &'static str, total: usize) -> Self {
        let now = Instant::now();
        Self { label, total, t0: now, last: now, last_bump: now }
    }

    fn tick(&mut self, con: &rusqlite::Connection, done: usize, extra: &str) {
        if self.last.elapsed().as_secs_f64() >= 5.0 || done == self.total {
            self.last = Instant::now();
            println!(
                "[{}] {}/{} {}({:.0}초)",
                self.label,
                format_num(done),
                format_num(self.total),
                extra,
                self.t0.elapsed().as_secs_f64()
            );
        }
        if self.last_bump.elapsed().as_secs() >= 60 {
            self.last_bump = Instant::now();
            let _ = bump_version(con);
        }
    }
}

fn name_phase(con: &mut rusqlite::Connection, emb: &mut Embedder) -> Result<()> {
    let rows: Vec<(i64, String, String, String)> = {
        let mut stmt = con.prepare("SELECT id, path, name, kind FROM files WHERE name_done=0")?;
        let r = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get::<_, Option<String>>(3)?.unwrap_or_default())))?
            .flatten()
            .collect();
        r
    };
    if rows.is_empty() {
        return Ok(());
    }
    let mut prog = Progress::new("파일명", rows.len());
    for (bi, batch) in rows.chunks(512).enumerate() {
        let texts: Vec<String> = batch.iter().map(|(_, p, n, k)| name_text(p, n, k)).collect();
        let vecs = emb.docs(&texts)?;
        let tx = con.transaction()?;
        for ((fid, path, _, _), vec) in batch.iter().zip(&vecs) {
            drop_vectors(&tx, *fid, Some("name"))?;
            add_vector(&tx, *fid, "name", &to_f16_bytes(vec), Some(path))?;
            tx.execute("UPDATE files SET name_done=1 WHERE id=?", params![fid])?;
        }
        tx.commit()?;
        prog.tick(con, (bi * 512 + batch.len()).min(rows.len()), "");
    }
    bump_version(con)?;
    Ok(())
}

/// 본문 조각을 모아 두었다가 한꺼번에 임베딩해서 쓴다
struct DocBuffer {
    files: Vec<(i64, String, Vec<String>)>, // (file_id, 파일명, 조각들)
    chunks: usize,
}

fn flush_docs(con: &mut rusqlite::Connection, emb: &mut Embedder, buf: &mut DocBuffer) -> Result<()> {
    if buf.files.is_empty() {
        return Ok(());
    }
    let texts: Vec<String> =
        buf.files.iter().flat_map(|(_, name, chunks)| chunks.iter().map(move |c| doc_text(name, c))).collect();
    let tx = con.transaction()?;
    match emb.docs(&texts) {
        Ok(vecs) => {
            let mut i = 0;
            for (fid, _, chunks) in &buf.files {
                for c in chunks {
                    add_vector(&tx, *fid, "text", &to_f16_bytes(&vecs[i]), Some(c))?;
                    i += 1;
                }
                tx.execute("UPDATE files SET content_state=1 WHERE id=?", params![fid])?;
            }
        }
        Err(_) => {
            // 묶음이 실패하면 파일 하나씩 다시 해 보고, 그래도 안 되면 실패로 적는다
            for (fid, name, chunks) in &buf.files {
                let f_texts: Vec<String> = chunks.iter().map(|c| doc_text(name, c)).collect();
                match emb.docs(&f_texts) {
                    Ok(vecs) => {
                        for (c, v) in chunks.iter().zip(&vecs) {
                            add_vector(&tx, *fid, "text", &to_f16_bytes(v), Some(c))?;
                        }
                        tx.execute("UPDATE files SET content_state=1 WHERE id=?", params![fid])?;
                    }
                    Err(_) => {
                        tx.execute("UPDATE files SET content_state=-1 WHERE id=?", params![fid])?;
                    }
                }
            }
        }
    }
    tx.commit()?;
    buf.files.clear();
    buf.chunks = 0;
    Ok(())
}

/// 문서 본문 → 조각 → 임베딩. Rust로 못 읽는 파일은 대기로 남기고 그 수를,
/// 글자가 없는 PDF(스캔본)는 목록으로 돌려준다.
fn docs_phase(
    con: &mut rusqlite::Connection,
    emb: &mut Embedder,
    cfg: &Config,
    rows: &[(i64, String, String, String, u64)],
) -> Result<(Vec<i64>, Vec<(i64, String)>)> {
    use rayon::prelude::*;
    let mut prog = Progress::new("문서", rows.len());
    let mut buf = DocBuffer { files: Vec::new(), chunks: 0 };
    let mut n_chunks = 0usize;
    let mut left = Vec::new();
    let mut scanned = Vec::new();
    let mut done = 0usize;
    // 32개씩 여러 스레드로 글자를 꺼낸 뒤, 순서대로 조각을 모아 임베딩한다
    for group in rows.chunks(32) {
        let extracted: Vec<Option<Result<Vec<String>>>> = group
            .par_iter()
            .map(|(_, path, _, ext, size)| {
                if *size > cfg.max_doc_mb * MB {
                    return None;
                }
                Some(
                    extract_text(Path::new(path), ext)
                        .map(|t| chunk_text(&t, cfg.chunk_chars, cfg.max_chunks_per_file)),
                )
            })
            .collect();
        for ((fid, path, name, ext, _), res) in group.iter().zip(extracted) {
            done += 1;
            match res {
                None => {
                    con.execute("UPDATE files SET content_state=2 WHERE id=?", params![fid])?;
                }
                Some(Err(e)) if e.downcast_ref::<NotSupported>().is_some() => left.push(*fid),
                Some(Err(_)) => {
                    con.execute("UPDATE files SET content_state=-1 WHERE id=?", params![fid])?;
                }
                Some(Ok(chunks)) if chunks.is_empty() && ext == "pdf" => scanned.push((*fid, path.clone())),
                Some(Ok(chunks)) if chunks.is_empty() => {
                    con.execute("UPDATE files SET content_state=1 WHERE id=?", params![fid])?;
                }
                Some(Ok(chunks)) => {
                    buf.chunks += chunks.len();
                    n_chunks += chunks.len();
                    buf.files.push((*fid, name.clone(), chunks));
                    if buf.chunks >= 64 {
                        flush_docs(con, emb, &mut buf)?;
                    }
                }
            }
            prog.tick(con, done, &format!("· 조각 {} ", format_num(n_chunks)));
        }
    }
    flush_docs(con, emb, &mut buf)?;
    Ok((left, scanned))
}

/// 스캔 PDF: 첫 쪽을 그림으로 그려 사진처럼 임베딩한다 (파이썬 images_phase + pdf_first_page_image).
/// 시각 인코더가 없으면 대기로 남기고 그 수를 돌려준다.
fn scanned_pdf_phase(con: &mut rusqlite::Connection, emb: &mut Embedder, rows: &[(i64, String)]) -> Result<Vec<i64>> {
    if !emb.has_vision() {
        return Ok(rows.iter().map(|r| r.0).collect());
    }
    let mut left = Vec::new();
    let mut prog = Progress::new("스캔 PDF", rows.len());
    for (i, (fid, path)) in rows.iter().enumerate() {
        let (state, vec) = match pdf_first_page_image(Path::new(path)) {
            Ok(None) => (2, None),
            Ok(Some(img)) => match emb.image_rgb(&img) {
                Ok(v) => (1, Some(v)),
                Err(_) => (-1, None),
            },
            Err(e) if e.downcast_ref::<NotSupported>().is_some() => {
                left.push(*fid);
                prog.tick(con, i + 1, "");
                continue;
            }
            Err(_) => (-1, None),
        };
        let tx = con.transaction()?;
        if let Some(v) = vec {
            add_vector(&tx, *fid, "image", &to_f16_bytes(&v), None)?;
        }
        tx.execute("UPDATE files SET content_state=? WHERE id=?", params![state, fid])?;
        tx.commit()?;
        prog.tick(con, i + 1, "");
    }
    Ok(left)
}

/// 사진: 여러 스레드로 불러 줄인 뒤 하나씩 임베딩 (파이썬 images_phase + load_image)
fn images_phase(con: &mut rusqlite::Connection, emb: &mut Embedder, rows: &[(i64, String, String)]) -> Result<Vec<i64>> {
    use rayon::prelude::*;
    if !emb.has_vision() {
        return Ok(rows.iter().map(|r| r.0).collect());
    }
    let mut left = Vec::new();
    let mut prog = Progress::new("사진", rows.len());
    let mut done = 0usize;
    for group in rows.chunks(16) {
        let loaded: Vec<Result<Option<image::RgbImage>>> =
            group.par_iter().map(|(_, path, _)| load_index_image(Path::new(path))).collect();
        let tx = con.transaction()?;
        for ((fid, _, ext), res) in group.iter().zip(loaded) {
            done += 1;
            let state = match res {
                Ok(None) => 2, // 아이콘처럼 너무 작은 그림
                Ok(Some(img)) => match emb.image_rgb(&img) {
                    Ok(v) => {
                        add_vector(&tx, *fid, "image", &to_f16_bytes(&v), None)?;
                        1
                    }
                    Err(_) => -1,
                },
                // image 크레이트가 모르는 형식(TIFF 등)은 파이썬 색인기 몫
                Err(_) if matches!(ext.as_str(), "tif" | "tiff") => {
                    left.push(*fid);
                    continue;
                }
                Err(_) => -1,
            };
            tx.execute("UPDATE files SET content_state=? WHERE id=?", params![state, fid])?;
        }
        tx.commit()?;
        prog.tick(con, done, "");
    }
    Ok(left)
}

fn content_phase(
    con: &mut rusqlite::Connection,
    emb: &mut Embedder,
    cfg: &Config,
    limit: usize,
    leave_unsupported: bool,
) -> Result<()> {
    let q = |cond: &str| -> Result<Vec<(i64, String, String, String, u64)>> {
        let sql = format!(
            "SELECT id, path, name, ext, size FROM files WHERE content_state=0 AND kind{} ORDER BY id",
            cond
        );
        let mut stmt = con.prepare(&sql)?;
        let r = stmt
            .query_map([], |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    r.get::<_, Option<i64>>(4)?.unwrap_or(0).max(0) as u64,
                ))
            })?
            .flatten()
            .collect();
        Ok(r)
    };
    let mut docs = q("!='image'")?;
    let imgs = q("='image'")?;
    // 너무 작은 사진(아이콘 등)은 대상 아님
    let small: Vec<i64> = imgs.iter().filter(|r| r.4 < cfg.min_image_kb * 1024).map(|r| r.0).collect();
    {
        let tx = con.transaction()?;
        for fid in &small {
            tx.execute("UPDATE files SET content_state=2 WHERE id=?", params![fid])?;
        }
        tx.commit()?;
    }
    let mut imgs: Vec<(i64, String, String)> = imgs
        .into_iter()
        .filter(|r| r.4 >= cfg.min_image_kb * 1024)
        .map(|(id, path, _, ext, _)| (id, path, ext))
        .collect();
    if limit > 0 {
        docs.truncate(limit);
        imgs.truncate(limit);
    }
    println!("[내용] 처리할 문서 {}개 · 사진 {}개", format_num(docs.len()), format_num(imgs.len()));
    let (mut left, scanned) = if docs.is_empty() { (Vec::new(), Vec::new()) } else { docs_phase(con, emb, cfg, &docs)? };
    if !scanned.is_empty() {
        left.extend(scanned_pdf_phase(con, emb, &scanned)?);
    }
    if !imgs.is_empty() {
        left.extend(images_phase(con, emb, &imgs)?);
    }
    if !left.is_empty() {
        if leave_unsupported {
            println!("[남은 일] Rust로 못 읽은 파일 {}개 → 파이썬 색인기가 이어서 처리합니다", format_num(left.len()));
        } else {
            // 파이썬이 없으면 다음에도 못 읽으므로 실패로 적어 매번 다시 시도하지 않게 한다
            let tx = con.transaction()?;
            for fid in &left {
                tx.execute("UPDATE files SET content_state=-1 WHERE id=?", params![fid])?;
            }
            tx.commit()?;
            println!("[알림] 읽지 못한 파일 {}개는 건너뜁니다 (TIFF, 손상된 문서 등)", format_num(left.len()));
        }
    }
    Ok(())
}

/// 배포본엔 모델이 없으므로 처음 한 번 받는다 (이미 있으면 바로 넘어간다)
fn download_models() -> Result<()> {
    use locall::download::{ensure_models, human, missing_bytes, models_missing, Progress};
    use std::io::Write;
    if !models_missing() {
        return Ok(());
    }
    println!(
        "[모델 받기] AI 모델 {}을(를) 처음 한 번 받습니다 (Hugging Face). 끊겨도 다음에 이어 받습니다.",
        human(missing_bytes())
    );
    let t = Instant::now();
    let mut last = Instant::now() - Duration::from_secs(10);
    let mut inline = false;
    // 속도는 이번에 받은 양으로 (이어 받기 전 앞부분은 뺀다)
    let mut start: Option<(u64, Instant)> = None;
    ensure_models(&mut |p| match p {
        Progress::Waiting => println!("[모델 받기] 검색 앱이 받고 있어 끝나기를 기다립니다…"),
        Progress::Bytes { done, total, file } => {
            let (from, since) = *start.get_or_insert((done, Instant::now()));
            if last.elapsed() >= Duration::from_secs(1) || done == total {
                let secs = since.elapsed().as_secs_f64().max(0.1);
                print!(
                    "\r[모델 받기] {} / {} ({}%) · {:.1}MB/s · {}      ",
                    human(done),
                    human(total),
                    done * 100 / total.max(1),
                    done.saturating_sub(from) as f64 / 1_048_576.0 / secs,
                    file
                );
                let _ = std::io::stdout().flush();
                inline = true;
                last = Instant::now();
            }
        }
        Progress::Checking { file } => {
            if std::mem::take(&mut inline) {
                println!();
            }
            println!("[모델 받기] {} 확인 중…", file);
        }
    })?;
    if inline {
        println!();
    }
    println!("[모델 받기] 끝 ({:.0}초)", t.elapsed().as_secs_f64());
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = load_config(None)?;

    let from_config = args.root.is_empty();
    let roots = if from_config { cfg.roots.clone() } else { args.root };

    let db_path = args.db.unwrap_or_else(default_db_path);
    let mut con = connect(&db_path, false)?;

    // 다른 모델/차원으로 만든 색인에 섞어 쓰지 않는다
    let signature = format!("{}@{}", cfg.model, cfg.dim);
    if let Some(saved) = get_meta(&con, "signature")? {
        if saved != signature {
            anyhow::bail!("색인이 다른 모델/차원({})으로 만들어졌습니다. {:?} 를 지우고 다시 실행하세요.", saved, db_path);
        }
    }
    set_meta(&con, "signature", &signature)?;

    scan_phase(&mut con, &cfg, &roots, from_config)?;
    bump_version(&con)?; // 떠 있는 검색 앱이 바뀐 색인을 다시 읽게 한다

    if args.scan_only {
        let mut stmt = con.prepare("SELECT kind, COUNT(*) FROM files WHERE content_state=0 GROUP BY kind")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
        let mut pending = Vec::new();
        for r in rows.flatten() {
            pending.push(format!("{} {}", kind_label(&r.0), format_num(r.1)));
        }
        let summary = if pending.is_empty() { "없음".to_string() } else { pending.join(" · ") };
        println!("[예상] 내용 임베딩 대기: {}", summary);
        return Ok(());
    }

    download_models()?;

    println!("[모델] 불러오는 중…");
    let t = Instant::now();
    // 색인은 묶음이 크므로 글자·사진 모두 GPU로 (없으면 CPU)
    let mut emb = Embedder::spawn_onnx(&cfg, GpuUse { text: true, vision: true })?;
    println!("[모델] {} ({:.1}초)", emb.device, t.elapsed().as_secs_f64());

    let result = name_phase(&mut con, &mut emb)
        .and_then(|_| content_phase(&mut con, &mut emb, &cfg, args.limit, args.leave_unsupported));
    bump_version(&con)?;
    result?;
    println!("[완료]");
    Ok(())
}
