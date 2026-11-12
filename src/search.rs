use crate::config::{is_image_content_ext, is_runnable_ext, Config};
use crate::store;
use anyhow::Result;
use half::f16;
use rayon::prelude::*;
use rusqlite::{params, Connection};
use simsimd::SpatialSimilarity;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    SpatialSimilarity::dot(a, b).unwrap_or_else(|| {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>() as f64
    }) as f32
}

/// 반정밀도 벡터끼리의 내적 (합산은 f32 이상으로).
/// 색인 벡터는 DB에 float16으로 저장되어 있으므로, f32로 늘리지 않고 그대로 들고 있으면
/// 메모리가 절반이다 (벡터 14만 개 기준 288MB → 144MB). 파이썬 검색기의 GPU 경로도 같은 방식이다.
#[inline]
pub fn dot_f16(a: &[f16], b: &[f16]) -> f32 {
    // half::f16과 simsimd::f16은 둘 다 u16 하나를 감싼 repr(transparent) 타입이라 메모리 배치가 같다
    let (sa, sb) = unsafe {
        (
            std::slice::from_raw_parts(a.as_ptr() as *const simsimd::f16, a.len()),
            std::slice::from_raw_parts(b.as_ptr() as *const simsimd::f16, b.len()),
        )
    };
    match SpatialSimilarity::dot(sa, sb) {
        Some(d) => d as f32,
        None => a.iter().zip(b).map(|(x, y)| x.to_f32() * y.to_f32()).sum(),
    }
}

/// 질의 벡터를 색인과 같은 반정밀도로
fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

use serde::Serialize;

pub const VKIND_NAME: u8 = 0;
pub const VKIND_TEXT: u8 = 1;
pub const VKIND_IMAGE: u8 = 2;

pub const KEYWORD_BONUS: f32 = 0.06;
pub const MIN_SCORE: f32 = 0.02;
pub const BASELINE_PRIOR: [f32; 3] = [0.62, 0.59, 0.56];
pub const PRIOR_WEIGHT: f32 = 20.0;
pub const TEXT_HIT_WEIGHT: f32 = 0.5;
pub const COMMON_TOKEN_HITS: usize = 5000;

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub id: i64,
    pub name: String,
    pub path: String,
    pub dir: String,
    pub kind: String,
    pub ext: String,
    pub size: u64,
    pub mtime: f64,
    pub score: f32,
    pub match_type: String,
    pub snippet: String,
    pub runnable: bool,
    pub thumb: bool,
}

#[derive(Debug, Clone)]
pub struct FileItem {
    pub id: i64,
    pub path: String,
    pub name: String,
    pub ext: String,
    pub kind: String,
    pub size: u64,
    pub mtime: f64,
}

pub struct Searcher {
    pub db_path: PathBuf,
    pub dim: usize,
    pub files: HashMap<i64, FileItem>,
    pub file_counts: HashMap<String, usize>,
    pub vec_ids: Vec<i64>,
    pub fids: Vec<i64>,
    pub row_kind: Vec<u8>,
    pub row_file: Vec<usize>,
    pub uniq: Vec<i64>,
    pub file_kind: Vec<String>,
    pub starts: Vec<usize>,
    pub ends: Vec<usize>,
    pub vectors: Vec<f16>, // shape: N * dim (정규화된 벡터, DB에 저장된 float16 그대로)
    pub version: Option<String>,
    vid_to_row: HashMap<i64, usize>,
    fid_to_idx: HashMap<i64, usize>,
    checked: Option<Instant>,
}

impl Searcher {
    pub fn new(cfg: &Config, db_path: &Path) -> Result<Self> {
        let mut s = Self {
            db_path: db_path.to_path_buf(),
            dim: cfg.dim,
            files: HashMap::new(),
            file_counts: HashMap::new(),
            vec_ids: Vec::new(),
            fids: Vec::new(),
            row_kind: Vec::new(),
            row_file: Vec::new(),
            uniq: Vec::new(),
            file_kind: Vec::new(),
            starts: Vec::new(),
            ends: Vec::new(),
            vectors: Vec::new(),
            version: None,
            vid_to_row: HashMap::new(),
            fid_to_idx: HashMap::new(),
            checked: None,
        };
        s.load()?;
        Ok(s)
    }

    pub fn load(&mut self) -> Result<()> {
        self.files.clear();
        self.file_counts.clear();
        self.vec_ids.clear();
        self.fids.clear();
        self.row_kind.clear();
        self.vectors.clear();
        self.uniq.clear();
        self.starts.clear();
        self.ends.clear();
        self.row_file.clear();
        self.file_kind.clear();
        self.vid_to_row.clear();
        self.fid_to_idx.clear();
        self.checked = Some(Instant::now());

        if !self.db_path.exists() {
            return Ok(());
        }

        let con = store::connect(&self.db_path, true)?;
        self.version = store::get_meta(&con, "version")?;

        // 1. files 로드
        {
            let mut stmt = con.prepare("SELECT id, path, name, ext, kind, size, mtime FROM files")?;
            let rows = stmt.query_map([], |row| {
                Ok(FileItem {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    name: row.get(2)?,
                    ext: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    kind: row.get(4)?,
                    size: row.get::<_, Option<i64>>(5)?.unwrap_or(0) as u64,
                    mtime: row.get::<_, Option<f64>>(6)?.unwrap_or(0.0),
                })
            })?;
            for item in rows.flatten() {
                *self.file_counts.entry(item.kind.clone()).or_insert(0) += 1;
                self.files.insert(item.id, item);
            }
        }

        // 2. vectors 로드
        {
            // 미리 자리를 잡아 두면 늘려 가며 복사하느라 순간적으로 두 배를 쓰는 일이 없다
            let total: i64 = con.query_row("SELECT COUNT(*) FROM vectors", [], |r| r.get(0))?;
            let total = total.max(0) as usize;
            self.vectors.reserve_exact(total * self.dim);
            self.vec_ids.reserve_exact(total);
            self.fids.reserve_exact(total);
            self.row_kind.reserve_exact(total);

            let mut stmt = con.prepare("SELECT id, file_id, kind, vec FROM vectors ORDER BY file_id, id")?;
            let rows = stmt.query_map([], |row| {
                let vid: i64 = row.get(0)?;
                let fid: i64 = row.get(1)?;
                let kind_str: String = row.get(2)?;
                let blob: Vec<u8> = row.get(3)?;
                Ok((vid, fid, kind_str, blob))
            })?;

            for r in rows.flatten() {
                let (vid, fid, kind_str, blob) = r;
                if !self.files.contains_key(&fid) {
                    continue;
                }

                let k = match kind_str.as_str() {
                    "text" => VKIND_TEXT,
                    "image" => VKIND_IMAGE,
                    _ => VKIND_NAME,
                };

                // blob: Little-endian float16 배열 (변환 없이 그대로 보관)
                if blob.len() == self.dim * 2 {
                    self.vec_ids.push(vid);
                    self.fids.push(fid);
                    self.row_kind.push(k);

                    self.vectors
                        .extend(blob.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]])));
                }
            }
        }

        if self.fids.is_empty() {
            return Ok(());
        }

        // 3. 파일별 구간 매핑 (uniq, starts, ends, row_file)
        let mut cur_fid = None;
        let mut file_idx = 0;

        for (i, &fid) in self.fids.iter().enumerate() {
            if cur_fid != Some(fid) {
                if cur_fid.is_some() {
                    self.ends.push(i);
                }
                cur_fid = Some(fid);
                self.uniq.push(fid);
                self.starts.push(i);
                if let Some(f) = self.files.get(&fid) {
                    self.file_kind.push(f.kind.clone());
                } else {
                    self.file_kind.push(String::new());
                }
                if i > 0 {
                    file_idx += 1;
                }
            }
            self.row_file.push(file_idx);
        }
        if !self.fids.is_empty() {
            self.ends.push(self.fids.len());
        }

        self.vid_to_row = self.vec_ids.iter().enumerate().map(|(row, &vid)| (vid, row)).collect();
        self.fid_to_idx = self.uniq.iter().enumerate().map(|(idx, &fid)| (fid, idx)).collect();

        Ok(())
    }

    /// 색인이 갱신됐으면(meta.version 변경) 다시 읽는다. 5초에 한 번만 확인한다.
    pub fn maybe_reload(&mut self) -> Result<bool> {
        if self.checked.map_or(false, |t| t.elapsed() < Duration::from_secs(5)) {
            return Ok(false);
        }
        self.checked = Some(Instant::now());
        if !self.db_path.exists() {
            return Ok(false);
        }
        let version = {
            let con = store::connect(&self.db_path, true)?;
            store::get_meta(&con, "version")?
        };
        if version == self.version {
            return Ok(false);
        }
        self.load()?;
        Ok(true)
    }

    pub fn stats(&self) -> (usize, usize) {
        (self.files.len(), self.vec_ids.len())
    }

    // ---------- 키워드 가산점 산출 ----------
    fn compute_keyword_bonus(
        &self,
        con: &Connection,
        q: &str,
    ) -> Result<(Vec<f32>, HashMap<usize, (usize, String)>)> {
        let n_files = self.uniq.len();
        let tokens: Vec<String> = q
            .split_whitespace()
            .map(|t| t.to_lowercase())
            .filter(|t| t.chars().count() >= 2)
            .take(8)
            .collect();

        let mut bonus = vec![0.0f32; n_files];
        let mut text_hit = HashMap::new();

        if tokens.is_empty() {
            return Ok((bonus, text_hit));
        }

        for tok in &tokens {
            let mut w = vec![0.0f32; n_files];
            if tok.chars().count() >= 3 {
                let phrase = format!("\"{}\"", tok.replace('"', "\"\""));
                let mut stmt = con.prepare_cached(
                    "SELECT rowid FROM fts WHERE fts MATCH ? LIMIT ?",
                )?;
                let rows = stmt.query_map(params![phrase, (COMMON_TOKEN_HITS + 1) as i64], |r| {
                    r.get::<_, i64>(0)
                })?;
                let vids: Vec<i64> = rows.flatten().collect();

                if !vids.is_empty() && vids.len() <= COMMON_TOKEN_HITS {
                    for vid in vids {
                        if let Some(&r) = self.vid_to_row.get(&vid) {
                            let fi = self.row_file[r];
                            let is_name = self.row_kind[r] == VKIND_NAME;
                            let weight = if is_name { 1.0 } else { TEXT_HIT_WEIGHT };
                            if weight > w[fi] {
                                w[fi] = weight;
                            }
                            if !is_name {
                                text_hit.entry(fi).or_insert((r, tok.clone()));
                            }
                        }
                    }
                }
            } else {
                let like = format!("%{}%", tok.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_"));
                let mut stmt = con.prepare_cached(
                    "SELECT id FROM files WHERE name LIKE ? ESCAPE '\\' LIMIT ?",
                )?;
                let rows = stmt.query_map(params![like, (COMMON_TOKEN_HITS + 1) as i64], |r| {
                    r.get::<_, i64>(0)
                })?;
                let fids: Vec<i64> = rows.flatten().collect();

                if !fids.is_empty() && fids.len() <= COMMON_TOKEN_HITS {
                    for fid in fids {
                        if let Some(&fi) = self.fid_to_idx.get(&fid) {
                            w[fi] = 1.0;
                        }
                    }
                }
            }

            for i in 0..n_files {
                bonus[i] += w[i];
            }
        }

        let scale = KEYWORD_BONUS / tokens.len() as f32;
        for b in &mut bonus {
            *b *= scale;
        }

        Ok((bonus, text_hit))
    }

    fn snippet(text: &str, token: Option<&str>, width: usize) -> String {
        let single_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let chars: Vec<char> = single_line.chars().collect();
        let at_char = if let Some(tok) = token {
            let tok_chars: Vec<char> = tok.to_lowercase().chars().collect();
            let lower_chars: Vec<char> = single_line.to_lowercase().chars().collect();
            if tok_chars.is_empty() {
                None
            } else {
                lower_chars.windows(tok_chars.len()).position(|w| w == tok_chars.as_slice())
            }
        } else {
            None
        };

        let start = match at_char {
            Some(idx) if idx > width / 3 => idx - width / 3,
            _ => 0,
        };
        let start = start.min(chars.len());
        let end = (start + width).min(chars.len());
        let cut: String = chars[start..end].iter().collect();

        format!(
            "{}{}{}",
            if start > 0 { "…" } else { "" },
            cut,
            if end < chars.len() { "…" } else { "" }
        )
    }

    fn fts_text(con: &Connection, vid: i64) -> String {
        con.query_row(
            "SELECT text FROM fts WHERE rowid=?",
            params![vid],
            |r| r.get::<_, String>(0),
        )
        .unwrap_or_default()
    }

    // ---------- 고속 하이브리드 검색 ----------
    ///
    /// `query_vec`는 검색어를 임베딩한 벡터. 모델이 아직 준비되지 않았으면 None을 넘기며,
    /// 이때는 의미 점수 없이 파일명/본문 키워드 일치만으로 순위를 매긴다.
    pub fn search(&self, q: &str, query_vec: Option<&[f32]>, kind: &str, n: usize) -> Result<Vec<SearchResult>> {
        let n_files = self.uniq.len();
        if n_files == 0 || self.vectors.is_empty() {
            return Ok(Vec::new());
        }
        if query_vec.map_or(false, |v| v.len() != self.dim) {
            anyhow::bail!("질의 벡터 차원이 색인({})과 다릅니다", self.dim);
        }

        let n_vecs = self.vec_ids.len();

        let sims: Vec<f32> = if let Some(query_vec) = query_vec {
            let query_vec = to_f16(query_vec);
            // 1. Rayon 병렬 SIMD 내적 연산 (raw 코사인 유사도)
            let raw: Vec<f32> = (0..n_vecs)
                .into_par_iter()
                .map(|i| {
                    let start = i * self.dim;
                    let end = start + self.dim;
                    let v = &self.vectors[start..end];
                    dot_f16(v, &query_vec)
                })
                .collect();

            // 2. 종류별(name, text, image) 편향 보정
            let mut sums = [0.0f64; 3];
            let mut counts = [0.0f64; 3];
            for (i, &k) in self.row_kind.iter().enumerate() {
                let ku = k as usize;
                sums[ku] += raw[i] as f64;
                counts[ku] += 1.0;
            }

            let mut base = [0.0f32; 3];
            for k in 0..3 {
                base[k] = ((sums[k] + (BASELINE_PRIOR[k] * PRIOR_WEIGHT) as f64)
                    / (counts[k] + PRIOR_WEIGHT as f64)) as f32;
            }

            (0..n_vecs)
                .into_par_iter()
                .map(|i| raw[i] - base[self.row_kind[i] as usize])
                .collect()
        } else {
            vec![0.0f32; n_vecs]
        };

        // 3. 파일 단위 max 점수 집계 (name + content)
        let mut per_file = vec![0.0f32; n_files];
        let mut best_vector = vec![0usize; n_files];

        for fi in 0..n_files {
            let s = self.starts[fi];
            let e = self.ends[fi];

            let mut max_name = 0.0f32;
            let mut max_content = 0.0f32;
            let mut best_idx = s;
            let mut best_sim = -999.0f32;

            for i in s..e {
                let sim = sims[i];
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
                if self.row_kind[i] == VKIND_NAME {
                    if sim > max_name {
                        max_name = sim;
                    }
                } else if sim > max_content {
                    max_content = sim;
                }
            }

            per_file[fi] = max_name + max_content;
            best_vector[fi] = best_idx;
        }

        // 4. 키워드 가산점 결합
        let con = store::connect(&self.db_path, true)?;
        let (bonus, text_hit) = self.compute_keyword_bonus(&con, q)?;

        // 5. 점수 합산 및 정렬
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(n_files);

        for fi in 0..n_files {
            if kind != "all" && self.file_kind[fi] != kind {
                continue;
            }
            let total = per_file[fi] + bonus[fi];
            if total >= MIN_SCORE {
                scored.push((fi, total));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 6. 결과 객체 생성
        let mut results = Vec::with_capacity(n.min(scored.len()));

        for (fi, score) in scored {
            if results.len() >= n {
                break;
            }
            let fid = self.uniq[fi];
            let item = match self.files.get(&fid) {
                Some(it) => it,
                None => continue,
            };

            let p = Path::new(&item.path);
            if !p.exists() {
                continue;
            }

            let best_idx = best_vector[fi];
            let mut match_type = match self.row_kind[best_idx] {
                VKIND_TEXT => "text".to_string(),
                VKIND_IMAGE => "image".to_string(),
                _ => "name".to_string(),
            };

            let mut snippet = String::new();
            if let Some((hit_row, tok)) = text_hit.get(&fi) {
                match_type = "text".to_string();
                let txt = Self::fts_text(&con, self.vec_ids[*hit_row]);
                snippet = Self::snippet(&txt, Some(tok), 180);
            } else if match_type == "text" {
                let txt = Self::fts_text(&con, self.vec_ids[best_idx]);
                snippet = Self::snippet(&txt, None, 180);
            }

            let dir = p.parent().map(|d| d.to_string_lossy().to_string()).unwrap_or_default();

            results.push(SearchResult {
                id: fid,
                name: item.name.clone(),
                path: item.path.clone(),
                dir,
                kind: item.kind.clone(),
                ext: item.ext.clone(),
                size: item.size,
                mtime: item.mtime,
                score: (score * 1000.0).round() / 1000.0,
                match_type,
                snippet,
                runnable: is_runnable_ext(&item.ext),
                thumb: item.kind == "image" && is_image_content_ext(&item.ext),
            });
        }

        Ok(results)
    }

    // ---------- 이미지로 이미지 검색 ----------
    pub fn search_image(&self, img_vec: &[f32], n: usize) -> Result<Vec<SearchResult>> {
        let n_files = self.uniq.len();
        if n_files == 0 || self.vectors.is_empty() || img_vec.len() != self.dim {
            return Ok(Vec::new());
        }

        let n_vecs = self.vec_ids.len();
        let img_vec = to_f16(img_vec);

        let raw: Vec<f32> = (0..n_vecs)
            .into_par_iter()
            .map(|i| {
                if self.row_kind[i] != VKIND_IMAGE {
                    -2.0f32
                } else {
                    let start = i * self.dim;
                    let end = start + self.dim;
                    let v = &self.vectors[start..end];
                    dot_f16(v, &img_vec)
                }
            })
            .collect();

        let mut scored: Vec<(usize, f32)> = Vec::new();

        for fi in 0..n_files {
            let s = self.starts[fi];
            let e = self.ends[fi];
            let mut max_val = -2.0f32;
            for i in s..e {
                if raw[i] > max_val {
                    max_val = raw[i];
                }
            }
            if max_val >= 0.25 {
                scored.push((fi, max_val));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut results = Vec::with_capacity(n.min(scored.len()));

        for (fi, val) in scored {
            if results.len() >= n {
                break;
            }
            let fid = self.uniq[fi];
            let item = match self.files.get(&fid) {
                Some(it) => it,
                None => continue,
            };

            let p = Path::new(&item.path);
            if !p.exists() {
                continue;
            }

            let pct = (val * 100.0).round() as i32;
            let (match_type, snippet) = if val >= 0.97 {
                ("same_image", format!("유사도 {}% · 같은 사진", pct))
            } else if val >= 0.85 {
                ("very_similar", format!("유사도 {}% · 매우 유사", pct))
            } else {
                ("image", format!("유사도 {}% · 비슷한 사진", pct))
            };

            let dir = p.parent().map(|d| d.to_string_lossy().to_string()).unwrap_or_default();

            results.push(SearchResult {
                id: fid,
                name: item.name.clone(),
                path: item.path.clone(),
                dir,
                kind: item.kind.clone(),
                ext: item.ext.clone(),
                size: item.size,
                mtime: item.mtime,
                score: (val * 1000.0).round() / 1000.0,
                match_type: match_type.to_string(),
                snippet,
                runnable: is_runnable_ext(&item.ext),
                thumb: true,
            });
        }

        Ok(results)
    }
}
