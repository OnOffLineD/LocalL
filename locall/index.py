"""색인 실행:  python -m locall.index [--root 폴더] [--scan-only] [--limit N]

중간에 끊어도(Ctrl+C) 다시 실행하면 이어서 진행한다. 바뀐 파일만 다시 처리한다.
"""
import argparse
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from . import config, scan, store
from .config import KIND_LABEL
from .embed import doc_text
from .extract import chunk_text, extract_text, load_image, pdf_first_page_image

MB = 1024 * 1024


def log(msg):
    print(msg, flush=True)


def name_text(path: str, name: str, kind: str) -> str:
    """파일명과 위치만으로 만드는 설명문 (내용을 못 읽는 파일도 찾을 수 있게)."""
    stem = name if kind == "dir" else os.path.splitext(name)[0]
    stem = re.sub(r"[_\-.]+", " ", stem)
    parents = [p for p in os.path.dirname(path).replace(":", "").split(os.sep) if p][-4:]
    return doc_text(name, f"{KIND_LABEL.get(kind, '파일')} {stem}. 위치: {' > '.join(parents)}")


def scan_phase(con, cfg, roots, prune_outside=False):
    """prune_outside: 설정의 검색 폴더 전체로 도는 중이면 True. 그러면 어느 검색 폴더에도 속하지 않는
    항목(검색 폴더 목록에서 뺀 폴더의 파일)도 색인에서 지운다. --root로 일부만 돌 때는 False."""
    existing = {r[1]: r for r in con.execute("SELECT id, path, size, mtime FROM files")}
    seen, report, kinds = set(), {}, Counter()
    new = changed = 0
    t0 = time.time()
    for path, name, is_dir, ext, kind, size, mtime in scan.walk(roots, cfg, report):
        kinds[kind] += 1
        state = 0 if config.has_content(kind, ext, cfg) else 2
        row = existing.get(path)
        if row is None:
            con.execute(
                "INSERT OR IGNORE INTO files(path, name, ext, kind, size, mtime, content_state)"
                " VALUES(?, ?, ?, ?, ?, ?, ?)",
                (path, name, ext, kind, size, mtime, state),
            )
            new += 1
        else:
            seen.add(row[0])
            if not is_dir and (row[2] != size or abs((row[3] or 0) - mtime) > 1):
                store.drop_vectors(con, row[0], "text")
                store.drop_vectors(con, row[0], "image")
                con.execute(
                    "UPDATE files SET size=?, mtime=?, content_state=? WHERE id=?",
                    (size, mtime, state, row[0]),
                )
                changed += 1
    # 훑은 루트 아래에서 사라진(또는 이제 제외되는) 항목 정리
    cloud_roots = {os.path.normcase(p) for p in report["cloud"]}
    live = [
        os.path.normcase(r).rstrip("\\") + "\\"
        for r in roots
        if os.path.isdir(r) or os.path.normcase(r) in cloud_roots
    ]
    # 지금 없는 검색 폴더(뽑힌 USB 등)도 포함: 그 파일은 다시 꽂을 때를 위해 남겨 둔다
    all_roots = tuple(os.path.normcase(r).rstrip("\\") + "\\" for r in roots)
    prune_outside = prune_outside and bool(all_roots)
    removed = outside = 0
    for path, row in existing.items():
        if row[0] in seen:
            continue
        path_n = os.path.normcase(path) + "\\"
        if path_n.startswith(tuple(live)):
            store.remove_file(con, row[0])
            removed += 1
        elif prune_outside and not path_n.startswith(all_roots):
            store.remove_file(con, row[0])
            outside += 1
    con.commit()

    extra = f" · 검색 폴더 밖 정리 {outside:,}" if outside else ""
    log(f"[훑기] {sum(kinds.values()):,}개 확인 ({time.time() - t0:.0f}초) · 새로 {new:,} · 변경 {changed:,} · 삭제 {removed:,}{extra}")
    log("       " + " · ".join(f"{KIND_LABEL.get(k, k)} {v:,}" for k, v in kinds.most_common()))
    for p in report["cloud"]:
        log(f"       클라우드라서 건너뜀: {p}")
    if report["denied"]:
        log(f"       접근 권한이 없어 건너뛴 폴더 {report['denied']:,}개")


def name_phase(con, emb):
    rows = con.execute("SELECT id, path, name, kind FROM files WHERE name_done=0").fetchall()
    if not rows:
        return
    prog = Progress(con, "파일명", len(rows))
    for i in range(0, len(rows), 512):
        batch = rows[i:i + 512]
        vecs = emb.docs([name_text(p, n, k) for _, p, n, k in batch], batch_size=128)
        for (fid, path, _, _), vec in zip(batch, vecs):
            store.drop_vectors(con, fid, "name")
            store.add_vector(con, fid, "name", vec, path)
            con.execute("UPDATE files SET name_done=1 WHERE id=?", (fid,))
        con.commit()
        prog.tick(min(i + 512, len(rows)))
    store.bump_version(con)


def _set_state(con, fid, state):
    con.execute("UPDATE files SET content_state=? WHERE id=?", (state, fid))


class Progress:
    def __init__(self, con, label, total):
        self.con, self.label, self.total = con, label, total
        self.t0 = self.last = self.last_bump = time.time()

    def tick(self, done, extra=""):
        now = time.time()
        if now - self.last >= 5 or done == self.total:
            self.last = now
            log(f"[{self.label}] {done:,}/{self.total:,} {extra}({now - self.t0:.0f}초)")
        if now - self.last_bump >= 60:
            self.last_bump = now
            store.bump_version(self.con)


def docs_phase(con, emb, cfg, rows):
    """문서 본문 → 조각 → 임베딩. 글자가 없는 PDF는 목록으로 돌려준다."""
    buf, buffered, scanned, n_chunks = [], 0, [], 0
    prog = Progress(con, "문서", len(rows))

    def flush():
        nonlocal buffered
        if not buf:
            return
        texts = [doc_text(name, c) for _, name, chunks in buf for c in chunks]
        try:
            vecs = emb.docs(texts, batch_size=16)
            i = 0
            for fid, _, chunks in buf:
                for c in chunks:
                    store.add_vector(con, fid, "text", vecs[i], c)
                    i += 1
                _set_state(con, fid, 1)
        except Exception:
            for fid, name, chunks in buf:
                try:
                    f_texts = [doc_text(name, c) for c in chunks]
                    f_vecs = emb.docs(f_texts, batch_size=16)
                    for c, v in zip(chunks, f_vecs):
                        store.add_vector(con, fid, "text", v, c)
                    _set_state(con, fid, 1)
                except Exception:
                    _set_state(con, fid, -1)
        con.commit()
        buf.clear()
        buffered = 0

    for done, (fid, path, name, ext, size) in enumerate(rows, 1):
        if size > cfg["max_doc_mb"] * MB:
            _set_state(con, fid, 2)
            continue
        try:
            text = extract_text(path, ext)
        except Exception:
            _set_state(con, fid, -1)
            continue
        chunks = chunk_text(text, cfg["chunk_chars"], cfg["max_chunks_per_file"])
        if chunks:
            buf.append((fid, name, chunks))
            buffered += len(chunks)
            n_chunks += len(chunks)
            if buffered >= 64:
                flush()
        elif ext == "pdf":
            scanned.append((fid, path))
        else:
            _set_state(con, fid, 1)
        prog.tick(done, f"· 조각 {n_chunks:,} ")
    flush()
    con.commit()
    return scanned


def images_phase(con, emb, cfg, rows, loader, label):
    """rows: (file_id, path). loader(path) → PIL 이미지 또는 None(대상 아님)."""
    prog = Progress(con, label, len(rows))

    def safe_load(item):
        try:
            return loader(item[1])
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=6) as pool:
        for i in range(0, len(rows), 32):
            batch = rows[i:i + 32]
            loaded = list(pool.map(safe_load, batch))
            good = [(fid, im) for (fid, _), im in zip(batch, loaded) if im is not None and im is not False]
            for (fid, _), im in zip(batch, loaded):
                if im is None:
                    _set_state(con, fid, 2)
                elif im is False:
                    _set_state(con, fid, -1)
            if good:
                try:
                    vecs = emb.images([im for _, im in good], batch_size=8)
                except Exception as e:
                    log(f"       이미지 묶음 실패, 건너뜀: {type(e).__name__}")
                    for fid, _ in good:
                        _set_state(con, fid, -1)
                    vecs = []
                for (fid, _), vec in zip(good, vecs):
                    store.add_vector(con, fid, "image", vec)
                    _set_state(con, fid, 1)
            con.commit()
            prog.tick(min(i + 32, len(rows)))


def content_phase(con, emb, cfg, limit):
    q = "SELECT id, path, name, ext, size FROM files WHERE content_state=0 AND kind{} ORDER BY id"
    docs = con.execute(q.format("!='image'")).fetchall()
    imgs = con.execute(q.format("='image'")).fetchall()
    small = [r for r in imgs if r[4] < cfg["min_image_kb"] * 1024]
    for r in small:
        _set_state(con, r[0], 2)
    con.commit()
    imgs = [r for r in imgs if r[4] >= cfg["min_image_kb"] * 1024]
    if limit:
        docs, imgs = docs[:limit], imgs[:limit]
    log(f"[내용] 처리할 문서 {len(docs):,}개 · 이미지 {len(imgs):,}개")

    if docs:
        scanned = docs_phase(con, emb, cfg, docs)
        if scanned:
            images_phase(con, emb, cfg, scanned, pdf_first_page_image, "스캔 PDF")
    if imgs:
        images_phase(con, emb, cfg, [(r[0], r[1]) for r in imgs], load_image, "이미지")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="LocalL 색인")
    ap.add_argument("--root", action="append", help="설정의 roots 대신 이 폴더만 색인 (여러 번 지정 가능)")
    ap.add_argument("--db", default=str(config.DEFAULT_DB))
    ap.add_argument("--scan-only", action="store_true", help="파일 목록만 훑고 임베딩은 하지 않음")
    ap.add_argument("--limit", type=int, default=0, help="내용 임베딩할 파일 수 제한 (시험용)")
    args = ap.parse_args()

    cfg = config.load()
    roots = [os.path.normpath(r) for r in args.root] if args.root else cfg["roots"]
    con = store.connect(args.db)
    signature = f"{cfg['model']}@{cfg['dim']}"
    saved = store.get_meta(con, "signature")
    if saved and saved != signature:
        sys.exit(f"색인이 다른 모델/차원({saved})으로 만들어졌습니다. {args.db} 를 지우고 다시 실행하세요.")
    store.set_meta(con, "signature", signature)
    con.commit()

    try:
        scan_phase(con, cfg, roots, prune_outside=not args.root)
        if args.scan_only:
            pending = con.execute(
                "SELECT kind, COUNT(*) FROM files WHERE content_state=0 GROUP BY kind"
            ).fetchall()
            log("[예상] 내용 임베딩 대기: " + (" · ".join(f"{KIND_LABEL.get(k, k)} {n:,}" for k, n in pending) or "없음"))
            return
        log("[모델] 불러오는 중…")
        from .embed import Embedder

        emb = Embedder(cfg, images=True)
        name_phase(con, emb)
        content_phase(con, emb, cfg, args.limit)
        log("[완료]")
    except KeyboardInterrupt:
        log("\n중단했습니다. 다시 실행하면 이어서 진행합니다.")
    finally:
        con.commit()
        store.bump_version(con)
        con.close()


if __name__ == "__main__":
    main()
