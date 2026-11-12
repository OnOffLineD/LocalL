"""검색: 의미(임베딩) 점수에 파일명/본문 키워드 일치 가산점을 더해 파일 단위로 순위를 매긴다."""
import os
import re
import threading
import time

import numpy as np

from . import store
from .config import IMAGE_CONTENT_EXTS, RUNNABLE_EXTS
from .embed import Embedder

VKIND = {"name": 0, "text": 1, "image": 2}
VKIND_NAME = {v: k for k, v in VKIND.items()}
KEYWORD_BONUS = 0.06      # 검색어 단어가 전부 파일명/경로에 있을 때의 가산점
MIN_SCORE = 0.02          # 이보다 낮으면 관련 없는 것으로 보고 결과에서 뺀다
# 파일명·본문·이미지 벡터는 유사도 눈금이 서로 다르다(파일명이 가장 높게, 이미지가 가장 낮게 나옴).
# 그래서 질의마다 종류별 평균을 빼서 "평균보다 얼마나 더 닮았는지"로 비교한다.
# 벡터가 적을 때는 평균이 불안정하므로 아래 기본값 쪽으로 당긴다.
BASELINE_PRIOR = (0.62, 0.59, 0.56)   # name, text, image
PRIOR_WEIGHT = 20
TEXT_HIT_WEIGHT = 0.5     # 본문에서만 발견되면 절반
COMMON_TOKEN_HITS = 5000  # 이보다 많이 걸리는 단어는 변별력이 없으므로 무시


class Searcher:
    def __init__(self, cfg: dict, db_path):
        import torch

        self.torch = torch
        self.cfg, self.db_path = cfg, str(db_path)
        self.emb = Embedder(cfg, images=False)
        self.emb.query("예열")  # 첫 추론은 1초 가까이 걸리므로 미리 한 번 돌려 둔다
        self.lock = threading.Lock()
        self.version, self.checked = None, 0.0
        self._load()

    # ---------- 색인 적재 ----------
    def _load(self):
        torch = self.torch
        self.M = None
        self.files, self.counts = {}, {}
        if not os.path.exists(self.db_path):
            return
        con = store.connect(self.db_path, readonly=True)
        try:
            self.version = store.get_meta(con, "version")
            self.files = {
                r[0]: r[1:] for r in con.execute("SELECT id, path, name, ext, kind, size, mtime FROM files")
            }
            vids, fids, kinds, blobs = [], [], [], []
            for vid, fid, kind, vec in con.execute(
                "SELECT id, file_id, kind, vec FROM vectors ORDER BY file_id, id"
            ):
                if fid in self.files:
                    vids.append(vid)
                    fids.append(fid)
                    kinds.append(VKIND[kind])
                    blobs.append(vec)
        finally:
            con.close()
        for info in self.files.values():
            self.counts[info[3]] = self.counts.get(info[3], 0) + 1
        if not vids:
            return

        dim = self.cfg["dim"]
        mat = np.frombuffer(b"".join(blobs), dtype=np.float16).reshape(-1, dim).copy()
        fids = np.asarray(fids, dtype=np.int64)
        self.vec_ids = np.asarray(vids, dtype=np.int64)
        self.row_kind = np.asarray(kinds, dtype=np.uint8)
        self.uniq, self.row_file = np.unique(fids, return_inverse=True)  # 파일별로 연속 구간
        self.starts = np.searchsorted(fids, self.uniq, side="left")
        self.ends = np.searchsorted(fids, self.uniq, side="right")
        self.vid_order = np.argsort(self.vec_ids)
        self.vid_sorted = self.vec_ids[self.vid_order]
        self.file_kind = np.array([self.files[f][3] for f in self.uniq])

        device = self.emb.device
        m = torch.from_numpy(mat).to(device)
        self.M = m if device == "cuda" else m.float()  # 저장된 벡터끼리의 내적은 반정밀도로 충분
        self.row_file_t = torch.from_numpy(self.row_file.astype(np.int64)).to(device)
        self.row_kind_t = torch.from_numpy(self.row_kind.astype(np.int64)).to(device)
        self.kind_count = torch.bincount(self.row_kind_t, minlength=3).float()
        self.prior_t = torch.tensor(BASELINE_PRIOR, device=device)
        self._masks = {}

    def _maybe_reload(self):
        if time.time() - self.checked < 5:
            return
        self.checked = time.time()
        if not os.path.exists(self.db_path):
            return
        con = store.connect(self.db_path, readonly=True)
        try:
            version = store.get_meta(con, "version")
        finally:
            con.close()
        if version != self.version:
            self._load()

    def _kind_mask(self, kind):
        if kind not in self._masks:
            self._masks[kind] = self.torch.from_numpy(self.file_kind == kind).to(self.M.device)
        return self._masks[kind]

    # ---------- 키워드 가산점 ----------
    def _keyword(self, con, q):
        n_files = len(self.uniq)
        tokens = [t for t in re.split(r"\s+", q.lower()) if len(t) >= 2][:8]
        bonus = np.zeros(n_files, dtype=np.float32)
        text_hit = {}  # 파일 → (벡터 행, 단어): 본문 발췌용
        if not tokens:
            return bonus, text_hit
        for tok in tokens:
            w = np.zeros(n_files, dtype=np.float32)
            if len(tok) >= 3:
                phrase = '"' + tok.replace('"', '""') + '"'
                vids = np.fromiter(
                    (r[0] for r in con.execute(
                        "SELECT rowid FROM fts WHERE fts MATCH ? LIMIT ?", (phrase, COMMON_TOKEN_HITS + 1))),
                    dtype=np.int64,
                )
                if len(vids) == 0 or len(vids) > COMMON_TOKEN_HITS:
                    continue
                pos = np.clip(np.searchsorted(self.vid_sorted, vids), 0, len(self.vid_sorted) - 1)
                ok = self.vid_sorted[pos] == vids
                rows = self.vid_order[pos[ok]]
                is_name = self.row_kind[rows] == VKIND["name"]
                np.maximum.at(w, self.row_file[rows], np.where(is_name, 1.0, TEXT_HIT_WEIGHT))
                for r in rows[~is_name]:
                    text_hit.setdefault(int(self.row_file[r]), (int(r), tok))
            else:  # trigram 색인은 3글자부터라 2글자 단어는 파일명만 본다
                like = "%" + tok.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                fids = np.fromiter(
                    (r[0] for r in con.execute(
                        "SELECT id FROM files WHERE name LIKE ? ESCAPE '\\' LIMIT ?", (like, COMMON_TOKEN_HITS + 1))),
                    dtype=np.int64,
                )
                if len(fids) == 0 or len(fids) > COMMON_TOKEN_HITS:
                    continue
                pos = np.clip(np.searchsorted(self.uniq, fids), 0, n_files - 1)
                w[pos[self.uniq[pos] == fids]] = 1.0
            bonus += w
        return bonus / len(tokens) * KEYWORD_BONUS, text_hit

    @staticmethod
    def _snippet(text, token=None, width=180):
        text = re.sub(r"\s+", " ", text or "")
        at = text.lower().find(token) if token else -1
        start = max(0, at - width // 3) if at > 0 else 0
        cut = text[start:start + width]
        return ("…" if start else "") + cut + ("…" if start + width < len(text) else "")

    # ---------- 검색 ----------
    def search(self, q: str, kind: str = "all", n: int = 30) -> list[dict]:
        q = (q or "").strip()
        if not q:
            return []
        with self.lock:
            self._maybe_reload()
            if self.M is None:
                return []
            torch = self.torch
            qv = torch.from_numpy(self.emb.query(q)).to(self.M.device, self.M.dtype)
            raw = (self.M @ qv).float()
            n_files = len(self.uniq)
            # 종류별 평균을 뺀 점수
            sums = torch.zeros(3, device=raw.device).index_add_(0, self.row_kind_t, raw)
            base = (sums + self.prior_t * PRIOR_WEIGHT) / (self.kind_count + PRIOR_WEIGHT)
            sims = raw - base[self.row_kind_t]
            is_name = self.row_kind_t == VKIND["name"]
            zeros = torch.zeros(n_files, device=raw.device)
            name_part = zeros.scatter_reduce(0, self.row_file_t, sims.masked_fill(~is_name, 0.0), "amax")
            content_part = zeros.scatter_reduce(0, self.row_file_t, sims.masked_fill(is_name, 0.0), "amax")
            per_file = name_part + content_part   # 이름과 내용이 둘 다 맞으면 더 높게
            con = store.connect(self.db_path, readonly=True)
            try:
                bonus, text_hit = self._keyword(con, q)
                total = per_file + torch.from_numpy(bonus).to(per_file.device)
                if kind and kind != "all":
                    total = total.masked_fill(~self._kind_mask(kind), -2.0)
                vals, idxs = torch.topk(total, min(n * 2, n_files))
                results = []
                for val, fi in zip(vals.tolist(), idxs.tolist()):
                    if val < MIN_SCORE or len(results) >= n:
                        break
                    fid = int(self.uniq[fi])
                    path, name, ext, fkind, size, mtime = self.files[fid]
                    if not os.path.exists(path):
                        continue
                    s, e = int(self.starts[fi]), int(self.ends[fi])
                    best = s + int(torch.argmax(sims[s:e]))   # 가장 크게 기여한 벡터
                    match = VKIND_NAME[int(self.row_kind[best])]
                    snippet = ""
                    if fi in text_hit:
                        row, tok = text_hit[fi]
                        match, snippet = "text", self._snippet(self._fts_text(con, row), tok)
                    elif match == "text":
                        snippet = self._snippet(self._fts_text(con, best))
                    results.append({
                        "id": fid, "name": name, "path": path, "dir": os.path.dirname(path),
                        "kind": fkind, "ext": ext, "size": size, "mtime": mtime,
                        "score": round(val, 3), "match": match, "snippet": snippet,
                        "runnable": ext in RUNNABLE_EXTS,
                        "thumb": fkind == "image" and ext in IMAGE_CONTENT_EXTS,
                    })
                return results
            finally:
                con.close()

    def _fts_text(self, con, row):
        r = con.execute("SELECT text FROM fts WHERE rowid=?", (int(self.vec_ids[row]),)).fetchone()
        return r[0] if r else ""

    def info(self, file_id: int):
        """(path, name, ext, kind, size, mtime) 또는 None. 색인에 있는 파일만 열 수 있다."""
        return self.files.get(file_id)

    def stats(self) -> dict:
        return {"files": len(self.files), "vectors": 0 if self.M is None else int(self.M.shape[0]),
                "kinds": self.counts}
