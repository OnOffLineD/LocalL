"""SQLite 색인 저장소.

files   : 파일/폴더 한 개당 한 줄
vectors : 임베딩 (kind = name | text | image), 파일 하나에 여러 개
fts     : 키워드 검색용. rowid가 vectors.id와 같다 (name → 전체 경로, text → 본문 조각)
"""
import os
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS files(
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  ext TEXT, kind TEXT, size INTEGER, mtime REAL,
  name_done INTEGER DEFAULT 0,
  content_state INTEGER DEFAULT 0   -- 0 대기, 1 완료, 2 대상 아님, -1 실패
);
CREATE TABLE IF NOT EXISTS vectors(
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL,
  kind TEXT NOT NULL,
  vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS vectors_file ON vectors(file_id);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text, tokenize='trigram');
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


def connect(db_path, readonly=False) -> sqlite3.Connection:
    db_path = str(db_path)
    if not readonly:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    if not readonly:
        con.executescript(SCHEMA)
    return con


def get_meta(con, key, default=None):
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(con, key, value):
    con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, str(value)))


def bump_version(con):
    """검색 서버가 색인이 바뀐 것을 알아채도록."""
    set_meta(con, "version", time.time_ns())
    con.commit()


def drop_vectors(con, file_id, kind=None):
    where, args = "file_id=?", [file_id]
    if kind:
        where += " AND kind=?"
        args.append(kind)
    con.execute(f"DELETE FROM fts WHERE rowid IN (SELECT id FROM vectors WHERE {where})", args)
    con.execute(f"DELETE FROM vectors WHERE {where}", args)


def add_vector(con, file_id, kind, vec, text=None):
    cur = con.execute(
        "INSERT INTO vectors(file_id, kind, vec) VALUES(?, ?, ?)", (file_id, kind, vec.tobytes())
    )
    if text:
        con.execute("INSERT INTO fts(rowid, text) VALUES(?, ?)", (cur.lastrowid, text))


def remove_file(con, file_id):
    drop_vectors(con, file_id)
    con.execute("DELETE FROM files WHERE id=?", (file_id,))
