use anyhow::{Context, Result};
use rusqlite::{params, Connection, OpenFlags};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS files(
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  ext TEXT, kind TEXT, size INTEGER, mtime REAL,
  name_done INTEGER DEFAULT 0,
  content_state INTEGER DEFAULT 0
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
"#;

pub fn connect(db_path: &Path, readonly: bool) -> Result<Connection> {
    if !readonly {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("DB 디렉터리 생성 실패: {:?}", parent))?;
        }
    }

    let flags = if readonly {
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX
    } else {
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE | OpenFlags::SQLITE_OPEN_NO_MUTEX
    };

    let con = Connection::open_with_flags(db_path, flags)
        .with_context(|| format!("DB 열기 실패: {:?}", db_path))?;

    con.busy_timeout(std::time::Duration::from_secs(30))?;
    if !readonly {
        con.pragma_update(None, "journal_mode", "WAL")?;
        con.pragma_update(None, "synchronous", "NORMAL")?;
    }

    if !readonly {
        con.execute_batch(SCHEMA)
            .with_context(|| "DB 스키마 초기화 실패")?;
    }

    Ok(con)
}

pub fn get_meta(con: &Connection, key: &str) -> Result<Option<String>> {
    let mut stmt = con.prepare_cached("SELECT value FROM meta WHERE key=?")?;
    let mut rows = stmt.query(params![key])?;
    if let Some(row) = rows.next()? {
        Ok(Some(row.get(0)?))
    } else {
        Ok(None)
    }
}

pub fn set_meta(con: &Connection, key: &str, value: &str) -> Result<()> {
    con.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
        params![key, value],
    )?;
    Ok(())
}

pub fn bump_version(con: &Connection) -> Result<()> {
    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    set_meta(con, "version", &now_ns.to_string())?;
    Ok(())
}

pub fn drop_vectors(con: &Connection, file_id: i64, kind: Option<&str>) -> Result<()> {
    if let Some(k) = kind {
        con.execute(
            "DELETE FROM fts WHERE rowid IN (SELECT id FROM vectors WHERE file_id=? AND kind=?)",
            params![file_id, k],
        )?;
        con.execute(
            "DELETE FROM vectors WHERE file_id=? AND kind=?",
            params![file_id, k],
        )?;
    } else {
        con.execute(
            "DELETE FROM fts WHERE rowid IN (SELECT id FROM vectors WHERE file_id=?)",
            params![file_id],
        )?;
        con.execute(
            "DELETE FROM vectors WHERE file_id=?",
            params![file_id],
        )?;
    }
    Ok(())
}

pub fn add_vector(con: &Connection, file_id: i64, kind: &str, vec: &[u8], text: Option<&str>) -> Result<i64> {
    con.execute(
        "INSERT INTO vectors(file_id, kind, vec) VALUES(?, ?, ?)",
        params![file_id, kind, vec],
    )?;
    let vec_id = con.last_insert_rowid();
    if let Some(t) = text {
        con.execute(
            "INSERT INTO fts(rowid, text) VALUES(?, ?)",
            params![vec_id, t],
        )?;
    }
    Ok(vec_id)
}

pub fn remove_file(con: &Connection, file_id: i64) -> Result<()> {
    drop_vectors(con, file_id, None)?;
    con.execute("DELETE FROM files WHERE id=?", params![file_id])?;
    Ok(())
}
