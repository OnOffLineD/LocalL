use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub hotkey: String,
    pub roots: Vec<String>,
    pub include_kinds: Vec<String>,
    pub content_kinds: Vec<String>,
    pub exclude_dir_names: Vec<String>,
    pub exclude_paths: Vec<String>,
    pub cloud_dir_names: Vec<String>,
    pub cloud_paths: Vec<String>,
    #[serde(default = "default_true")]
    pub skip_hidden: bool,
    #[serde(default = "default_chunk_chars")]
    pub chunk_chars: usize,
    #[serde(default = "default_max_chunks")]
    pub max_chunks_per_file: usize,
    #[serde(default = "default_max_doc_mb")]
    pub max_doc_mb: u64,
    #[serde(default = "default_min_image_kb")]
    pub min_image_kb: u64,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_dim")]
    pub dim: usize,
}

fn default_true() -> bool { true }
fn default_chunk_chars() -> usize { 1200 }
fn default_max_chunks() -> usize { 16 }
fn default_max_doc_mb() -> u64 { 200 }
fn default_min_image_kb() -> u64 { 8 }
fn default_model() -> String { "google/embeddinggemma-2".into() }
fn default_dim() -> usize { 512 }

pub fn base_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            // target/debug 또는 target/release 안에 있는 경우 상위로 이동
            if parent.ends_with("debug") || parent.ends_with("release") {
                if let Some(p2) = parent.parent().and_then(|p| p.parent()) {
                    return p2.to_path_buf();
                }
            }
            return parent.to_path_buf();
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

pub fn default_config_path() -> PathBuf {
    base_dir().join("config.json")
}

pub fn default_db_path() -> PathBuf {
    base_dir().join("data").join("index.db")
}

pub fn default_models_path() -> PathBuf {
    base_dir().join("models")
}

pub static KIND_EXTS: &[(&str, &str)] = &[
    ("doc", "pdf docx doc pptx ppt xlsx xls hwp hwpx txt md rtf csv odt epub"),
    ("image", "jpg jpeg png webp bmp gif tif tiff heic psd"),
    ("video", "mp4 mkv avi mov wmv webm m4v"),
    ("audio", "mp3 wav flac m4a ogg aac wma"),
    ("archive", "zip 7z rar tar gz iso"),
    ("app", "exe msi lnk bat cmd url apk"),
    ("code", "py js ts tsx jsx java c cpp h hpp cs go rs rb php html css json yaml yml toml xml sql sh ipynb"),
];

pub fn ext_to_kind_map() -> &'static HashMap<String, &'static str> {
    static MAP: OnceLock<HashMap<String, &'static str>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut map = HashMap::new();
        for &(kind, exts) in KIND_EXTS {
            for ext in exts.split_whitespace() {
                map.insert(ext.to_lowercase(), kind);
            }
        }
        map
    })
}

pub fn kind_label(kind: &str) -> &'static str {
    match kind {
        "dir" => "폴더",
        "doc" => "문서",
        "image" => "이미지",
        "video" => "동영상",
        "audio" => "오디오",
        "archive" => "압축 파일",
        "app" => "프로그램",
        "code" => "코드",
        _ => "기타",
    }
}

pub static TEXT_EXTS: &[&str] = &[
    "txt", "md", "csv", "py", "js", "ts", "tsx", "jsx", "java", "c", "cpp", "h", "hpp",
    "cs", "go", "rs", "rb", "php", "html", "css", "json", "yaml", "yml", "toml", "xml", "sql", "sh", "ipynb"
];

pub static PARSED_DOC_EXTS: &[&str] = &["pdf", "docx", "pptx", "xlsx", "hwp", "hwpx"];
pub static IMAGE_CONTENT_EXTS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "gif", "tif", "tiff"];
pub static RUNNABLE_EXTS: &[&str] = &["exe", "msi", "lnk", "bat", "cmd", "url"];

pub fn is_text_ext(ext: &str) -> bool {
    TEXT_EXTS.contains(&ext)
}

pub fn is_parsed_doc_ext(ext: &str) -> bool {
    PARSED_DOC_EXTS.contains(&ext)
}

pub fn is_image_content_ext(ext: &str) -> bool {
    IMAGE_CONTENT_EXTS.contains(&ext)
}

pub fn is_runnable_ext(ext: &str) -> bool {
    RUNNABLE_EXTS.contains(&ext)
}

pub fn has_content(kind: &str, ext: &str, cfg: &Config) -> bool {
    if !cfg.content_kinds.iter().any(|k| k == kind) {
        return false;
    }
    if kind == "image" {
        return is_image_content_ext(ext);
    }
    is_text_ext(ext) || is_parsed_doc_ext(ext)
}

pub fn load_config(path: Option<&Path>) -> Result<Config> {
    let base = base_dir();
    let cfg_path = path.map(PathBuf::from).unwrap_or_else(default_config_path);
    let content = std::fs::read_to_string(&cfg_path)
        .with_context(|| format!("설정 파일을 읽을 수 없습니다: {:?}", cfg_path))?;

    let mut cfg: Config = serde_json::from_str(&content)
        .with_context(|| format!("설정 파일 JSON 파싱 실패: {:?}", cfg_path))?;

    let cur_drive = match base.to_str().and_then(|s| s.chars().next()) {
        Some(letter) if base.to_str().map_or(false, |s| s.contains(':')) => {
            format!("{}:\\", letter.to_ascii_uppercase())
        }
        _ => String::new(),
    };

    let mut resolved_roots = Vec::new();
    let home = dirs::home_dir();

    for r in &cfg.roots {
        if r == "<current_drive>" || r == "<drive>" || r == "%DRIVE%" {
            if !cur_drive.is_empty() && !resolved_roots.contains(&cur_drive) {
                resolved_roots.push(cur_drive.clone());
            }
        } else if r == "<project>" || r == "." {
            let s = base.to_string_lossy().to_string();
            if !resolved_roots.contains(&s) {
                resolved_roots.push(s);
            }
        } else if r.starts_with("~/") || r.starts_with("~\\") {
            if let Some(h) = &home {
                let expanded = h.join(&r[2..]);
                let s = expanded.to_string_lossy().to_string();
                if !resolved_roots.contains(&s) {
                    resolved_roots.push(s);
                }
            }
        } else {
            resolved_roots.push(r.clone());
        }
    }
    cfg.roots = resolved_roots;

    cfg.exclude_dir_names = cfg.exclude_dir_names.into_iter().map(|s| s.to_lowercase()).collect();
    cfg.cloud_dir_names = cfg.cloud_dir_names.into_iter().map(|s| s.to_lowercase()).collect();

    Ok(cfg)
}

pub fn save_hotkey(hotkey: &str) -> Result<()> {
    let cfg_path = default_config_path();
    let content = std::fs::read_to_string(&cfg_path)
        .with_context(|| format!("설정 파일을 읽을 수 없습니다: {:?}", cfg_path))?;

    // "hotkey": "..." 패턴 직접 치환 (서식 및 주석 완벽 보존)
    if let Some(pos) = content.find("\"hotkey\"") {
        let rest = &content[pos..];
        if let Some(colon) = rest.find(':') {
            let after_colon = &rest[colon + 1..];
            if let Some(first_quote) = after_colon.find('"') {
                let after_first = &after_colon[first_quote + 1..];
                if let Some(second_quote) = after_first.find('"') {
                    let start_idx = pos + colon + 1 + first_quote + 1;
                    let end_idx = start_idx + second_quote;
                    let mut new_content = String::with_capacity(content.len() + 16);
                    new_content.push_str(&content[..start_idx]);
                    new_content.push_str(hotkey);
                    new_content.push_str(&content[end_idx..]);
                    std::fs::write(&cfg_path, new_content)?;
                    return Ok(());
                }
            }
        }
    }

    // fallback: JSON 파싱 후 저장
    if let Ok(mut val) = serde_json::from_str::<serde_json::Value>(&content) {
        if let Some(obj) = val.as_object_mut() {
            obj.insert("hotkey".into(), serde_json::Value::String(hotkey.to_string()));
            let formatted = serde_json::to_string_pretty(&val)?;
            std::fs::write(&cfg_path, formatted)?;
            return Ok(());
        }
    }

    Err(anyhow::anyhow!("hotkey 값을 저장할 수 없습니다."))
}
