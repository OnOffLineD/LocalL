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

/// LocalL이 있는 드라이브 (`D:\`). 드라이브 문자가 없는 경로(UNC 등)면 빈 문자열.
fn current_drive(base: &Path) -> String {
    match base.to_str().and_then(|s| s.chars().next()) {
        Some(letter) if base.to_str().map_or(false, |s| s.contains(':')) => {
            format!("{}:\\", letter.to_ascii_uppercase())
        }
        _ => String::new(),
    }
}

/// 설정 항목 하나를 실제 경로로 푼다 (`<current_drive>`, `<project>`, `~/…`).
/// 풀 수 없으면(설치 드라이브를 모르거나 홈 폴더가 없으면) None.
pub fn resolve_root(raw: &str, base: &Path, cur_drive: &str, home: Option<&Path>) -> Option<String> {
    if raw == "<current_drive>" || raw == "<drive>" || raw == "%DRIVE%" {
        (!cur_drive.is_empty()).then(|| cur_drive.to_string())
    } else if raw == "<project>" || raw == "." {
        Some(base.to_string_lossy().to_string())
    } else if raw.starts_with("~/") || raw.starts_with("~\\") {
        home.map(|h| h.join(&raw[2..]).to_string_lossy().to_string())
    } else {
        Some(raw.to_string())
    }
}

/// 여러 항목을 풀어 겹치지 않게 모은다
fn resolve_all(raws: &[String]) -> Vec<String> {
    let base = base_dir();
    let cur_drive = current_drive(&base);
    let home = dirs::home_dir();
    let mut out: Vec<String> = Vec::new();
    for r in raws {
        if let Some(p) = resolve_root(r, &base, &cur_drive, home.as_deref()) {
            if !out.iter().any(|o| same_path(o, &p)) {
                out.push(p);
            }
        }
    }
    out
}

pub fn load_config(path: Option<&Path>) -> Result<Config> {
    let cfg_path = path.map(PathBuf::from).unwrap_or_else(default_config_path);
    let content = std::fs::read_to_string(&cfg_path)
        .with_context(|| format!("설정 파일을 읽을 수 없습니다: {:?}", cfg_path))?;

    let mut cfg: Config = serde_json::from_str(&content)
        .with_context(|| format!("설정 파일 JSON 파싱 실패: {:?}", cfg_path))?;

    cfg.roots = resolve_all(&cfg.roots);
    // 파이썬 색인기(locall/config.py)처럼 제외·클라우드 경로의 ~ 도 푼다
    cfg.exclude_paths = resolve_all(&cfg.exclude_paths);
    cfg.cloud_paths = resolve_all(&cfg.cloud_paths);

    cfg.exclude_dir_names = cfg.exclude_dir_names.into_iter().map(|s| s.to_lowercase()).collect();
    cfg.cloud_dir_names = cfg.cloud_dir_names.into_iter().map(|s| s.to_lowercase()).collect();

    Ok(cfg)
}

// ---------------------------------------------------------------------------
// 검색 폴더(roots)·제외 폴더(exclude_paths) 목록 편집 (설정 화면용)
// ---------------------------------------------------------------------------

/// 앱 안에서 편집하는 경로 목록
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum PathList {
    /// 검색할 폴더·드라이브 (`roots`)
    #[serde(rename = "roots")]
    Roots,
    /// 검색에서 뺄 폴더 (`exclude_paths`)
    #[serde(rename = "exclude")]
    Excludes,
}

impl PathList {
    pub fn key(self) -> &'static str {
        match self {
            PathList::Roots => "roots",
            PathList::Excludes => "exclude_paths",
        }
    }
}

/// 설정 파일에 적힌 그대로의 항목(`raw`)과 실제 경로(`path`, 풀 수 없으면 빈 문자열)
#[derive(Debug, Clone, Serialize)]
pub struct PathEntry {
    pub raw: String,
    pub path: String,
}

/// 설정 파일을 고치는 일은 한 번에 하나씩 (빠르게 여러 번 눌러도 서로 덮어쓰지 않게)
fn config_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// 비교용: `/`→`\`, 끝의 `\` 제거, 소문자
fn path_key(p: &str) -> String {
    p.replace('/', "\\").trim_end_matches('\\').to_lowercase()
}

fn same_path(a: &str, b: &str) -> bool {
    path_key(a) == path_key(b)
}

/// 사용자가 고른 경로를 설정에 적을 모양으로: 따옴표·공백 제거, `/`→`\`,
/// 끝의 `\` 제거 (단 드라이브 `D:\`는 그대로)
pub fn normalize_input_path(p: &str) -> Option<String> {
    let p = p.trim().trim_matches('"').trim().replace('/', "\\");
    let trimmed = p.trim_end_matches('\\');
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.len() == 2 && trimmed.ends_with(':') {
        return Some(format!("{}\\", trimmed.to_ascii_uppercase()));
    }
    Some(trimmed.to_string())
}

fn read_raw_list(content: &str, list: PathList) -> Result<Vec<String>> {
    let val: serde_json::Value = serde_json::from_str(content).context("설정 파일 JSON 파싱 실패")?;
    Ok(val
        .get(list.key())
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default())
}

fn read_config_text() -> Result<(PathBuf, String)> {
    let cfg_path = default_config_path();
    let content = std::fs::read_to_string(&cfg_path)
        .with_context(|| format!("설정 파일을 읽을 수 없습니다: {:?}", cfg_path))?;
    Ok((cfg_path, content))
}

/// 목록을 설정 파일에 적힌 그대로 읽고, 각 항목의 실제 경로를 함께 돌려준다
pub fn read_list(list: PathList) -> Result<Vec<PathEntry>> {
    let (_, content) = read_config_text()?;
    let base = base_dir();
    let cur_drive = current_drive(&base);
    let home = dirs::home_dir();
    Ok(read_raw_list(&content, list)?
        .into_iter()
        .map(|raw| {
            let path = resolve_root(&raw, &base, &cur_drive, home.as_deref()).unwrap_or_default();
            PathEntry { raw, path }
        })
        .collect())
}

/// 목록에 경로를 더한다. 이미 있으면(같은 실제 경로) false.
pub fn add_path(list: PathList, path: &str) -> Result<bool> {
    let path = normalize_input_path(path).ok_or_else(|| anyhow::anyhow!("빈 경로는 추가할 수 없습니다."))?;
    let _guard = config_lock();
    let entries = read_list(list)?;
    if entries.iter().any(|e| same_path(&e.path, &path) || same_path(&e.raw, &path)) {
        return Ok(false);
    }
    let mut raws: Vec<String> = entries.into_iter().map(|e| e.raw).collect();
    raws.push(path);
    save_list_locked(list, &raws)?;
    Ok(true)
}

/// 설정 파일에 적힌 그대로의 항목(`raw`)을 목록에서 뺀다. 없었으면 false.
pub fn remove_path(list: PathList, raw: &str) -> Result<bool> {
    let _guard = config_lock();
    let (_, content) = read_config_text()?;
    let mut raws = read_raw_list(&content, list)?;
    let before = raws.len();
    raws.retain(|r| r != raw);
    if raws.len() == before {
        return Ok(false);
    }
    save_list_locked(list, &raws)?;
    Ok(true)
}

/// 목록을 통째로 바꿔 적는다 (자리표시자 `~/…`, `<current_drive>`는 적힌 그대로 둔다)
pub fn save_list(list: PathList, items: &[String]) -> Result<()> {
    let _guard = config_lock();
    save_list_locked(list, items)
}

fn save_list_locked(list: PathList, items: &[String]) -> Result<()> {
    let (cfg_path, content) = read_config_text()?;
    // 배열 부분만 바꿔 서식·다른 항목을 그대로 둔다. 결과가 설정으로 읽히지 않으면 JSON으로 다시 쓴다.
    let new_content = match replace_array_text(&content, list.key(), items) {
        Some(text) if serde_json::from_str::<Config>(&text).is_ok() => text,
        _ => {
            let mut val: serde_json::Value = serde_json::from_str(&content).context("설정 파일 JSON 파싱 실패")?;
            let obj = val.as_object_mut().ok_or_else(|| anyhow::anyhow!("설정 파일 형식이 올바르지 않습니다."))?;
            obj.insert(list.key().into(), serde_json::json!(items));
            serde_json::to_string_pretty(&val)?
        }
    };
    write_atomic(&cfg_path, &new_content)
}

/// 중간에 끊겨도 설정 파일이 반쯤 쓰인 채 남지 않게, 옆 파일에 쓴 뒤 바꿔 끼운다
fn write_atomic(path: &Path, content: &str) -> Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, content).with_context(|| format!("설정 파일을 쓸 수 없습니다: {:?}", tmp))?;
    std::fs::rename(&tmp, path).with_context(|| format!("설정 파일을 바꿀 수 없습니다: {:?}", path))?;
    Ok(())
}

/// JSON 문자열을 건너뛴다. `i`는 여는 따옴표 위치, 닫는 따옴표 다음 위치를 돌려준다.
fn skip_json_string(b: &[u8], i: usize) -> Option<usize> {
    let mut j = i + 1;
    while j < b.len() {
        match b[j] {
            b'\\' => j += 2,
            b'"' => return Some(j + 1),
            _ => j += 1,
        }
    }
    None
}

fn skip_ws(b: &[u8], mut j: usize) -> usize {
    while j < b.len() && b[j].is_ascii_whitespace() {
        j += 1;
    }
    j
}

/// 최상위 객체의 `"key": [ ... ]` 배열 부분만 `items`로 바꾼 글을 돌려준다.
/// 나머지 글자(서식, 다른 항목)는 그대로 둔다. 키가 없거나 배열이 아니면 None.
pub fn replace_array_text(content: &str, key: &str, items: &[String]) -> Option<String> {
    let b = content.as_bytes();
    let mut depth = 0i32;
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'"' => {
                let end = skip_json_string(b, i)?;
                if depth == 1 && &content[i + 1..end - 1] == key {
                    let colon = skip_ws(b, end);
                    if b.get(colon) == Some(&b':') {
                        let open = skip_ws(b, colon + 1);
                        if b.get(open) == Some(&b'[') {
                            let close = find_array_end(b, open)?;
                            let arr = format_array(content, i, open, close, items);
                            return Some(format!("{}{}{}", &content[..open], arr, &content[close + 1..]));
                        }
                    }
                    return None;
                }
                i = end;
                continue;
            }
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    None
}

/// `open`의 `[`와 짝이 맞는 `]` 위치
fn find_array_end(b: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut j = open;
    while j < b.len() {
        match b[j] {
            b'"' => {
                j = skip_json_string(b, j)?;
                continue;
            }
            b'[' | b'{' => depth += 1,
            b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(j);
                }
            }
            _ => {}
        }
        j += 1;
    }
    None
}

/// 새 배열 글자. 키가 있는 줄의 들여쓰기에 맞추고, 원래 배열이 여러 줄이었으면 그 항목 들여쓰기를 따른다.
fn format_array(content: &str, key_pos: usize, open: usize, close: usize, items: &[String]) -> String {
    if items.is_empty() {
        return "[]".to_string();
    }
    let nl = if content.contains("\r\n") { "\r\n" } else { "\n" };
    let line_start = content[..key_pos].rfind('\n').map_or(0, |p| p + 1);
    let key_indent: String = content[line_start..key_pos].chars().take_while(|c| *c == ' ' || *c == '\t').collect();
    let inner = &content[open + 1..close];
    let item_indent = inner
        .find('\n')
        .map(|p| inner[p + 1..].chars().take_while(|c| *c == ' ' || *c == '\t').collect::<String>())
        .filter(|s| !s.is_empty() && inner.trim() != "")
        .unwrap_or_else(|| format!("{}  ", key_indent));
    let body: Vec<String> = items
        .iter()
        .map(|s| format!("{}{}", item_indent, serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())))
        .collect();
    format!("[{nl}{}{nl}{}]", body.join(&format!(",{nl}")), key_indent)
}

pub fn save_hotkey(hotkey: &str) -> Result<()> {
    let _guard = config_lock();
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

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = include_str!("../config.json");

    fn parse_list(text: &str, key: &str) -> Vec<String> {
        let v: serde_json::Value = serde_json::from_str(text).unwrap();
        v[key].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect()
    }

    #[test]
    fn replaces_only_the_array() {
        let items = vec!["~/Desktop".to_string(), "<current_drive>".to_string(), r"D:\자료 모음\[2024] 사진".to_string()];
        let out = replace_array_text(SAMPLE, "roots", &items).unwrap();
        assert_eq!(parse_list(&out, "roots"), items);
        // 다른 항목과 서식은 그대로
        assert!(serde_json::from_str::<Config>(&out).is_ok());
        let after = &out[out.find("\"include_kinds\"").unwrap()..];
        assert_eq!(after, &SAMPLE[SAMPLE.find("\"include_kinds\"").unwrap()..]);
        assert!(out.starts_with("{\n  \"hotkey\": \"ctrl+shift+space\",\n  \"roots\": [\n    \"~/Desktop\",\n"));
    }

    #[test]
    fn fills_and_empties_one_line_array() {
        let items = vec![r"C:\Users\me\AppData\Local\Temp".to_string()];
        let out = replace_array_text(SAMPLE, "exclude_paths", &items).unwrap();
        assert_eq!(parse_list(&out, "exclude_paths"), items);
        assert!(out.contains("  \"exclude_paths\": [\n    \"C:\\\\Users\\\\me\\\\AppData\\\\Local\\\\Temp\"\n  ],"));
        let back = replace_array_text(&out, "exclude_paths", &[]).unwrap();
        assert_eq!(back, SAMPLE);
    }

    #[test]
    fn ignores_key_text_inside_strings_and_nested_objects() {
        let text = "{\n  \"note\": \"\\\"roots\\\": [1]\",\n  \"inner\": { \"roots\": [\"x\"] },\n  \"roots\": [\"a]\", \"b\"]\n}";
        let out = replace_array_text(text, "roots", &["c".to_string()]).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["roots"], serde_json::json!(["c"]));
        assert_eq!(v["inner"]["roots"], serde_json::json!(["x"]));
        assert_eq!(v["note"], serde_json::json!("\"roots\": [1]"));
    }

    #[test]
    fn keeps_crlf_line_endings() {
        let text = SAMPLE.replace('\n', "\r\n");
        let out = replace_array_text(&text, "roots", &["a".to_string(), "b".to_string()]).unwrap();
        assert!(out.contains("\"roots\": [\r\n    \"a\",\r\n    \"b\"\r\n  ],"));
        assert!(!out.replace("\r\n", "").contains('\n'));
    }

    #[test]
    fn missing_key_or_not_array() {
        assert!(replace_array_text("{\"a\": 1}", "roots", &[]).is_none());
        assert!(replace_array_text("{\"roots\": \"x\"}", "roots", &[]).is_none());
    }

    #[test]
    fn normalizes_input_paths() {
        assert_eq!(normalize_input_path("d:").as_deref(), Some(r"D:\"));
        assert_eq!(normalize_input_path(r"d:\").as_deref(), Some(r"D:\"));
        assert_eq!(normalize_input_path(r#" "D:/자료/사진/" "#).as_deref(), Some(r"D:\자료\사진"));
        assert_eq!(normalize_input_path(r"\\nas\share\").as_deref(), Some(r"\\nas\share"));
        assert_eq!(normalize_input_path("  "), None);
        assert!(same_path(r"D:\Photos\", "d:/photos"));
    }

    #[test]
    fn resolves_placeholders() {
        let base = Path::new(r"E:\LocalL");
        let home = PathBuf::from("HOME");
        assert_eq!(resolve_root("<current_drive>", base, r"E:\", None).as_deref(), Some(r"E:\"));
        assert_eq!(resolve_root("<current_drive>", base, "", None), None);
        assert_eq!(resolve_root("<project>", base, r"E:\", None).as_deref(), Some(r"E:\LocalL"));
        assert_eq!(resolve_root("~/Desktop", base, "", None), None);
        assert_eq!(
            resolve_root("~/Desktop", base, "", Some(&home)),
            Some(home.join("Desktop").to_string_lossy().to_string())
        );
        assert_eq!(resolve_root(r"D:\data", base, "", None).as_deref(), Some(r"D:\data"));
        assert_eq!(current_drive(Path::new(r"e:\LocalL")), r"E:\");
        assert_eq!(current_drive(Path::new(r"\\nas\LocalL")), "");
    }
}
