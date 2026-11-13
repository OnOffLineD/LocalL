#![windows_subsystem = "windows"]

use anyhow::{anyhow, Result};
use locall::config::{base_dir, default_db_path, is_image_content_ext, load_config, Config, PathList};
use locall::embed::{find_python, Embedder, ImageWorker};
use locall::hotkey::HotkeyManager;
use locall::search::{SearchResult, Searcher};
use locall::update::{self, Release};
use locall::autostart;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};
use tao::dpi::{LogicalPosition, LogicalSize};
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
use tao::window::WindowBuilder;
use wry::http::{header, Response};
use wry::WebViewBuilder;

const FETCH: usize = 28; // 검색해 오는 결과 수
const MODEL_LOADING: &str = "AI 모델을 불러오는 중… 그동안은 이름·본문에 검색어가 들어간 파일만 찾습니다.";
const IMAGE_MODEL_LOADING: &str = "사진 검색 모델을 불러오는 중… (처음 한 번, 20초 안팎) 준비되면 자동으로 다시 찾습니다.";
const NO_INDEX: &str = "아직 색인이 없습니다. ⚙ 설정 → 색인 만들기 / 갱신을 실행하세요.";

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum IpcMessage {
    #[serde(rename = "search")]
    Search {
        query: String,
        kind: String,
        #[serde(default)]
        seq: u64,
    },
    #[serde(rename = "search_image")]
    SearchImage {
        path: String,
        #[serde(default)]
        seq: u64,
    },
    #[serde(rename = "paste_image")]
    PasteImage { data: String },
    #[serde(rename = "open")]
    Open { path: String, reveal: bool },
    #[serde(rename = "close")]
    Close,
    #[serde(rename = "pick_image")]
    PickImage,
    #[serde(rename = "run_index")]
    RunIndex,
    #[serde(rename = "update_hotkey")]
    UpdateHotkey { hotkey: String },
    #[serde(rename = "quit")]
    Quit,
    #[serde(rename = "get_config")]
    GetConfig,
    #[serde(rename = "set_autostart")]
    SetAutostart { enabled: bool },
    #[serde(rename = "check_update")]
    CheckUpdate,
    #[serde(rename = "install_update")]
    InstallUpdate,
    #[serde(rename = "get_folders")]
    GetFolders,
    /// 폴더 선택 대화상자로 골라 목록에 더한다
    #[serde(rename = "pick_folder")]
    PickFolder { list: PathList },
    /// 경로를 바로 더한다 (드라이브 칩)
    #[serde(rename = "add_path")]
    AddPath { list: PathList, path: String },
    /// 설정 파일에 적힌 그대로의 항목(raw)을 뺀다
    #[serde(rename = "remove_path")]
    RemovePath { list: PathList, raw: String },
}

/// 검색 폴더·제외 폴더 편집 요청
#[derive(Debug)]
enum FolderOp {
    Get,
    Pick(PathList),
    Add(PathList, String),
    Remove(PathList, String),
}

#[derive(Debug)]
enum UserEvent {
    Render { seq: u64, json: String },
    Status(String),
    /// 사진 검색을 시작한다 (경로, 검색창에 보일 이름)
    ImageChosen { path: String, label: String },
    PickImage,
    Hide,
    Quit,
    RunIndex,
    UpdateHotkey(String),
    GetConfig,
    SetAutostart(bool),
    CheckUpdate,
    InstallUpdate,
    /// 새 버전 확인 결과 (수동 확인이면 오류도 보인다)
    UpdateChecked { result: Result<Option<Release>, String>, manual: bool },
    UpdateProgress(String),
    UpdateInstalled,
    UpdateFailed(String),
    Folders(FolderOp),
    /// 폴더 목록 화면 (바뀐 게 있었는지와 함께)
    FoldersView { view: serde_json::Value, changed: bool },
}

/// 설정 메뉴의 "업데이트" 항목에 보일 상태
#[derive(Clone, Serialize)]
struct UpdateView {
    version: &'static str,
    /// idle | checking | latest | available | downloading | error
    status: &'static str,
    latest: Option<String>,
    message: String,
}

/// 백그라운드에서 새 버전을 확인한다 (`delay` 뒤에)
fn spawn_update_check(proxy: EventLoopProxy<UserEvent>, delay: Duration, manual: bool) {
    thread::spawn(move || {
        thread::sleep(delay);
        let result = update::check().map_err(|e| format!("{:#}", e));
        let _ = proxy.send_event(UserEvent::UpdateChecked { result, manual });
    });
}

#[derive(Clone)]
enum Query {
    Text { query: String, kind: String },
    Image { path: String },
}

enum Job {
    Search { seq: u64, query: Query },
    /// 백그라운드에서 모델을 다 올렸거나(Ok) 실패했다(Err)
    Embedder(Result<Embedder, String>),
    /// 사진 검색용 파이썬 작업자를 다 띄웠거나(Ok) 실패했다(Err)
    ImageWorker(Result<ImageWorker, String>),
    /// 처음 실행 때 모델을 받는 중인 상황 (상태줄에 보인다)
    Download(String),
}

/// 검색하지 않을 때의 상태줄. 모델을 받는 중이면 색인이 없어도 받는 상황을 같이 보인다.
fn idle_status(has_index: bool, note: &str, downloading: bool) -> String {
    match (has_index, downloading) {
        (true, _) => note.to_string(),
        (false, true) => format!("{}  {}", note, NO_INDEX),
        (false, false) => NO_INDEX.to_string(),
    }
}

/// 배포본엔 모델이 없다: 처음 실행 때 받는다. 받는 상황은 검색 스레드로 보내 상태줄에 띄운다.
fn fetch_models(tx: &Sender<Job>) -> Result<()> {
    use locall::download::{ensure_models, human, models_missing, Progress};
    if !models_missing() {
        return Ok(());
    }
    let mut last = std::time::Instant::now() - std::time::Duration::from_secs(10);
    ensure_models(&mut |p| {
        let text = match p {
            Progress::Waiting => "AI 모델을 색인기가 받고 있습니다. 끝나면 이어서 불러옵니다…".to_string(),
            Progress::Checking { .. } => "AI 모델 받은 파일을 확인하는 중…".to_string(),
            Progress::Bytes { done, total, .. } => {
                if last.elapsed() < std::time::Duration::from_millis(500) && done < total {
                    return;
                }
                last = std::time::Instant::now();
                format!(
                    "AI 모델 받는 중 (처음 한 번) {} / {} · {}%  그동안은 이름·본문에 검색어가 들어간 파일만 찾습니다.",
                    human(done),
                    human(total),
                    done * 100 / total.max(1)
                )
            }
        };
        let _ = tx.send(Job::Download(text));
    })?;
    let _ = tx.send(Job::Download("AI 모델을 다 받았습니다. 불러오는 중…".to_string()));
    Ok(())
}

fn wide(s: &str) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    std::ffi::OsStr::new(s).encode_wide().chain(Some(0)).collect()
}

fn is_image_path(p: &Path) -> bool {
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    p.is_absolute() && is_image_content_ext(&ext) && p.is_file()
}

fn decode_base64(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in text.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            _ => return None,
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

/// URL의 %XX를 풀어 원래 글자로 (미리보기 요청의 파일 경로용)
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
        match b[i] {
            b'%' if i + 2 < b.len() => match (hex(b[i + 1]), hex(b[i + 2])) {
                (Some(h), Some(l)) => {
                    out.push(h * 16 + l);
                    i += 3;
                    continue;
                }
                _ => out.push(b'%'),
            },
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// 미리보기를 만드는 전용 스레드 (사진 해독이 무거워서 창·검색 스레드와 나눈다)
fn thumb_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .thread_name(|i| format!("thumb-{}", i))
            .build()
            .expect("미리보기 스레드")
    })
}

/// 창 안의 화면(index.html과 assets/ui의 아이콘들)
fn serve_ui(path: &str, html_content: &'static str, base_ui_dir: &Path) -> Response<Cow<'static, [u8]>> {
    if path == "/" || path == "/index.html" {
        return Response::builder()
            .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
            .body(Cow::Borrowed(html_content.as_bytes()))
            .unwrap();
    }

    let asset_name = path.trim_start_matches("/assets/").trim_start_matches('/');
    let file_path = base_ui_dir.join(asset_name);

    if file_path.exists() {
        if let Ok(bytes) = fs::read(&file_path) {
            let mime = if asset_name.ends_with(".png") {
                "image/png"
            } else if asset_name.ends_with(".jpg") || asset_name.ends_with(".jpeg") {
                "image/jpeg"
            } else if asset_name.ends_with(".svg") {
                "image/svg+xml"
            } else {
                "application/octet-stream"
            };

            return Response::builder()
                .header(header::CONTENT_TYPE, mime)
                .body(Cow::Owned(bytes))
                .unwrap();
        }
    }

    Response::builder()
        .status(404)
        .body(Cow::Borrowed("Not Found".as_bytes()))
        .unwrap()
}

fn file_label(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

fn open_file(path: &str, reveal: bool) {
    let p = Path::new(path);
    if !p.exists() {
        return;
    }
    if reveal {
        use std::os::windows::process::CommandExt;
        // explorer는 \" 이스케이프를 이해하지 못하므로 인자를 그대로(raw) 넘긴다
        let _ = std::process::Command::new("explorer")
            .raw_arg(format!("/select,\"{}\"", path.replace('/', "\\")))
            .spawn();
    } else {
        use windows::core::PCWSTR;
        use windows::Win32::UI::WindowsAndMessaging::SW_SHOW;

        let wide_path = wide(path);
        let op = wide("open");
        // 파일은 그 파일이 있는 폴더를 작업 폴더로 해서 연다 (실행 파일·배치 파일용)
        let dir = if p.is_dir() { None } else { p.parent().map(|d| wide(&d.to_string_lossy())) };
        unsafe {
            windows::Win32::UI::Shell::ShellExecuteW(
                None,
                PCWSTR(op.as_ptr()),
                PCWSTR(wide_path.as_ptr()),
                PCWSTR::null(),
                dir.as_ref().map_or(PCWSTR::null(), |d| PCWSTR(d.as_ptr())),
                SW_SHOW,
            );
        }
    }
}

/// 사진 선택 대화상자. 취소하면 None.
fn pick_image_file(owner: windows::Win32::Foundation::HWND) -> Option<String> {
    use windows::core::{PCWSTR, PWSTR};
    use windows::Win32::UI::Controls::Dialogs::{
        GetOpenFileNameW, OFN_FILEMUSTEXIST, OFN_HIDEREADONLY, OFN_NOCHANGEDIR, OFN_PATHMUSTEXIST, OPENFILENAMEW,
    };

    let filter: Vec<u16> = "이미지 파일\0*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.gif;*.tif;*.tiff\0모든 파일\0*.*\0\0"
        .encode_utf16()
        .collect();
    let title = wide("검색할 사진 선택");
    let mut file = [0u16; 2048];
    let mut ofn = OPENFILENAMEW {
        lStructSize: std::mem::size_of::<OPENFILENAMEW>() as u32,
        hwndOwner: owner,
        lpstrFilter: PCWSTR(filter.as_ptr()),
        lpstrFile: PWSTR(file.as_mut_ptr()),
        nMaxFile: file.len() as u32,
        lpstrTitle: PCWSTR(title.as_ptr()),
        Flags: OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY | OFN_NOCHANGEDIR,
        ..Default::default()
    };
    if unsafe { GetOpenFileNameW(&mut ofn) }.as_bool() {
        let len = file.iter().position(|&c| c == 0).unwrap_or(0);
        Some(String::from_utf16_lossy(&file[..len]))
    } else {
        None
    }
}

/// 폴더 선택 대화상자 (여러 개, 드라이브도 고를 수 있다). 취소하면 빈 목록.
fn pick_folders(owner: windows::Win32::Foundation::HWND, title: &str) -> Vec<String> {
    use windows::core::HSTRING;
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CoTaskMemFree, CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED,
    };
    use windows::Win32::UI::Shell::{
        FileOpenDialog, IFileOpenDialog, FOS_ALLOWMULTISELECT, FOS_FORCEFILESYSTEM, FOS_PATHMUSTEXIST,
        FOS_PICKFOLDERS, SIGDN_FILESYSPATH,
    };

    let mut out = Vec::new();
    unsafe {
        // 창 스레드는 이미 COM이 켜져 있다 (WebView2). 한 번 더 불러도 해가 없다.
        let _ = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
        let dialog: IFileOpenDialog = match CoCreateInstance(&FileOpenDialog, None, CLSCTX_INPROC_SERVER) {
            Ok(d) => d,
            Err(_) => return out,
        };
        let opts = dialog.GetOptions().unwrap_or_default();
        let _ = dialog.SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_ALLOWMULTISELECT | FOS_PATHMUSTEXIST);
        let _ = dialog.SetTitle(&HSTRING::from(title));
        // 취소하면 Err(ERROR_CANCELLED)
        if dialog.Show(owner).is_err() {
            return out;
        }
        let Ok(items) = dialog.GetResults() else { return out };
        for i in 0..items.GetCount().unwrap_or(0) {
            let Ok(item) = items.GetItemAt(i) else { continue };
            if let Ok(name) = item.GetDisplayName(SIGDN_FILESYSPATH) {
                if let Ok(path) = name.to_string() {
                    out.push(path);
                }
                CoTaskMemFree(Some(name.0 as *const _));
            }
        }
    }
    out
}

/// 설정 화면의 "검색 폴더 관리" 내용. 폴더가 있는지 확인하는 데 시간이 걸릴 수 있어
/// (끊긴 네트워크 드라이브 등) 창 스레드가 아닌 곳에서 부른다.
fn folders_view(cfg: &Config, error: Option<String>) -> serde_json::Value {
    use locall::config::read_list;
    use locall::scan::{detect_cloud_paths, is_under, list_drives, norm_path};

    let cloud = detect_cloud_paths(cfg);
    let mut error = error;
    let mut read = |list| {
        read_list(list).unwrap_or_else(|e| {
            error.get_or_insert_with(|| format!("{:#}", e));
            Vec::new()
        })
    };
    let roots = read(PathList::Roots);
    let excludes = read(PathList::Excludes);
    let root_norms: Vec<String> = roots.iter().filter(|r| !r.path.is_empty()).map(|r| norm_path(&r.path)).collect();

    let roots_json: Vec<_> = roots
        .iter()
        .map(|r| {
            let n = norm_path(&r.path);
            let status = if r.path.is_empty() {
                "unresolved"
            } else if is_under(&n, &cloud) {
                "cloud"
            } else if !Path::new(&r.path).is_dir() {
                "missing"
            } else {
                "ok"
            };
            let tag = match r.raw.as_str() {
                "<current_drive>" | "<drive>" | "%DRIVE%" => "install",
                "<project>" | "." => "project",
                _ => "",
            };
            serde_json::json!({ "raw": r.raw, "path": r.path, "status": status, "tag": tag })
        })
        .collect();

    let excludes_json: Vec<_> = excludes
        .iter()
        .map(|x| {
            let n = norm_path(&x.path);
            // 어떤 검색 폴더와도 겹치지 않으면(안에도, 위에도 없으면) 효과가 없다
            let overlaps = root_norms.iter().any(|r| is_under(&n, std::slice::from_ref(r)) || is_under(r, std::slice::from_ref(&n)));
            let status = if x.path.is_empty() {
                "unresolved"
            } else if !overlaps {
                "outside"
            } else if !Path::new(&x.path).is_dir() {
                "missing"
            } else {
                "ok"
            };
            serde_json::json!({ "raw": x.raw, "path": x.path, "status": status })
        })
        .collect();

    // 아직 검색 폴더가 아닌 드라이브만 (클라우드 가상 드라이브 제외)
    let drives: Vec<_> = list_drives()
        .into_iter()
        .filter(|d| {
            let n = norm_path(&d.path);
            !is_under(&n, &cloud) && !root_norms.iter().any(|r| r.trim_end_matches('\\') == n.trim_end_matches('\\'))
        })
        .collect();

    serde_json::json!({
        "roots": roots_json,
        "excludes": excludes_json,
        "drives": drives,
        "error": error,
    })
}

/// 폴더 목록을 고치고(또는 읽기만 하고) 결과 화면을 창으로 보낸다. 다른 스레드에서 돈다.
fn spawn_folder_job(proxy: EventLoopProxy<UserEvent>, cfg: Config, op: FolderOp, picked: Vec<String>) {
    use locall::config::{add_path, remove_path};
    thread::spawn(move || {
        let mut changed = false;
        let mut error = None;
        let mut apply = |r: Result<bool>| match r {
            Ok(c) => changed |= c,
            Err(e) => {
                error.get_or_insert_with(|| format!("{:#}", e));
            }
        };
        match op {
            FolderOp::Get => {}
            FolderOp::Pick(list) => {
                for p in &picked {
                    apply(add_path(list, p));
                }
            }
            FolderOp::Add(list, path) => apply(add_path(list, &path)),
            FolderOp::Remove(list, raw) => apply(remove_path(list, &raw)),
        }
        let view = folders_view(&cfg, error);
        let _ = proxy.send_event(UserEvent::FoldersView { view, changed });
    });
}

fn run_query(searcher: &Searcher, embedder: &mut Option<Embedder>, query: &Query) -> Result<Vec<SearchResult>> {
    match query {
        Query::Text { query, kind } => {
            let vec = match embedder.as_mut().map(|e| e.query(query)) {
                Some(Ok(v)) => Some(v),
                Some(Err(_)) => {
                    // 작업자가 죽었으면 버리고, 이름 일치만으로라도 답한다
                    if embedder.as_mut().map_or(false, |e| !e.is_alive()) {
                        *embedder = None;
                    }
                    None
                }
                None => None,
            };
            searcher.search(query, vec.as_deref(), kind, FETCH)
        }
        Query::Image { path } => {
            let e = embedder
                .as_mut()
                .ok_or_else(|| anyhow!("AI 모델을 불러오는 중입니다. 잠시 후 자동으로 다시 찾습니다."))?;
            let vec = e.image(Path::new(path))?;
            searcher.search_image(&vec, FETCH)
        }
    }
}

/// 검색 전용 스레드. 색인과 임베딩 작업자를 혼자 쥐고, 밀린 요청은 최신 것만 처리한다.
fn search_worker(
    cfg: Config,
    db_path: PathBuf,
    jobs: Receiver<Job>,
    job_tx: Sender<Job>,
    proxy: EventLoopProxy<UserEvent>,
) {
    let status = |text: &str| {
        let _ = proxy.send_event(UserEvent::Status(text.to_string()));
    };
    status(MODEL_LOADING);

    let mut searcher = match Searcher::new(&cfg, &db_path) {
        Ok(s) => s,
        Err(e) => {
            status(&format!("색인을 불러오지 못했습니다: {:#}", e));
            return;
        }
    };

    let mut embedder: Option<Embedder> = None;
    let mut model_note = MODEL_LOADING.to_string();
    let mut last: Option<(u64, Query)> = None;
    // 사진 작업자는 ONNX 사용 시 처음 사진을 찾을 때 따로 띄운다. 그동안에도 글자 검색은 계속된다.
    let mut image_loading = false;
    let mut image_note: Option<String> = None;
    let mut downloading = false;

    while let Ok(first) = jobs.recv() {
        let mut pending = vec![first];
        pending.extend(jobs.try_iter());

        let mut rerun = false;
        let mut latest = None;
        for job in pending {
            match job {
                Job::Search { seq, query } => latest = Some((seq, query)),
                Job::Download(text) => {
                    model_note = text;
                    downloading = true;
                }
                Job::Embedder(Ok(e)) => {
                    embedder = Some(e);
                    model_note.clear();
                    downloading = false;
                    rerun = true;
                }
                Job::Embedder(Err(e)) => {
                    model_note = format!("AI 모델을 불러오지 못해 검색어가 들어간 파일만 찾습니다: {}", e);
                    downloading = false;
                    rerun = true;
                }
                Job::ImageWorker(loaded) => {
                    image_loading = false;
                    let waiting = matches!(last, Some((_, Query::Image { .. })));
                    match loaded {
                        Ok(w) => {
                            if let Some(e) = embedder.as_mut() {
                                e.set_image_worker(w);
                            }
                            // 사진 검색을 기다리던 중이면 그 검색을 다시 돌린다
                            rerun |= waiting;
                        }
                        // 실패하면 다시 돌리지 않는다 (다음 사진 검색 때 다시 시도)
                        Err(e) => image_note = Some(format!("사진 검색 모델을 불러오지 못했습니다: {}", e)),
                    }
                }
            }
        }

        if searcher.maybe_reload().unwrap_or(false) {
            rerun = true;
        }
        let (seq, query) = match latest.or_else(|| if rerun { last.clone() } else { None }) {
            Some(x) => x,
            None => {
                let note = image_note.take().unwrap_or_else(|| model_note.clone());
                status(&idle_status(searcher.stats().0 > 0, &note, downloading));
                continue;
            }
        };
        last = Some((seq, query.clone()));

        // 사진 작업자가 필요하면 다른 스레드에서 띄우고, 이 스레드는 다음 요청을 받으러 간다
        if matches!(query, Query::Image { .. }) && embedder.as_mut().map_or(false, |e| e.needs_image_worker()) {
            if !image_loading {
                image_loading = true;
                let (cfg, tx) = (cfg.clone(), job_tx.clone());
                thread::spawn(move || {
                    let loaded = ImageWorker::spawn(&cfg).map_err(|e| format!("{:#}", e));
                    let _ = tx.send(Job::ImageWorker(loaded));
                });
            }
            // 결과는 보내지 않는다: 빈 결과를 보내면 화면이 "찾지 못했습니다"로 바뀐다.
            // 화면은 "찾는 중…" 그대로 두고, 준비되면 위에서 이 검색을 다시 돌려 결과를 보낸다.
            status(IMAGE_MODEL_LOADING);
            continue;
        }

        let had_model = embedder.is_some();
        let outcome = run_query(&searcher, &mut embedder, &query);
        if had_model && embedder.is_none() {
            model_note = "AI 모델 작업자가 종료되어 검색어가 들어간 파일만 찾습니다 (data\\embed.log 참고).".to_string();
        }

        match outcome {
            Ok(results) => {
                status(&idle_status(searcher.stats().0 > 0, &model_note, downloading));
                if let Ok(json) = serde_json::to_string(&results) {
                    let _ = proxy.send_event(UserEvent::Render { seq, json });
                }
            }
            Err(e) => {
                status(&format!("검색 실패: {:#}", e));
                let _ = proxy.send_event(UserEvent::Render { seq, json: "[]".to_string() });
            }
        }
    }
}

/// 이미 떠 있으면 None. 반환된 핸들은 프로세스가 끝날 때까지 쥐고 있어야 한다.
/// `wait`이면 (업데이트 직후) 옛 프로세스가 끝나길 최대 10초 기다린다.
fn single_instance(wait: bool) -> Option<windows::Win32::Foundation::HANDLE> {
    use windows::core::w;
    use windows::Win32::Foundation::{CloseHandle, GetLastError, ERROR_ALREADY_EXISTS};
    use windows::Win32::System::Threading::CreateMutexW;
    use windows::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_ICONINFORMATION, MB_OK};

    unsafe {
        let deadline = Instant::now() + Duration::from_secs(10);
        let (handle, exists) = loop {
            let handle = CreateMutexW(None, false, w!("LocalL_rust_single_instance")).ok()?;
            let exists = GetLastError() == ERROR_ALREADY_EXISTS;
            if !(wait && exists && Instant::now() < deadline) {
                break (handle, exists);
            }
            let _ = CloseHandle(handle);
            thread::sleep(Duration::from_millis(200));
        };
        if exists {
            MessageBoxW(
                None,
                w!("LocalL이 이미 실행 중입니다.\n단축키를 눌러 검색창을 여세요."),
                w!("LocalL"),
                MB_ICONINFORMATION | MB_OK,
            );
            return None;
        }
        Some(handle)
    }
}

fn run_app() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let started_by_autostart = args.iter().any(|a| a == autostart::ARG);
    let after_update = args.iter().any(|a| a == update::AFTER_UPDATE_ARG);
    // 업데이트 중에 exe 이름이 바뀌기 전의 경로 (업데이트 뒤 다시 켤 때 쓴다)
    let exe_path = std::env::current_exe()?;

    let _instance = match single_instance(after_update) {
        Some(h) => h,
        None => return Ok(()),
    };
    update::cleanup();
    autostart::refresh_path();

    let cfg = load_config(None)?;
    let db_path = default_db_path();
    // 폴더 화면에서 클라우드 폴더를 알아볼 때 쓴다 (검색 폴더 목록 자체는 매번 설정 파일에서 읽는다)
    let cfg_for_folders = cfg.clone();

    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();

    // 0. 검색 스레드와 모델 적재 스레드 (창은 기다리지 않고 바로 뜬다)
    let (job_tx, job_rx) = channel::<Job>();
    {
        let (cfg, db_path, proxy) = (cfg.clone(), db_path.clone(), proxy.clone());
        let tx = job_tx.clone();
        thread::spawn(move || search_worker(cfg, db_path, job_rx, tx, proxy));
    }
    {
        let (cfg, tx) = (cfg.clone(), job_tx.clone());
        thread::spawn(move || {
            if let Err(e) = fetch_models(&tx) {
                let msg = format!("모델을 받지 못했습니다 ({:#}). 인터넷 연결을 확인하고 다시 실행하면 이어 받습니다.", e);
                let _ = tx.send(Job::Embedder(Err(msg)));
                return;
            }
            let loaded = Embedder::spawn(&cfg).map_err(|e| format!("{:#}", e));
            let _ = tx.send(Job::Embedder(loaded));
        });
    }

    // 1. 헤드리스 플로팅 윈도우 생성 (작업표시줄/앱 아이콘 포함)
    let icon_bytes = include_bytes!("../../assets/ui/app_icon.png");
    let window_icon = image::load_from_memory(icon_bytes)
        .ok()
        .and_then(|img| {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            tao::window::Icon::from_rgba(rgba.into_raw(), w, h).ok()
        });

    let mut builder = WindowBuilder::new()
        .with_title("LocalL")
        .with_decorations(false) // 헤드리스
        .with_transparent(true)   // 투명 창
        .with_always_on_top(true) // 최상위
        .with_resizable(false)
        .with_inner_size(LogicalSize::new(760.0, 560.0))
        // PC를 켤 때 자동으로 시작됐으면 창은 숨긴 채 단축키만 기다린다
        .with_visible(!started_by_autostart);

    if let Some(icon) = window_icon {
        builder = builder.with_window_icon(Some(icon));
    }

    let window = builder.build(&event_loop)?;

    // 화면 중앙 상단에 위치 설정
    if let Some(monitor) = window.primary_monitor() {
        let screen_size = monitor.size();
        let scale = monitor.scale_factor();
        let sw = screen_size.width as f64 / scale;
        let sh = screen_size.height as f64 / scale;
        let x = (sw - 760.0) / 2.0;
        let y = sh * 0.16;
        window.set_outer_position(LogicalPosition::new(x, y));
    }

    // 네 귀퉁이 하얀색 영역 완전 제거: Windows DWM 라운딩 및 물리적 둥근 리전 설정
    #[cfg(windows)]
    {
        use tao::platform::windows::WindowExtWindows;
        use windows::Win32::Foundation::HWND;
        use windows::Win32::Graphics::Dwm::{DwmSetWindowAttribute, DWMWA_WINDOW_CORNER_PREFERENCE};
        use windows::Win32::Graphics::Gdi::{CreateRoundRectRgn, SetWindowRgn};

        let hwnd = HWND(window.hwnd() as _);
        let corner_pref = 2u32; // DWMWCP_ROUND
        unsafe {
            let _ = DwmSetWindowAttribute(
                hwnd,
                DWMWA_WINDOW_CORNER_PREFERENCE,
                &corner_pref as *const _ as *const _,
                std::mem::size_of::<u32>() as u32,
            );

            let p_size = window.inner_size();
            let scale = window.scale_factor();
            let radius = (32.0 * scale) as i32;
            let rgn = CreateRoundRectRgn(0, 0, (p_size.width + 1) as i32, (p_size.height + 1) as i32, radius, radius);
            let _ = SetWindowRgn(hwnd, rgn, true);
        }
    }

    // 2. 전역 단축키 매니저 시작
    let mut hotkey_mgr = HotkeyManager::new(&cfg.hotkey).ok();
    let mut current_hotkey = cfg.hotkey.clone();

    // 3. Wry WebView 생성
    let jobs_for_ipc = job_tx.clone();
    let proxy_for_ipc = proxy.clone();

    let html_content = include_str!("../../assets/ui/index.html");
    let base_ui_dir = base_dir().join("assets").join("ui");
    let pasted_image_path = base_dir().join("data").join("query_image");

    let webview = WebViewBuilder::new()
        .with_transparent(true)
        .with_asynchronous_custom_protocol("locall".into(), move |_id, req, responder| {
            let path = req.uri().path();
            // 사진 미리보기: thumb?p=<파일 경로>. 해독이 무거우니 다른 스레드에서 만들어 답한다.
            if path == "/thumb" {
                let file = req
                    .uri()
                    .query()
                    .and_then(|q| q.split('&').find_map(|kv| kv.strip_prefix("p=")))
                    .map(percent_decode)
                    .unwrap_or_default();
                thumb_pool().spawn(move || {
                    let resp = match locall::thumb::thumbnail_png(Path::new(&file)) {
                        Some(png) => Response::builder()
                            .header(header::CONTENT_TYPE, "image/png")
                            .header(header::CACHE_CONTROL, "max-age=600")
                            .body(Cow::Owned(png)),
                        None => Response::builder().status(404).body(Cow::Borrowed("Not Found".as_bytes())),
                    };
                    responder.respond(resp.unwrap());
                });
                return;
            }
            responder.respond(serve_ui(path, html_content, &base_ui_dir));
        })
        .with_ipc_handler(move |req| {
            if let Ok(msg) = serde_json::from_str::<IpcMessage>(req.body()) {
                match msg {
                    IpcMessage::Search { query, kind, seq } => {
                        // 사진 파일 경로를 적거나 붙여넣으면 그 사진으로 찾는다
                        let unquoted = query.trim().trim_matches(|c| c == '"' || c == '\'');
                        if is_image_path(Path::new(unquoted)) {
                            let _ = proxy_for_ipc.send_event(UserEvent::ImageChosen {
                                path: unquoted.to_string(),
                                label: file_label(unquoted),
                            });
                        } else {
                            let _ = jobs_for_ipc.send(Job::Search { seq, query: Query::Text { query, kind } });
                        }
                    }
                    IpcMessage::SearchImage { path, seq } => {
                        let _ = jobs_for_ipc.send(Job::Search { seq, query: Query::Image { path } });
                    }
                    IpcMessage::PasteImage { data } => {
                        // 클립보드의 사진(캡처, 복사한 파일)은 내용만 넘어오므로 파일로 적어 두고 찾는다
                        match decode_base64(&data).map(|b| fs::write(&pasted_image_path, b)) {
                            Some(Ok(())) => {
                                let _ = proxy_for_ipc.send_event(UserEvent::ImageChosen {
                                    path: pasted_image_path.to_string_lossy().to_string(),
                                    label: "클립보드 이미지".to_string(),
                                });
                            }
                            _ => {
                                let _ = proxy_for_ipc
                                    .send_event(UserEvent::Status("붙여넣은 사진을 읽지 못했습니다.".to_string()));
                            }
                        }
                    }
                    IpcMessage::Open { path, reveal } => {
                        open_file(&path, reveal);
                        let _ = proxy_for_ipc.send_event(UserEvent::Hide);
                    }
                    IpcMessage::Close => {
                        let _ = proxy_for_ipc.send_event(UserEvent::Hide);
                    }
                    IpcMessage::PickImage => {
                        let _ = proxy_for_ipc.send_event(UserEvent::PickImage);
                    }
                    IpcMessage::RunIndex => {
                        let _ = proxy_for_ipc.send_event(UserEvent::RunIndex);
                    }
                    IpcMessage::UpdateHotkey { hotkey } => {
                        let _ = proxy_for_ipc.send_event(UserEvent::UpdateHotkey(hotkey));
                    }
                    IpcMessage::Quit => {
                        let _ = proxy_for_ipc.send_event(UserEvent::Quit);
                    }
                    IpcMessage::GetConfig => {
                        let _ = proxy_for_ipc.send_event(UserEvent::GetConfig);
                    }
                    IpcMessage::SetAutostart { enabled } => {
                        let _ = proxy_for_ipc.send_event(UserEvent::SetAutostart(enabled));
                    }
                    IpcMessage::CheckUpdate => {
                        let _ = proxy_for_ipc.send_event(UserEvent::CheckUpdate);
                    }
                    IpcMessage::InstallUpdate => {
                        let _ = proxy_for_ipc.send_event(UserEvent::InstallUpdate);
                    }
                    IpcMessage::GetFolders => {
                        let _ = proxy_for_ipc.send_event(UserEvent::Folders(FolderOp::Get));
                    }
                    IpcMessage::PickFolder { list } => {
                        let _ = proxy_for_ipc.send_event(UserEvent::Folders(FolderOp::Pick(list)));
                    }
                    IpcMessage::AddPath { list, path } => {
                        let _ = proxy_for_ipc.send_event(UserEvent::Folders(FolderOp::Add(list, path)));
                    }
                    IpcMessage::RemovePath { list, raw } => {
                        let _ = proxy_for_ipc.send_event(UserEvent::Folders(FolderOp::Remove(list, raw)));
                    }
                }
            }
        })
        .with_url("locall://app/index.html")
        .build(&window)?;

    let mut last_shown = Instant::now();
    let mut last_hotkey = Instant::now();
    let mut current_status = String::new();

    // 새 버전 확인: 창이 뜨고 조금 뒤에 백그라운드에서
    let mut update_view = UpdateView { version: update::VERSION, status: "checking", latest: None, message: String::new() };
    let mut pending_release: Option<Release> = None;
    // 검색 폴더를 바꾼 뒤 아직 색인을 갱신하지 않았으면 true
    let mut folders_changed = false;
    spawn_update_check(proxy.clone(), Duration::from_secs(5), false);

    // 4. 이벤트 루프 실행
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(30),
        );

        // 전역 단축키 폴링 (디바운스 200ms)
        if let Some(hk) = &hotkey_mgr {
            if hk.try_recv() && last_hotkey.elapsed() > Duration::from_millis(200) {
                last_hotkey = Instant::now();
                let is_vis = window.is_visible();
                if is_vis {
                    window.set_visible(false);
                } else {
                    last_shown = Instant::now();
                    window.set_visible(true);

                    #[cfg(windows)]
                    unsafe {
                        use tao::platform::windows::WindowExtWindows;
                        use windows::Win32::Foundation::HWND;
                        use windows::Win32::UI::WindowsAndMessaging::SetForegroundWindow;
                        use windows::Win32::UI::Input::KeyboardAndMouse::{keybd_event, SetFocus, KEYEVENTF_KEYUP};

                        let hwnd = HWND(window.hwnd() as _);
                        // Windows 포커스 전환 잠금 우회 (Alt 트릭)
                        keybd_event(0x12, 0, Default::default(), 0);
                        keybd_event(0x12, 0, KEYEVENTF_KEYUP, 0);
                        let _ = SetForegroundWindow(hwnd);
                        let _ = SetFocus(hwnd);
                    }

                    window.set_focus();
                    let _ = webview.evaluate_script("window.focusInput();");
                }
            }
        }

        match event {
            Event::UserEvent(user_event) => match user_event {
                UserEvent::Render { seq, json } => {
                    let js = format!("window.renderSearchResults({}, {});", json, seq);
                    let _ = webview.evaluate_script(&js);
                }
                UserEvent::Status(text) => set_status(&webview, &mut current_status, text),
                UserEvent::ImageChosen { path, label } => {
                    let js = format!(
                        "window.startImageSearch({}, {});",
                        serde_json::json!(path),
                        serde_json::json!(label)
                    );
                    let _ = webview.evaluate_script(&js);
                }
                UserEvent::PickImage => {
                    use tao::platform::windows::WindowExtWindows;
                    use windows::Win32::Foundation::HWND;

                    let picked = pick_image_file(HWND(window.hwnd() as _));
                    // 대화상자 때문에 잃은 포커스로 창이 숨지 않게 하고, 검색창으로 돌아온다
                    last_shown = Instant::now();
                    window.set_visible(true);
                    window.set_focus();
                    let _ = webview.evaluate_script("window.focusInput();");
                    if let Some(path) = picked {
                        let js = format!(
                            "window.startImageSearch({}, {});",
                            serde_json::json!(path),
                            serde_json::json!(file_label(&path))
                        );
                        let _ = webview.evaluate_script(&js);
                    }
                }
                UserEvent::Hide => {
                    window.set_visible(false);
                }
                UserEvent::Quit => {
                    *control_flow = ControlFlow::Exit;
                }
                UserEvent::RunIndex => {
                    use std::os::windows::process::CommandExt;
                    folders_changed = false;
                    const CREATE_NEW_CONSOLE: u32 = 0x00000010;

                    // Rust 색인기가 파일명·본문·PDF·사진을 모두 한다. 파이썬이 함께 있으면 Rust가
                    // 못 읽은 몇 안 되는 파일(TIFF, 일부 손상 문서)을 대기로 남겨 파이썬이 마저 한다.
                    let base = base_dir();
                    let rel_exe = base.join("target").join("release").join("locall-index.exe");
                    let rust_exe = if rel_exe.exists() { rel_exe } else { base.join("locall-index.exe") };
                    let python = find_python();
                    let mut steps = vec![format!(
                        "\"{}\" --db \"{}\"{}",
                        rust_exe.display(),
                        db_path.display(),
                        if python.is_some() { " --leave-unsupported" } else { "" }
                    )];
                    if let Some(python) = python {
                        steps.push(format!(
                            "\"{}\" -W ignore -m locall.index --db \"{}\"",
                            python.display(),
                            db_path.display()
                        ));
                    }
                    let mut cmd = std::process::Command::new("cmd");
                    // /s: 바깥 따옴표 한 쌍만 벗기므로 경로에 공백이 있어도 안전하다
                    cmd.raw_arg(format!("/s /k \"{}\"", steps.join(" & ")));
                    let _ = cmd.current_dir(&base).creation_flags(CREATE_NEW_CONSOLE).spawn();
                }
                UserEvent::UpdateHotkey(new_hk) => {
                    let lower = new_hk.to_lowercase();
                    let registered = if lower == current_hotkey && hotkey_mgr.is_some() {
                        Ok(()) // 지금 쓰는 단축키 그대로: 다시 등록하면 자기 자신과 충돌한다
                    } else {
                        HotkeyManager::new(&lower).map(|mgr| hotkey_mgr = Some(mgr))
                    };
                    match registered {
                        Ok(()) => {
                            current_hotkey = lower.clone();
                            let _ = locall::config::save_hotkey(&lower);
                            let js = format!("window.onHotkeyUpdated(true, {});", serde_json::json!(lower));
                            let _ = webview.evaluate_script(&js);
                        }
                        Err(e) => {
                            let js = format!("window.onHotkeyUpdated(false, {});", serde_json::json!(e.to_string()));
                            let _ = webview.evaluate_script(&js);
                        }
                    }
                }
                UserEvent::GetConfig => {
                    let count = |list| locall::config::read_list(list).map(|v| v.len()).ok();
                    let cfg = serde_json::json!({
                        "hotkey": current_hotkey,
                        "autostart": autostart::is_enabled(),
                        "update": update_view,
                        "folders": { "roots": count(PathList::Roots), "excludes": count(PathList::Excludes) },
                    });
                    let js = format!(
                        "window.onConfigLoaded({}); window.setStatus({});",
                        cfg,
                        serde_json::json!(current_status)
                    );
                    let _ = webview.evaluate_script(&js);
                }
                UserEvent::SetAutostart(enabled) => {
                    let js = match autostart::set_enabled(enabled) {
                        Ok(()) => format!("window.onAutostartUpdated(true, {});", autostart::is_enabled()),
                        Err(e) => format!(
                            "window.onAutostartUpdated(false, {}, {});",
                            autostart::is_enabled(),
                            serde_json::json!(format!("{:#}", e))
                        ),
                    };
                    let _ = webview.evaluate_script(&js);
                }
                UserEvent::CheckUpdate => {
                    if update_view.status != "checking" && update_view.status != "downloading" {
                        update_view.status = "checking";
                        update_view.message.clear();
                        push_update_view(&webview, &update_view);
                        spawn_update_check(proxy.clone(), Duration::ZERO, true);
                    }
                }
                UserEvent::UpdateChecked { result, manual } => {
                    match result {
                        Ok(Some(rel)) => {
                            update_view.status = "available";
                            update_view.latest = Some(rel.version.clone());
                            update_view.message.clear();
                            let text = format!("새 버전 v{}이(가) 있습니다 · ⚙ 설정 → 업데이트에서 설치하세요.", rel.version);
                            set_status(&webview, &mut current_status, text);
                            pending_release = Some(rel);
                        }
                        Ok(None) => {
                            update_view.status = "latest";
                            update_view.message.clear();
                        }
                        // 자동 확인 실패(오프라인 등)는 조용히 넘긴다
                        Err(e) if manual => {
                            update_view.status = "error";
                            update_view.message = e;
                        }
                        Err(_) => update_view.status = "idle",
                    }
                    push_update_view(&webview, &update_view);
                }
                UserEvent::InstallUpdate => {
                    if update_view.status == "downloading" {
                        // 이미 받는 중
                    } else if let Some(rel) = pending_release.clone() {
                        update_view.status = "downloading";
                        update_view.message = "받는 중…".to_string();
                        push_update_view(&webview, &update_view);
                        let proxy = proxy.clone();
                        thread::spawn(move || {
                            let mut last = Instant::now() - Duration::from_secs(1);
                            let result = update::install(&rel, &mut |done, total| {
                                if last.elapsed() < Duration::from_millis(300) && done < total {
                                    return;
                                }
                                last = Instant::now();
                                let text = format!(
                                    "업데이트 받는 중 v{} · {} / {}",
                                    rel.version,
                                    locall::download::human(done),
                                    locall::download::human(total)
                                );
                                let _ = proxy.send_event(UserEvent::UpdateProgress(text));
                            });
                            let _ = proxy.send_event(match result {
                                Ok(()) => UserEvent::UpdateInstalled,
                                Err(e) => UserEvent::UpdateFailed(format!("{:#}", e)),
                            });
                        });
                    } else {
                        let _ = proxy.send_event(UserEvent::CheckUpdate);
                    }
                }
                UserEvent::UpdateProgress(text) => {
                    update_view.message = text.clone();
                    push_update_view(&webview, &update_view);
                    set_status(&webview, &mut current_status, text);
                }
                UserEvent::UpdateInstalled => {
                    match std::process::Command::new(&exe_path).arg(update::AFTER_UPDATE_ARG).spawn() {
                        Ok(_) => *control_flow = ControlFlow::Exit,
                        Err(e) => {
                            update_view.status = "error";
                            update_view.message = format!("업데이트는 설치했지만 다시 켜지 못했습니다. LocalL을 다시 실행해 주세요 ({})", e);
                            push_update_view(&webview, &update_view);
                            set_status(&webview, &mut current_status, update_view.message.clone());
                        }
                    }
                }
                UserEvent::Folders(op) => {
                    let picked = if let FolderOp::Pick(list) = &op {
                        use tao::platform::windows::WindowExtWindows;
                        use windows::Win32::Foundation::HWND;

                        let title = match list {
                            PathList::Roots => "검색할 폴더 선택",
                            PathList::Excludes => "검색에서 뺄 폴더 선택",
                        };
                        last_shown = Instant::now();
                        let picked = pick_folders(HWND(window.hwnd() as _), title);
                        // 대화상자 때문에 잃은 포커스로 창이 숨지 않게 한다
                        last_shown = Instant::now();
                        window.set_visible(true);
                        window.set_focus();
                        picked
                    } else {
                        Vec::new()
                    };
                    spawn_folder_job(proxy.clone(), cfg_for_folders.clone(), op, picked);
                }
                UserEvent::FoldersView { mut view, changed } => {
                    folders_changed |= changed;
                    view["changed"] = serde_json::json!(folders_changed);
                    let js = format!("window.onFoldersLoaded && window.onFoldersLoaded({});", view);
                    let _ = webview.evaluate_script(&js);
                }
                UserEvent::UpdateFailed(e) => {
                    update_view.status = if pending_release.is_some() { "available" } else { "error" };
                    update_view.message = format!("업데이트 실패: {}", e);
                    push_update_view(&webview, &update_view);
                    set_status(&webview, &mut current_status, update_view.message.clone());
                }
            },
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Focused(focused) => {
                    // 창이 나타난 지 400ms 이내에는 포커스 아웃에 의한 즉시 꺼짐 방지
                    if !focused && window.is_visible() && last_shown.elapsed() > Duration::from_millis(400) {
                        window.set_visible(false);
                    }
                }
                _ => (),
            },
            _ => (),
        }
    });
}

fn set_status(webview: &wry::WebView, current: &mut String, text: String) {
    if text != *current {
        *current = text;
        let js = format!("window.setStatus({});", serde_json::json!(current));
        let _ = webview.evaluate_script(&js);
    }
}

fn push_update_view(webview: &wry::WebView, view: &UpdateView) {
    if let Ok(json) = serde_json::to_string(view) {
        let _ = webview.evaluate_script(&format!("window.onUpdateState && window.onUpdateState({});", json));
    }
}

fn main() {
    if let Err(e) = run_app() {
        let err_msg = format!("LocalL 실행 실패: {:?}", e);
        #[cfg(windows)]
        unsafe {
            use windows::core::PCWSTR;
            let wide_msg = wide(&err_msg);
            let wide_title = wide("LocalL Error");
            let _ = windows::Win32::UI::WindowsAndMessaging::MessageBoxW(
                None,
                PCWSTR(wide_msg.as_ptr()),
                PCWSTR(wide_title.as_ptr()),
                windows::Win32::UI::WindowsAndMessaging::MB_ICONERROR | windows::Win32::UI::WindowsAndMessaging::MB_OK,
            );
        }
    }
}
