use crate::config::{ext_to_kind_map, Config};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub const ATTR_HIDDEN: u32 = 0x2;
pub const ATTR_SYSTEM: u32 = 0x4;
pub const ATTR_REPARSE: u32 = 0x400;
pub const ATTR_CLOUD: u32 = 0x1000 | 0x40000 | 0x400000; // OFFLINE | RECALL_ON_OPEN | RECALL_ON_DATA_ACCESS

pub struct FileEntry {
    pub path: String,
    pub name: String,
    pub is_dir: bool,
    pub ext: String,
    pub kind: String,
    pub size: u64,
    pub mtime: f64,
}

#[derive(Default, Debug)]
pub struct ScanReport {
    pub cloud: Vec<String>,
    pub denied: usize,
}

fn norm_path(p: &str) -> String {
    p.replace('/', "\\").to_lowercase()
}

fn is_under(path_n: &str, parents: &[String]) -> bool {
    let mut path_with_slash = path_n.to_string();
    if !path_with_slash.ends_with('\\') {
        path_with_slash.push('\\');
    }
    parents.iter().any(|parent| {
        let mut p_with_slash = parent.trim_end_matches('\\').to_string();
        p_with_slash.push('\\');
        path_n == parent || path_with_slash.starts_with(&p_with_slash)
    })
}

#[cfg(windows)]
fn get_volume_label(drive: &str) -> String {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    use windows::core::PCWSTR;
    use windows::Win32::Storage::FileSystem::GetVolumeInformationW;

    let root: Vec<u16> = OsStr::new(drive).encode_wide().chain(Some(0)).collect();
    let mut name_buf = [0u16; 261];
    let ok = unsafe {
        GetVolumeInformationW(
            PCWSTR(root.as_ptr()),
            Some(&mut name_buf),
            None,
            None,
            None,
            None,
        )
    };
    if ok.is_ok() {
        let len = name_buf.iter().position(|&c| c == 0).unwrap_or(name_buf.len());
        String::from_utf16_lossy(&name_buf[..len])
    } else {
        String::new()
    }
}

#[cfg(not(windows))]
fn get_volume_label(_drive: &str) -> String {
    String::new()
}

pub fn detect_cloud_paths(cfg: &Config) -> Vec<String> {
    let mut found = cfg.cloud_paths.clone();

    for var in ["OneDrive", "OneDriveConsumer", "OneDriveCommercial"] {
        if let Ok(val) = std::env::var(var) {
            found.push(val);
        }
    }

    for letter in b'A'..=b'Z' {
        let drive = format!("{}:\\", letter as char);
        if Path::new(&drive).exists() {
            let label = get_volume_label(&drive).to_lowercase();
            if label.contains("google drive") {
                found.push(drive);
            }
        }
    }

    let bases = [std::env::var("LOCALAPPDATA").ok(), std::env::var("APPDATA").ok()];
    for base in bases.into_iter().flatten() {
        let info_path = PathBuf::from(base).join("Dropbox").join("info.json");
        if info_path.is_file() {
            if let Ok(content) = std::fs::read_to_string(&info_path) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(obj) = json.as_object() {
                        for account in obj.values() {
                            if let Some(p) = account.get("path").and_then(|v| v.as_str()) {
                                found.push(p.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    let mut set = HashSet::new();
    let mut out = Vec::new();
    for p in found {
        let n = norm_path(&p);
        if set.insert(n.clone()) {
            out.push(n);
        }
    }
    out.sort();
    out
}

pub fn walk_files<F>(roots: &[String], cfg: &Config, report: &mut ScanReport, mut on_entry: F)
where
    F: FnMut(FileEntry),
{
    use std::os::windows::fs::MetadataExt;

    let cloud = detect_cloud_paths(cfg);
    let excluded: Vec<String> = cfg.exclude_paths.iter().map(|p| norm_path(p)).collect();
    let exclude_names: HashSet<String> = cfg.exclude_dir_names.iter().cloned().collect();
    let cloud_names: HashSet<String> = cfg.cloud_dir_names.iter().cloned().collect();
    let include_kinds: HashSet<String> = cfg.include_kinds.iter().cloned().collect();
    let ext_map = ext_to_kind_map();

    let mut stack = Vec::new();

    for r in roots {
        let r_norm = norm_path(r);
        if is_under(&r_norm, &cloud) {
            report.cloud.push(r.clone());
        } else if Path::new(r).is_dir() {
            stack.push(r.clone());
        }
    }

    while let Some(dir_path) = stack.pop() {
        let rd = match std::fs::read_dir(&dir_path) {
            Ok(rd) => rd,
            Err(_) => {
                report.denied += 1;
                continue;
            }
        };

        let mut entries = Vec::new();
        let mut names = HashSet::new();

        for item in rd.flatten() {
            let name = item.file_name().to_string_lossy().to_string();
            names.insert(name.to_lowercase());
            entries.push((item, name));
        }

        // 파이썬 가상환경 건너뜀
        if names.contains("pyvenv.cfg") {
            continue;
        }

        let is_unity = names.contains("assets") && names.contains("projectsettings");

        for (item, name) in entries {
            let meta = match item.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };

            let attrs = meta.file_attributes();
            if (attrs & (ATTR_CLOUD | ATTR_REPARSE)) != 0 {
                continue;
            }
            if cfg.skip_hidden && (attrs & (ATTR_HIDDEN | ATTR_SYSTEM)) != 0 {
                continue;
            }

            let lname = name.to_lowercase();
            let p_str = item.path().to_string_lossy().to_string();

            let mtime_sec = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);

            if meta.is_dir() {
                if lname.starts_with('.') || exclude_names.contains(&lname) {
                    continue;
                }
                if is_unity && matches!(lname.as_str(), "library" | "temp" | "obj" | "logs") {
                    continue;
                }

                let path_n = norm_path(&p_str);
                if is_under(&path_n, &excluded) {
                    continue;
                }
                if is_under(&path_n, &cloud) || cloud_names.contains(&lname) || lname.starts_with("onedrive - ") {
                    report.cloud.push(p_str);
                    continue;
                }

                stack.push(p_str.clone());

                if include_kinds.contains("dir") {
                    on_entry(FileEntry {
                        path: p_str,
                        name,
                        is_dir: true,
                        ext: String::new(),
                        kind: "dir".to_string(),
                        size: 0,
                        mtime: mtime_sec,
                    });
                }
            } else {
                let ext = Path::new(&lname)
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();

                if let Some(&kind) = ext_map.get(&ext) {
                    if include_kinds.contains(kind) && !lname.starts_with("~$") {
                        on_entry(FileEntry {
                            path: p_str,
                            name,
                            is_dir: false,
                            ext,
                            kind: kind.to_string(),
                            size: meta.file_size(),
                            mtime: mtime_sec,
                        });
                    }
                }
            }
        }
    }
}
