//! GitHub Releases에서 새 버전을 확인하고 설치하기.
//!
//! - 확인: `releases/latest`의 태그(`v0.2.0`)를 이 프로그램의 버전과 비교한다.
//! - 설치: `LocalL-windows-x64.zip`을 받아 `.sha256`과 맞춰 본 뒤 `data/update/new`에 풀고,
//!   프로그램 폴더의 파일을 하나씩 바꾼다. 실행 중인 exe는 지울 수 없지만 이름은 바꿀 수 있으므로
//!   옛 파일은 `이름.old`로 비켜 두고, 다음 실행 때 [`cleanup`]이 지운다.
//! - `config.json`은 이미 있으면 덮어쓰지 않는다 (단축키 등 사용자 설정 유지).

use crate::config::base_dir;
use crate::download::make_agent;
use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

pub const REPO: &str = "OnOffLineD/LocalL";
pub const ASSET: &str = "LocalL-windows-x64.zip";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// 업데이트 뒤 다시 켤 때 붙는 인자 (옛 프로세스가 끝나길 기다린다)
pub const AFTER_UPDATE_ARG: &str = "--after-update";

/// 덮어쓰지 않는 파일 (이미 있을 때)
const KEEP_IF_EXISTS: &[&str] = &["config.json"];
/// 옛 파일 정리 때 들어가지 않는 폴더
const SKIP_DIRS: &[&str] = &["data", "models", "python", ".venv", "target", ".git"];

#[derive(Debug, Clone, Serialize)]
pub struct Release {
    /// `0.2.0` (앞의 v는 뗀다)
    pub version: String,
    pub zip_url: String,
    pub sha_url: String,
    pub size: u64,
}

/// `v1.2.3`, `1.2.3-beta` → (1, 2, 3). 모자라는 자리는 0.
pub fn parse_version(s: &str) -> Option<(u64, u64, u64)> {
    let s = s.trim().trim_start_matches(['v', 'V']);
    let core = s.split(['-', '+']).next()?;
    let mut parts = core.split('.');
    let mut next = || -> Option<u64> {
        match parts.next() {
            None => Some(0),
            Some(p) => p.parse().ok(),
        }
    };
    let v = (next()?, next()?, next()?);
    if parts.next().is_some() {
        return None;
    }
    Some(v)
}

pub fn is_newer(remote: &str, local: &str) -> bool {
    match (parse_version(remote), parse_version(local)) {
        (Some(r), Some(l)) => r > l,
        _ => false,
    }
}

/// target/debug, target/release에서 돌고 있으면 개발용 빌드다. 저장소 폴더를 덮어쓰지 않도록 설치를 막는다.
pub fn is_dev_build() -> bool {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.ends_with("debug") || p.ends_with("release")))
        .unwrap_or(false)
}

fn get_text(url: &str, accept: &str) -> Result<String> {
    let agent = make_agent();
    let mut resp = agent
        .get(url)
        .header("Accept", accept)
        .header("X-GitHub-Api-Version", "2022-11-28")
        .call()
        .map_err(|e| anyhow!("{}", e))?;
    let text = resp.body_mut().read_to_string().map_err(|e| anyhow!("{}", e))?;
    Ok(text)
}

/// 최신 릴리스가 지금보다 새것이면 Some.
pub fn check() -> Result<Option<Release>> {
    let url = format!("https://api.github.com/repos/{}/releases/latest", REPO);
    let text = get_text(&url, "application/vnd.github+json").context("업데이트를 확인하지 못했습니다")?;
    parse_release(&text, VERSION)
}

fn parse_release(json: &str, current: &str) -> Result<Option<Release>> {
    let v: serde_json::Value = serde_json::from_str(json).context("릴리스 정보를 읽지 못했습니다")?;
    let tag = v["tag_name"].as_str().ok_or_else(|| anyhow!("릴리스에 태그가 없습니다"))?;
    if v["draft"].as_bool() == Some(true) || v["prerelease"].as_bool() == Some(true) || !is_newer(tag, current) {
        return Ok(None);
    }
    let assets = v["assets"].as_array().cloned().unwrap_or_default();
    let find = |name: &str| assets.iter().find(|a| a["name"].as_str() == Some(name));
    let zip = find(ASSET).ok_or_else(|| anyhow!("릴리스 {}에 {}이(가) 없습니다", tag, ASSET))?;
    let sha = find(&format!("{}.sha256", ASSET)).ok_or_else(|| anyhow!("릴리스 {}에 체크섬 파일이 없습니다", tag))?;
    let url_of = |a: &serde_json::Value| a["browser_download_url"].as_str().map(str::to_string);
    Ok(Some(Release {
        version: tag.trim_start_matches(['v', 'V']).to_string(),
        zip_url: url_of(zip).ok_or_else(|| anyhow!("내려받기 주소가 없습니다"))?,
        sha_url: url_of(sha).ok_or_else(|| anyhow!("체크섬 주소가 없습니다"))?,
        size: zip["size"].as_u64().unwrap_or(0),
    }))
}

fn update_dir(base: &Path) -> PathBuf {
    base.join("data").join("update")
}

/// 새 버전을 받아 프로그램 폴더에 넣는다. 끝나면 호출한 쪽이 프로그램을 다시 켜야 한다.
/// `progress(받은 바이트, 전체 바이트)`
pub fn install(rel: &Release, progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    if is_dev_build() {
        bail!("개발용 빌드(target 폴더)에서는 업데이트를 설치하지 않습니다");
    }
    let base = base_dir();
    let dir = update_dir(&base);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).with_context(|| format!("폴더를 만들 수 없습니다: {}", dir.display()))?;

    // 1. 체크섬, 2. zip (받으면서 해시)
    let sha_text = get_text(&rel.sha_url, "application/octet-stream").context("체크섬 파일을 받지 못했습니다")?;
    let want = sha_text
        .split_whitespace()
        .next()
        .map(str::to_lowercase)
        .filter(|h| h.len() == 64 && h.chars().all(|c| c.is_ascii_hexdigit()))
        .ok_or_else(|| anyhow!("체크섬 파일 형식이 맞지 않습니다"))?;

    let zip_path = dir.join(ASSET);
    let got = download(&rel.zip_url, &zip_path, rel.size, progress)?;
    if got != want {
        let _ = fs::remove_file(&zip_path);
        bail!("받은 파일이 손상되었습니다 (체크섬 불일치). 다시 시도해 주세요");
    }

    // 3. 풀기
    let staging = dir.join("new");
    let files = extract(&zip_path, &staging)?;
    if !files.iter().any(|f| f == Path::new("locall.exe")) {
        bail!("업데이트 파일에 locall.exe가 없습니다");
    }

    // 4. 바꿔 넣기
    apply(&staging, &base, &files)?;
    Ok(())
}

fn download(url: &str, dest: &Path, size: u64, progress: &mut dyn FnMut(u64, u64)) -> Result<String> {
    let agent = make_agent();
    let mut resp = agent
        .get(url)
        .header("Accept", "application/octet-stream")
        .call()
        .map_err(|e| anyhow!("업데이트를 받지 못했습니다: {}", e))?;
    let limit = if size > 0 { size + 1 } else { 512 * 1024 * 1024 };
    let mut reader = resp.body_mut().with_config().limit(limit).reader();
    let mut out = File::create(dest)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 256 * 1024];
    let (mut done, mut last) = (0u64, Instant::now());
    progress(0, size);
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        done += n as u64;
        if last.elapsed() >= Duration::from_millis(250) {
            progress(done, size);
            last = Instant::now();
        }
    }
    out.flush()?;
    progress(done, size.max(done));
    if size > 0 && done != size {
        bail!("받다가 끊겼습니다 ({}/{} 바이트)", done, size);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{:02x}", b)).collect())
}

/// zip을 `staging`에 풀고, 풀린 파일들의 상대 경로를 돌려준다. 폴더 밖으로 나가는 경로는 거부한다.
fn extract(zip_path: &Path, staging: &Path) -> Result<Vec<PathBuf>> {
    let mut archive = zip::ZipArchive::new(File::open(zip_path)?).context("업데이트 파일(zip)을 열 수 없습니다")?;
    fs::create_dir_all(staging)?;
    let mut files = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let rel = entry
            .enclosed_name()
            .filter(|p| is_safe_relative(p))
            .ok_or_else(|| anyhow!("zip 안에 허용되지 않는 경로가 있습니다: {}", entry.name()))?;
        let out = staging.join(&rel);
        if entry.is_dir() {
            fs::create_dir_all(&out)?;
            continue;
        }
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut f = File::create(&out)?;
        std::io::copy(&mut entry, &mut f)?;
        files.push(rel);
    }
    Ok(files)
}

/// 상대 경로이고 `..`이 없으며, 사용자 데이터 폴더(data, models…)를 건드리지 않는지
fn is_safe_relative(p: &Path) -> bool {
    let mut comps = p.components().peekable();
    let first_ok = match comps.peek() {
        Some(Component::Normal(first)) => {
            let f = first.to_string_lossy().to_lowercase();
            !SKIP_DIRS.contains(&f.as_str())
        }
        _ => false,
    };
    first_ok && p.components().all(|c| matches!(c, Component::Normal(_)))
}

fn old_path(p: &Path) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".old");
    PathBuf::from(s)
}

/// `staging`의 파일들로 `base`의 파일을 바꾼다. 하나라도 실패하면 바꾼 것을 모두 되돌린다.
fn apply(staging: &Path, base: &Path, files: &[PathBuf]) -> Result<()> {
    // (대상, 옛 파일을 비켜 뒀는지)
    let mut done: Vec<(PathBuf, bool)> = Vec::new();
    let result = (|| -> Result<()> {
        for rel in files {
            let dest = base.join(rel);
            let keep = KEEP_IF_EXISTS.iter().any(|k| rel == Path::new(k));
            if keep && dest.exists() {
                continue;
            }
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            let had_old = dest.exists();
            if had_old {
                let old = old_path(&dest);
                let _ = fs::remove_file(&old);
                fs::rename(&dest, &old).with_context(|| format!("파일을 바꿀 수 없습니다: {}", dest.display()))?;
            }
            done.push((dest.clone(), had_old));
            let src = staging.join(rel);
            if fs::rename(&src, &dest).is_err() {
                fs::copy(&src, &dest).with_context(|| format!("새 파일을 넣을 수 없습니다: {}", dest.display()))?;
            }
        }
        Ok(())
    })();

    if result.is_err() {
        for (dest, had_old) in done.into_iter().rev() {
            let _ = fs::remove_file(&dest);
            if had_old {
                let _ = fs::rename(old_path(&dest), &dest);
            }
        }
    }
    result
}

/// 지난 업데이트에서 남은 `*.old`와 받은 파일을 지운다 (실행할 때마다).
pub fn cleanup() {
    cleanup_in(&base_dir());
}

fn cleanup_in(base: &Path) {
    let _ = fs::remove_dir_all(update_dir(base));
    let walker = walkdir::WalkDir::new(base).max_depth(5).into_iter().filter_entry(|e| {
        e.depth() != 1
            || !e.file_type().is_dir()
            || !SKIP_DIRS.contains(&e.file_name().to_string_lossy().to_lowercase().as_str())
    });
    for entry in walker.flatten() {
        if entry.file_type().is_file() && entry.file_name().to_string_lossy().ends_with(".old") {
            let _ = fs::remove_file(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("locall-update-test-{}-{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn versions() {
        assert_eq!(parse_version("v1.2.3"), Some((1, 2, 3)));
        assert_eq!(parse_version("0.2"), Some((0, 2, 0)));
        assert_eq!(parse_version("1.0.0-beta.1"), Some((1, 0, 0)));
        assert_eq!(parse_version("abc"), None);
        assert_eq!(parse_version("1.2.3.4"), None);
        assert!(is_newer("v0.1.1", "0.1.0"));
        assert!(is_newer("v0.10.0", "0.9.9"));
        assert!(!is_newer("v0.1.0", "0.1.0"));
        assert!(!is_newer("v0.0.9", "0.1.0"));
        assert!(!is_newer("garbage", "0.1.0"));
    }

    #[test]
    fn release_json() {
        let json = r#"{"tag_name":"v0.2.0","draft":false,"prerelease":false,"assets":[
            {"name":"LocalL-windows-x64.zip","size":123,"browser_download_url":"https://x/z.zip"},
            {"name":"LocalL-windows-x64.zip.sha256","size":90,"browser_download_url":"https://x/z.sha256"}]}"#;
        let r = parse_release(json, "0.1.0").unwrap().unwrap();
        assert_eq!(r.version, "0.2.0");
        assert_eq!(r.zip_url, "https://x/z.zip");
        assert_eq!(r.sha_url, "https://x/z.sha256");
        assert_eq!(r.size, 123);
        assert!(parse_release(json, "0.2.0").unwrap().is_none());
        let no_asset = r#"{"tag_name":"v0.3.0","assets":[]}"#;
        assert!(parse_release(no_asset, "0.1.0").is_err());
    }

    #[test]
    fn safe_paths() {
        assert!(is_safe_relative(Path::new("locall.exe")));
        assert!(is_safe_relative(Path::new("assets/ui/index.html")));
        assert!(!is_safe_relative(Path::new("../evil.exe")));
        assert!(!is_safe_relative(Path::new("/etc/passwd")));
        assert!(!is_safe_relative(Path::new("data/index.db")));
        assert!(!is_safe_relative(Path::new("models/onnx/model.onnx")));
    }

    #[test]
    fn apply_replaces_keeps_config_and_cleans_up() {
        let base = temp_dir("base");
        let staging = temp_dir("staging");
        fs::write(base.join("locall.exe"), "old exe").unwrap();
        fs::write(base.join("config.json"), "user config").unwrap();
        fs::create_dir_all(base.join("data")).unwrap();
        fs::write(base.join("data/keep.old"), "user data").unwrap();

        fs::write(staging.join("locall.exe"), "new exe").unwrap();
        fs::write(staging.join("config.json"), "default config").unwrap();
        fs::create_dir_all(staging.join("assets/ui")).unwrap();
        fs::write(staging.join("assets/ui/index.html"), "new ui").unwrap();
        let files: Vec<PathBuf> = ["locall.exe", "config.json", "assets/ui/index.html"].iter().map(PathBuf::from).collect();

        apply(&staging, &base, &files).unwrap();
        assert_eq!(fs::read_to_string(base.join("locall.exe")).unwrap(), "new exe");
        assert_eq!(fs::read_to_string(base.join("locall.exe.old")).unwrap(), "old exe");
        assert_eq!(fs::read_to_string(base.join("config.json")).unwrap(), "user config");
        assert_eq!(fs::read_to_string(base.join("assets/ui/index.html")).unwrap(), "new ui");

        cleanup_in(&base);
        assert!(!base.join("locall.exe.old").exists());
        assert!(base.join("data/keep.old").exists(), "data 폴더는 건드리지 않는다");

        let _ = fs::remove_dir_all(&base);
        let _ = fs::remove_dir_all(&staging);
    }

    fn make_zip(path: &Path, names: &[&str]) {
        let mut w = zip::ZipWriter::new(File::create(path).unwrap());
        for name in names {
            w.start_file(*name, zip::write::SimpleFileOptions::default()).unwrap();
            w.write_all(name.as_bytes()).unwrap();
        }
        w.finish().unwrap();
    }

    #[test]
    fn extract_zip() {
        let dir = temp_dir("extract");
        let good = dir.join("good.zip");
        make_zip(&good, &["locall.exe", "assets/ui/icon.svg", "assets\\ui\\index.html"]);
        let files = extract(&good, &dir.join("out")).unwrap();
        assert_eq!(files.len(), 3);
        assert_eq!(fs::read_to_string(dir.join("out/assets/ui/icon.svg")).unwrap(), "assets/ui/icon.svg");

        for bad in ["../evil.exe", "data/index.db", "/abs.exe"] {
            let z = dir.join("bad.zip");
            make_zip(&z, &["locall.exe", bad]);
            assert!(extract(&z, &dir.join("bad-out")).is_err(), "{} 는 거부해야 한다", bad);
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_rolls_back_on_failure() {
        let base = temp_dir("rollback");
        let staging = temp_dir("rollback-staging");
        fs::write(base.join("locall.exe"), "old exe").unwrap();
        fs::write(staging.join("locall.exe"), "new exe").unwrap();
        // 두 번째 파일은 staging에 없어서 실패한다
        let files: Vec<PathBuf> = ["locall.exe", "missing.dll"].iter().map(PathBuf::from).collect();

        assert!(apply(&staging, &base, &files).is_err());
        assert_eq!(fs::read_to_string(base.join("locall.exe")).unwrap(), "old exe");
        assert!(!base.join("locall.exe.old").exists());

        let _ = fs::remove_dir_all(&base);
        let _ = fs::remove_dir_all(&staging);
    }
}
