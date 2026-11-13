//! AI 모델 파일 받기. 배포본에는 모델(약 1.7GB)을 넣지 않고, 처음 실행할 때 Hugging Face에서 받는다.
//!
//! - 리비전을 고정하고 파일마다 크기와 SHA-256을 확인한다. 다른 파일이면 버리고 다시 받는다.
//! - `이름.part`에 받다가 끊기면 다음에 이어 받는다 (HTTP Range).
//! - 검색 앱과 색인기가 동시에 받지 않도록 `models/onnx/.download.lock`을 잡는다.

use crate::config::base_dir;
use anyhow::{anyhow, bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const REPO: &str = "onnx-community/embeddinggemma-2-ONNX";
/// 받는 리비전 (커밋). 바꾸면 아래 크기·SHA-256도 함께 바꿔야 한다.
pub const REVISION: &str = "daa72c51243991dfcaf9f9137d2c573d8f7790c0";

pub struct ModelFile {
    /// 저장소 안의 경로이자 `models/onnx/` 아래 경로
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

pub const FILES: &[ModelFile] = &[
    ModelFile {
        path: "tokenizer.json",
        size: 32_170_510,
        sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    },
    ModelFile {
        path: "onnx/model.onnx",
        size: 421_421,
        sha256: "bc47de15f81208a5c99e5ab10f746d5e33b51ea228b7dc0bef9c133a94f1c1c3",
    },
    ModelFile {
        path: "onnx/model.onnx_data",
        size: 1_084_170_240,
        sha256: "9fd452bfc6916e5a92f080f8dada35598ca4512b21c65897d4c5974d5f9630a2",
    },
    ModelFile {
        path: "onnx/vision_encoder.onnx",
        size: 121_253,
        sha256: "a3fb44f8b5b152cfee6afe16e2cd69e774b7e1d3b822d1db1190ddf7b08b575e",
    },
    ModelFile {
        path: "onnx/vision_encoder.onnx_data",
        size: 671_026_176,
        sha256: "ca8d28c99feb8129618f141978a27adae25090e0dadb68a30c235e191c01e37e",
    },
];

/// 받는 중인 상황
pub enum Progress<'a> {
    /// 다른 프로그램(검색 앱이나 색인기)이 받는 중이라 기다린다
    Waiting,
    /// 받은 양 / 받아야 할 전체 양 (바이트), 지금 받는 파일
    Bytes { done: u64, total: u64, file: &'a str },
    /// 받은 파일이 맞는지 확인하는 중 (이어 받기 전 앞부분 확인 포함)
    Checking { file: &'a str },
}

/// `models/onnx/`
pub fn models_dir() -> PathBuf {
    base_dir().join("models").join("onnx")
}

fn local_path(dir: &Path, f: &ModelFile) -> PathBuf {
    dir.join(f.path.replace('/', std::path::MAIN_SEPARATOR_STR))
}

fn part_path(p: &Path) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

/// 없거나 크기가 다른 파일들. 실행할 때마다 부르므로 해시는 보지 않고 크기만 본다
/// (해시는 받을 때 확인한다).
pub fn missing_files() -> Vec<&'static ModelFile> {
    missing_in(&models_dir())
}

fn missing_in(dir: &Path) -> Vec<&'static ModelFile> {
    FILES
        .iter()
        .filter(|f| std::fs::metadata(local_path(dir, f)).map_or(true, |m| m.len() != f.size))
        .collect()
}

/// 아직 받아야 할 크기 (바이트). 이어 받을 `.part`가 있으면 그만큼 뺀다.
pub fn missing_bytes() -> u64 {
    let dir = models_dir();
    missing_in(&dir)
        .iter()
        .map(|f| f.size - std::fs::metadata(part_path(&local_path(&dir, f))).map_or(0, |m| m.len().min(f.size)))
        .sum()
}

/// 모델 파일이 하나라도 없는지
pub fn models_missing() -> bool {
    !missing_files().is_empty()
}

/// 모델 파일이 다 있으면 바로 돌아오고, 없으면 받는다. 받는 동안 `progress`를 자주 부른다.
pub fn ensure_models(progress: &mut dyn FnMut(Progress)) -> Result<()> {
    ensure_in(&models_dir(), progress)
}

fn ensure_in(dir: &Path, progress: &mut dyn FnMut(Progress)) -> Result<()> {
    if missing_in(dir).is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(dir.join("onnx")).with_context(|| format!("폴더를 만들 수 없습니다: {}", dir.display()))?;
    let lock = acquire_lock(dir, progress)?;
    let result = download_missing(dir, progress);
    drop(lock);
    // 다른 쪽이 잡고 있으면 지워지지 않는다 (그래도 괜찮다)
    let _ = std::fs::remove_file(dir.join(".download.lock"));
    result
}

fn download_missing(dir: &Path, progress: &mut dyn FnMut(Progress)) -> Result<()> {

    // 기다리는 동안 다른 쪽이 다 받았을 수 있다
    let todo = missing_in(dir);
    let total: u64 = todo.iter().map(|f| f.size).sum();
    let mut done: u64 = todo
        .iter()
        .map(|f| std::fs::metadata(part_path(&local_path(dir, f))).map_or(0, |m| m.len().min(f.size)))
        .sum();

    let agent = make_agent();
    for f in todo {
        download_one(&agent, dir, f, &mut done, total, progress)?;
    }
    Ok(())
}

/// 검색 앱과 색인기가 함께 받지 않게. 잠금 파일을 공유 없이 열어 두고, 이미 열려 있으면 기다린다.
/// 프로그램이 죽으면 Windows가 파일을 닫으므로 잠금이 남지 않는다.
fn acquire_lock(dir: &Path, progress: &mut dyn FnMut(Progress)) -> Result<File> {
    let path = dir.join(".download.lock");
    let mut told = false;
    loop {
        let mut opts = OpenOptions::new();
        opts.write(true).create(true).truncate(false);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            opts.share_mode(0);
        }
        match opts.open(&path) {
            Ok(f) => return Ok(f),
            // ERROR_SHARING_VIOLATION(32): 다른 쪽이 받는 중
            Err(e) if e.raw_os_error() == Some(32) => {
                if !told {
                    progress(Progress::Waiting);
                    told = true;
                }
                std::thread::sleep(Duration::from_secs(1));
            }
            Err(e) => return Err(e).with_context(|| format!("잠금 파일을 만들 수 없습니다: {}", path.display())),
        }
    }
}

pub(crate) fn make_agent() -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .user_agent("LocalL")
        .timeout_connect(Some(Duration::from_secs(20)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .build();
    ureq::Agent::new_with_config(config)
}

fn url(f: &ModelFile) -> String {
    format!("https://huggingface.co/{}/resolve/{}/{}", REPO, REVISION, f.path)
}

/// 파일 하나 받기. 끊기면 몇 번 이어서 다시 시도한다.
fn download_one(
    agent: &ureq::Agent,
    dir: &Path,
    f: &ModelFile,
    done: &mut u64,
    total: u64,
    progress: &mut dyn FnMut(Progress),
) -> Result<()> {
    let dest = local_path(dir, f);
    let part = part_path(&dest);
    let name = f.path.rsplit('/').next().unwrap_or(f.path);

    // 크기가 다른 옛 파일은 치운다 (같은 크기면 missing에 안 들어온다)
    if dest.exists() {
        std::fs::remove_file(&dest).with_context(|| format!("옛 파일을 지울 수 없습니다: {}", dest.display()))?;
    }

    // 이어 받은 파일이 틀리면 앞부분이 상했을 수 있으니 처음부터 한 번 더 받는다
    let resumed = std::fs::metadata(&part).map_or(false, |m| m.len() > 0);
    match download_once(agent, f, &part, name, done, total, progress) {
        Err(e) if resumed && e.downcast_ref::<Corrupt>().is_some() => {
            download_once(agent, f, &part, name, done, total, progress)?;
        }
        r => r?,
    }
    std::fs::rename(&part, &dest).with_context(|| format!("파일 이름을 바꿀 수 없습니다: {}", dest.display()))?;
    Ok(())
}

/// 받은 파일의 SHA-256이 다르다 (`.part`는 지웠다)
#[derive(Debug)]
struct Corrupt(String);

impl std::fmt::Display for Corrupt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}이(가) 손상되어 지웠습니다. 다시 실행하면 처음부터 받습니다.", self.0)
    }
}

impl std::error::Error for Corrupt {}

/// `.part`에 이어 받아 끝까지 채우고 SHA-256을 확인한다. 틀리면 `.part`를 지우고 `Corrupt`.
fn download_once(
    agent: &ureq::Agent,
    f: &ModelFile,
    part: &Path,
    name: &str,
    done: &mut u64,
    total: u64,
    progress: &mut dyn FnMut(Progress),
) -> Result<()> {
    // 이미 받아 둔 앞부분을 해시에 먼저 넣는다
    let mut hasher = Sha256::new();
    let mut have = std::fs::metadata(part).map_or(0, |m| m.len());
    if have > f.size {
        std::fs::remove_file(part)?;
        *done = done.saturating_sub(have.min(f.size));
        have = 0;
    }
    if have > 0 {
        progress(Progress::Checking { file: name });
        let mut r = File::open(part)?;
        let mut buf = vec![0u8; 1 << 20];
        let mut left = have;
        while left > 0 {
            let want = left.min(buf.len() as u64) as usize;
            let n = r.read(&mut buf[..want])?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            left -= n as u64;
        }
    }

    let mut attempts = 0;
    while have < f.size {
        match fetch_range(agent, f, part, &mut have, &mut hasher, done, total, name, progress) {
            Ok(()) => {}
            Err(e) => {
                attempts += 1;
                if attempts >= 5 {
                    return Err(e).with_context(|| format!("{}을(를) 받지 못했습니다", name));
                }
                std::thread::sleep(Duration::from_secs(2 * attempts));
            }
        }
    }

    progress(Progress::Checking { file: name });
    let got = hex(&hasher.finalize());
    if got != f.sha256 {
        let _ = std::fs::remove_file(part);
        *done = done.saturating_sub(f.size);
        return Err(Corrupt(name.to_string()).into());
    }
    Ok(())
}

/// `have`바이트 뒤부터 끝까지 받아 `.part`에 덧붙인다.
#[allow(clippy::too_many_arguments)]
fn fetch_range(
    agent: &ureq::Agent,
    f: &ModelFile,
    part: &Path,
    have: &mut u64,
    hasher: &mut Sha256,
    done: &mut u64,
    total: u64,
    name: &str,
    progress: &mut dyn FnMut(Progress),
) -> Result<()> {
    let mut req = agent.get(&url(f));
    if *have > 0 {
        req = req.header("Range", &format!("bytes={}-", *have));
    }
    let mut resp = req.call().map_err(|e| anyhow!("{}: {}", url(f), e))?;

    let mut out = OpenOptions::new().create(true).write(true).truncate(false).open(part)?;
    match resp.status().as_u16() {
        206 => {
            out.seek(SeekFrom::Start(*have))?;
            out.set_len(*have)?;
        }
        200 => {
            // 서버가 이어 받기를 안 받아 줬다: 처음부터
            *done = done.saturating_sub(*have);
            *have = 0;
            *hasher = Sha256::new();
            out.set_len(0)?;
            out.seek(SeekFrom::Start(0))?;
        }
        s => bail!("{}: HTTP {}", url(f), s),
    }

    let mut reader = resp.body_mut().with_config().limit(f.size + 1).reader();
    let mut buf = vec![0u8; 256 * 1024];
    let mut last = Instant::now();
    progress(Progress::Bytes { done: *done, total, file: name });
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        if *have + n as u64 > f.size {
            bail!("{}: 예상보다 큽니다", name);
        }
        out.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        *have += n as u64;
        *done += n as u64;
        if last.elapsed() >= Duration::from_millis(250) {
            progress(Progress::Bytes { done: *done, total, file: name });
            last = Instant::now();
        }
    }
    out.flush()?;
    progress(Progress::Bytes { done: *done, total, file: name });
    if *have < f.size {
        bail!("{}: 연결이 끊겼습니다 ({}/{} 바이트)", name, *have, f.size);
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// 사람이 읽는 크기 (MB/GB)
pub fn human(bytes: u64) -> String {
    let mb = bytes as f64 / 1_048_576.0;
    if mb >= 1024.0 {
        format!("{:.2}GB", mb / 1024.0)
    } else if mb < 10.0 {
        format!("{:.1}MB", mb)
    } else {
        format!("{:.0}MB", mb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_add_up() {
        let total: u64 = FILES.iter().map(|f| f.size).sum();
        assert_eq!(total, 1_787_909_600);
        assert!(FILES.iter().all(|f| f.sha256.len() == 64));
    }
}
