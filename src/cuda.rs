//! NVIDIA CUDA GPU 감지 및 CUDA/cuDNN + ONNX Runtime GPU 온디맨드 다운로드.
//!
//! 배포본에는 CUDA 런타임을 넣지 않고, 사용자가 설정(`index_cuda`)이나 `--cuda` 옵션으로
//! GPU 색인을 켠 경우에만 공식 배포처(NuGet Microsoft.ML.OnnxRuntime.Gpu.Windows,
//! PyPI NVIDIA 공식 CUDA 12 / cuDNN 9 wheel)에서 필요한 DLL만 `models/onnx/cuda/`에 받아 푼다.
//!
//! - 각 아카이브는 크기와 SHA-256을 검증한다.
//! - `.part`에 이어 받기(HTTP Range)를 지원하며, DLL 추출이 끝나면 임시 아카이브는 지운다.
//! - 시스템 `PATH`에 이미 CUDA 12 / cuDNN 9 DLL이 있으면 해당 패키지는 건너뛴다.

use crate::config::base_dir;
use crate::download::make_agent;
use anyhow::{anyhow, bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// 최소 VRAM 기준 (실제 4GB 카드가 OS 예약분 제외 ~3.7~3.9GiB로 보고되는 경우를 감안)
pub const MIN_VRAM_BYTES: u64 = 3_500 * 1024 * 1024;
/// 최소 Compute Capability (Pascal, GTX 10 시리즈 이상)
pub const MIN_COMPUTE_MAJOR: i32 = 6;
/// 최소 CUDA 드라이버 버전 (CUDA 12.0 = 12000)
pub const MIN_DRIVER_CUDA: i32 = 12000;

/// 감지된 NVIDIA GPU 정보
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub device_id: i32,
    pub name: String,
    pub vram_bytes: u64,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub driver_cuda: i32,
}

impl GpuInfo {
    pub fn vram_gb(&self) -> f64 {
        self.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn driver_cuda_str(&self) -> String {
        format!("{}.{}", self.driver_cuda / 1000, (self.driver_cuda % 1000) / 10)
    }
}

/// GPU 사용 가능 여부 확인 결과
#[derive(Debug, Clone)]
pub enum GpuCheck {
    Ready(GpuInfo),
    Unavailable(String),
}

/// `models/onnx/cuda/`
pub fn cuda_dir() -> PathBuf {
    base_dir().join("models").join("onnx").join("cuda")
}

/// 공식 배포처 아카이브 정의 (NuGet `.nupkg` / PyPI `.whl` 모두 ZIP 형식)
pub struct CudaArchive {
    pub label: &'static str,
    pub url: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    /// 이 DLL들이 모두 준비되어 있으면(로컬 또는 시스템 PATH) 아카이브를 건너뛴다
    pub required_dlls: &'static [&'static str],
    /// 시스템 PATH에 이미 있을 때 건너뛰기를 허용할지 (ORT GPU DLL은 반드시 로컬 cuda/에 필요)
    pub allow_system_path: bool,
    /// true면 아카이브 내 모든 `.dll`을 추출(cuDNN 모듈 DLL 전체), false면 `required_dlls`만 추출
    pub extract_all_dlls: bool,
}

pub const ARCHIVES: &[CudaArchive] = &[
    CudaArchive {
        label: "ONNX Runtime GPU 1.24.4",
        url: "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.gpu.windows/1.24.4/microsoft.ml.onnxruntime.gpu.windows.1.24.4.nupkg",
        size: 172_417_405,
        sha256: "e897a13d318483e71e1eef91005634846201ab50bc6a582ae913dc5a6ccc0240",
        required_dlls: &[
            "onnxruntime.dll",
            "onnxruntime_providers_shared.dll",
            "onnxruntime_providers_cuda.dll",
        ],
        allow_system_path: false,
        extract_all_dlls: false,
    },
    CudaArchive {
        label: "CUDA Runtime 12.9",
        url: "https://files.pythonhosted.org/packages/59/df/e7c3a360be4f7b93cee39271b792669baeb3846c58a4df6dfcf187a7ffab/nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl",
        size: 3_591_604,
        sha256: "8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891",
        required_dlls: &["cudart64_12.dll"],
        allow_system_path: true,
        extract_all_dlls: false,
    },
    CudaArchive {
        label: "cuBLAS 12.9",
        url: "https://files.pythonhosted.org/packages/20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
        size: 553_162_896,
        sha256: "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661",
        required_dlls: &["cublas64_12.dll", "cublasLt64_12.dll"],
        allow_system_path: true,
        extract_all_dlls: false,
    },
    CudaArchive {
        label: "cuFFT 11.4",
        url: "https://files.pythonhosted.org/packages/20/ee/29955203338515b940bd4f60ffdbc073428f25ef9bfbce44c9a066aedc5c/nvidia_cufft_cu12-11.4.1.4-py3-none-win_amd64.whl",
        size: 200_067_309,
        sha256: "8e5bfaac795e93f80611f807d42844e8e27e340e0cde270dcb6c65386d795b80",
        required_dlls: &["cufft64_11.dll"],
        allow_system_path: true,
        extract_all_dlls: false,
    },
    CudaArchive {
        label: "cuDNN 9.28",
        url: "https://files.pythonhosted.org/packages/c7/bd/1d6709d2223e99f04079f16005f8ae6e11a14ee16b5eab6833686cf72fe1/nvidia_cudnn_cu12-9.28.1.4-py3-none-win_amd64.whl",
        size: 696_300_805,
        sha256: "3467aafa9b260e6adf5fbbfc0e21d180133305b42032aa1160be2d3142bb8b09",
        required_dlls: &[
            "cudnn64_9.dll",
            "cudnn_ops64_9.dll",
            "cudnn_cnn64_9.dll",
            "cudnn_adv64_9.dll",
            "cudnn_graph64_9.dll",
            "cudnn_heuristic64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll",
        ],
        allow_system_path: true,
        extract_all_dlls: true,
    },
];

/// CUDA 런타임 다운로드/추출 진행 상태
pub enum CudaProgress<'a> {
    Waiting,
    Bytes { done: u64, total: u64, file: &'a str },
    Checking { file: &'a str },
    Extracting { file: &'a str },
}

// ---------------------------------------------------------------------------
// 1. NVIDIA GPU 감지 (nvcuda.dll Driver API 동적 호출)
// ---------------------------------------------------------------------------

#[cfg(windows)]
pub fn detect_gpu() -> Option<GpuInfo> {
    use std::ffi::{c_char, c_int, CStr};
    use windows::core::{s, w};
    use windows::Win32::Foundation::FreeLibrary;
    use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

    type CuInit = unsafe extern "C" fn(u32) -> c_int;
    type CuDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> c_int;
    type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> c_int;
    type CuDeviceGet = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
    type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
    type CuDeviceTotalMem = unsafe extern "C" fn(*mut usize, c_int) -> c_int;
    type CuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;

    const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
    const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

    unsafe {
        let lib = LoadLibraryW(w!("nvcuda.dll")).ok()?;
        let res = (|| -> Option<GpuInfo> {
            let cu_init: CuInit = std::mem::transmute(GetProcAddress(lib, s!("cuInit"))?);
            let cu_drv: CuDriverGetVersion = std::mem::transmute(GetProcAddress(lib, s!("cuDriverGetVersion"))?);
            let cu_cnt: CuDeviceGetCount = std::mem::transmute(GetProcAddress(lib, s!("cuDeviceGetCount"))?);
            let cu_dev: CuDeviceGet = std::mem::transmute(GetProcAddress(lib, s!("cuDeviceGet"))?);
            let cu_name: CuDeviceGetName = std::mem::transmute(GetProcAddress(lib, s!("cuDeviceGetName"))?);
            let cu_mem: CuDeviceTotalMem = std::mem::transmute(
                GetProcAddress(lib, s!("cuDeviceTotalMem_v2")).or_else(|| GetProcAddress(lib, s!("cuDeviceTotalMem")))?,
            );
            let cu_attr: CuDeviceGetAttribute =
                std::mem::transmute(GetProcAddress(lib, s!("cuDeviceGetAttribute"))?);

            if cu_init(0) != 0 {
                return None;
            }
            let mut driver_cuda: c_int = 0;
            if cu_drv(&mut driver_cuda) != 0 {
                return None;
            }
            let mut count: c_int = 0;
            if cu_cnt(&mut count) != 0 || count <= 0 {
                return None;
            }

            let mut best: Option<GpuInfo> = None;
            for idx in 0..count {
                let mut dev: c_int = 0;
                if cu_dev(&mut dev, idx) != 0 {
                    continue;
                }
                let mut buf = [0u8; 256];
                let name = if cu_name(buf.as_mut_ptr() as *mut c_char, buf.len() as c_int, dev) == 0 {
                    CStr::from_ptr(buf.as_ptr() as *const c_char).to_string_lossy().trim().to_string()
                } else {
                    format!("NVIDIA GPU #{}", idx)
                };
                let mut mem: usize = 0;
                if cu_mem(&mut mem, dev) != 0 {
                    continue;
                }
                let mut cc_major: c_int = 0;
                let mut cc_minor: c_int = 0;
                let _ = cu_attr(&mut cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
                let _ = cu_attr(&mut cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);

                let info = GpuInfo {
                    device_id: idx,
                    name,
                    vram_bytes: mem as u64,
                    cc_major,
                    cc_minor,
                    driver_cuda,
                };
                if best.as_ref().map_or(true, |b| info.vram_bytes > b.vram_bytes) {
                    best = Some(info);
                }
            }
            best
        })();
        let _ = FreeLibrary(lib);
        res
    }
}

#[cfg(not(windows))]
pub fn detect_gpu() -> Option<GpuInfo> {
    None
}

/// CUDA 색인 실행 가능 여부를 사양 기준으로 점검한다.
pub fn check_gpu() -> GpuCheck {
    let Some(gpu) = detect_gpu() else {
        return GpuCheck::Unavailable("NVIDIA CUDA GPU 또는 드라이버를 찾지 못했습니다".to_string());
    };
    if gpu.driver_cuda < MIN_DRIVER_CUDA {
        return GpuCheck::Unavailable(format!(
            "{}: 드라이버 CUDA 지원 버전({})이 낮습니다 (12.0 이상 필요, 드라이버 업데이트 권장)",
            gpu.name,
            gpu.driver_cuda_str()
        ));
    }
    if gpu.cc_major < MIN_COMPUTE_MAJOR {
        return GpuCheck::Unavailable(format!(
            "{}: 아키텍처(CC {}.{})가 오래되어 지원되지 않습니다 (6.0 / GTX 10 시리즈 이상 필요)",
            gpu.name, gpu.cc_major, gpu.cc_minor
        ));
    }
    if gpu.vram_bytes < MIN_VRAM_BYTES {
        return GpuCheck::Unavailable(format!(
            "{}: VRAM({:.1}GB)이 부족합니다 (최소 4GB 필요)",
            gpu.name,
            gpu.vram_gb()
        ));
    }
    GpuCheck::Ready(gpu)
}

// ---------------------------------------------------------------------------
// 2. DLL 존재 확인 및 온디맨드 다운로드 / 추출
// ---------------------------------------------------------------------------

/// `dll_name`이 `models/onnx/cuda/` 또는 (허용 시) 시스템 `PATH`에 존재하는지 확인
fn dll_available(dir: &Path, dll_name: &str, allow_system_path: bool) -> bool {
    if dir.join(dll_name).is_file() {
        return true;
    }
    if !allow_system_path {
        return false;
    }
    find_on_path(dll_name).is_some()
}

fn find_on_path(dll_name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let p = dir.join(dll_name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn archive_missing(dir: &Path, a: &CudaArchive) -> bool {
    a.required_dlls
        .iter()
        .any(|dll| !dll_available(dir, dll, a.allow_system_path))
}

/// 아직 받아야 할 아카이브 목록
pub fn missing_archives() -> Vec<&'static CudaArchive> {
    let dir = cuda_dir();
    ARCHIVES.iter().filter(|a| archive_missing(&dir, a)).collect()
}

/// CUDA 런타임이 아직 완전히 준비되지 않았는지
pub fn cuda_runtime_missing() -> bool {
    !missing_archives().is_empty()
}

fn part_path(dir: &Path, a: &CudaArchive) -> PathBuf {
    let fname = a.url.rsplit('/').next().unwrap_or("archive.zip");
    dir.join(format!("{}.part", fname))
}

/// 아직 받아야 할 총 바이트 (`.part` 이어받기 분 제외)
pub fn missing_bytes() -> u64 {
    let dir = cuda_dir();
    missing_archives()
        .iter()
        .map(|a| a.size - fs::metadata(part_path(&dir, a)).map_or(0, |m| m.len().min(a.size)))
        .sum()
}

/// CUDA 런타임 DLL이 모두 준비되어 있으면 바로 돌아오고, 부족한 패키지들만 공식 출처에서 받아 추출한다.
pub fn ensure_cuda_runtime(progress: &mut dyn FnMut(CudaProgress)) -> Result<()> {
    let dir = cuda_dir();
    if missing_archives().is_empty() {
        return Ok(());
    }
    fs::create_dir_all(&dir).with_context(|| format!("폴더를 만들 수 없습니다: {}", dir.display()))?;
    let lock = acquire_lock(&dir, progress)?;
    let result = download_and_extract(&dir, progress);
    drop(lock);
    let _ = fs::remove_file(dir.join(".download.lock"));
    result
}

fn acquire_lock(dir: &Path, progress: &mut dyn FnMut(CudaProgress)) -> Result<File> {
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
            Err(e) if e.raw_os_error() == Some(32) => {
                if !told {
                    progress(CudaProgress::Waiting);
                    told = true;
                }
                std::thread::sleep(Duration::from_secs(1));
            }
            Err(e) => return Err(e).with_context(|| format!("잠금 파일을 만들 수 없습니다: {}", path.display())),
        }
    }
}

fn download_and_extract(dir: &Path, progress: &mut dyn FnMut(CudaProgress)) -> Result<()> {
    let todo: Vec<&'static CudaArchive> = ARCHIVES.iter().filter(|a| archive_missing(dir, a)).collect();
    if todo.is_empty() {
        return Ok(());
    }
    let total: u64 = todo.iter().map(|a| a.size).sum();
    let mut done: u64 = todo
        .iter()
        .map(|a| fs::metadata(part_path(dir, a)).map_or(0, |m| m.len().min(a.size)))
        .sum();

    let agent = make_agent();
    for a in todo {
        let part = part_path(dir, a);
        match download_once(&agent, a, &part, &mut done, total, progress) {
            Ok(()) => {}
            Err(e) if e.downcast_ref::<Corrupt>().is_some() => {
                download_once(&agent, a, &part, &mut done, total, progress)?;
            }
            Err(e) => return Err(e),
        }
        progress(CudaProgress::Extracting { file: a.label });
        extract_dlls(&part, dir, a)?;
        let _ = fs::remove_file(&part);
    }
    Ok(())
}

#[derive(Debug)]
struct Corrupt(String);

impl std::fmt::Display for Corrupt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} 파일이 손상되어 지웠습니다. 다시 시도합니다.", self.0)
    }
}

impl std::error::Error for Corrupt {}

fn download_once(
    agent: &ureq::Agent,
    a: &CudaArchive,
    part: &Path,
    done: &mut u64,
    total: u64,
    progress: &mut dyn FnMut(CudaProgress),
) -> Result<()> {
    let mut hasher = Sha256::new();
    let mut have = fs::metadata(part).map_or(0, |m| m.len());
    if have > a.size {
        fs::remove_file(part)?;
        *done = done.saturating_sub(have.min(a.size));
        have = 0;
    }
    if have > 0 {
        progress(CudaProgress::Checking { file: a.label });
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
    while have < a.size {
        match fetch_range(agent, a, part, &mut have, &mut hasher, done, total, progress) {
            Ok(()) => {}
            Err(e) => {
                attempts += 1;
                if attempts >= 5 {
                    return Err(e).with_context(|| format!("{}을(를) 받지 못했습니다", a.label));
                }
                std::thread::sleep(Duration::from_secs(2 * attempts));
            }
        }
    }

    progress(CudaProgress::Checking { file: a.label });
    let got = hex(&hasher.finalize());
    if got != a.sha256 {
        let _ = fs::remove_file(part);
        *done = done.saturating_sub(a.size);
        return Err(Corrupt(a.label.to_string()).into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn fetch_range(
    agent: &ureq::Agent,
    a: &CudaArchive,
    part: &Path,
    have: &mut u64,
    hasher: &mut Sha256,
    done: &mut u64,
    total: u64,
    progress: &mut dyn FnMut(CudaProgress),
) -> Result<()> {
    let mut req = agent.get(a.url);
    if *have > 0 {
        req = req.header("Range", &format!("bytes={}-", *have));
    }
    let mut resp = req.call().map_err(|e| anyhow!("{}: {}", a.url, e))?;

    let mut out = OpenOptions::new().create(true).write(true).truncate(false).open(part)?;
    match resp.status().as_u16() {
        206 => {
            out.seek(SeekFrom::Start(*have))?;
            out.set_len(*have)?;
        }
        200 => {
            *done = done.saturating_sub(*have);
            *have = 0;
            *hasher = Sha256::new();
            out.set_len(0)?;
            out.seek(SeekFrom::Start(0))?;
        }
        s => bail!("{}: HTTP {}", a.url, s),
    }

    let mut reader = resp.body_mut().with_config().limit(a.size + 1).reader();
    let mut buf = vec![0u8; 256 * 1024];
    let mut last = Instant::now();
    progress(CudaProgress::Bytes { done: *done, total, file: a.label });
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        if *have + n as u64 > a.size {
            bail!("{}: 예상보다 큽니다", a.label);
        }
        out.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        *have += n as u64;
        *done += n as u64;
        if last.elapsed() >= Duration::from_millis(250) {
            progress(CudaProgress::Bytes { done: *done, total, file: a.label });
            last = Instant::now();
        }
    }
    out.flush()?;
    progress(CudaProgress::Bytes { done: *done, total, file: a.label });
    if *have < a.size {
        bail!("{}: 연결이 끊겼습니다 ({}/{} 바이트)", a.label, *have, a.size);
    }
    Ok(())
}

/// 검증된 ZIP 아카이브(.nupkg / .whl)에서 필요한 DLL만 `dest_dir`에 원자적으로 추출한다.
fn extract_dlls(archive_path: &Path, dest_dir: &Path, a: &CudaArchive) -> Result<()> {
    let f = File::open(archive_path).with_context(|| format!("아카이브를 열 수 없습니다: {}", archive_path.display()))?;
    let mut zip = zip::ZipArchive::new(f).with_context(|| format!("ZIP 해석 실패: {}", a.label))?;
    let mut extracted = 0usize;

    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().replace('\\', "/");
        // NuGet의 경우 다른 아키텍처(x86/arm64) DLL이 섞여 있을 수 있으므로 win-x64만 허용
        if name.starts_with("runtimes/") && !name.contains("win-x64") {
            continue;
        }
        let Some(base) = name.rsplit('/').next() else {
            continue;
        };
        if !base.to_ascii_lowercase().ends_with(".dll") {
            continue;
        }
        let want = if a.extract_all_dlls {
            true
        } else {
            a.required_dlls.iter().any(|r| r.eq_ignore_ascii_case(base))
        };
        if !want {
            continue;
        }
        let dest = dest_dir.join(base);
        let tmp = dest_dir.join(format!("{}.tmp", base));
        {
            let mut out = File::create(&tmp)?;
            std::io::copy(&mut entry, &mut out)?;
            out.flush()?;
        }
        if dest.exists() {
            let _ = fs::remove_file(&dest);
        }
        fs::rename(&tmp, &dest)
            .with_context(|| format!("DLL 배치 실패: {}", dest.display()))?;
        extracted += 1;
    }

    // 필수 DLL이 모두 존재하는지 최종 확인
    for dll in a.required_dlls {
        if !dest_dir.join(dll).is_file() {
            bail!("{}에서 필수 DLL({})을 찾지 못했습니다 (추출 {}개)", a.label, dll, extracted);
        }
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

// ---------------------------------------------------------------------------
// 3. CUDA DLL 경로 등록 및 프리로드
// ---------------------------------------------------------------------------

/// `models/onnx/cuda/`를 Windows DLL 검색 경로에 추가하고 의존 DLL을 순서대로 프리로드한다.
/// 준비된 CUDA용 `onnxruntime.dll` 경로를 반환한다.
#[cfg(windows)]
pub fn prepare_cuda_dlls() -> Option<PathBuf> {
    use std::os::windows::ffi::OsStrExt;
    use windows::core::PCWSTR;
    use windows::Win32::System::LibraryLoader::{LoadLibraryW, SetDllDirectoryW};

    let dir = cuda_dir();
    let ort_dll = dir.join("onnxruntime.dll");
    let provider_dll = dir.join("onnxruntime_providers_cuda.dll");
    if !ort_dll.is_file() || !provider_dll.is_file() {
        return None;
    }

    let wide_dir: Vec<u16> = dir.as_os_str().encode_wide().chain(Some(0)).collect();
    unsafe {
        let _ = SetDllDirectoryW(PCWSTR(wide_dir.as_ptr()));
    }

    // 의존성 순서대로 프리로드 (로컬 cuda/에 있으면 로컬 우선, 없으면 시스템 PATH)
    const PRELOAD_ORDER: &[&str] = &[
        "cudart64_12.dll",
        "cublasLt64_12.dll",
        "cublas64_12.dll",
        "cufft64_11.dll",
        "cudnn64_9.dll",
        "onnxruntime_providers_shared.dll",
    ];
    for dll in PRELOAD_ORDER {
        let local = dir.join(dll);
        let target: PathBuf = if local.is_file() { local } else { PathBuf::from(dll) };
        let w: Vec<u16> = target.as_os_str().encode_wide().chain(Some(0)).collect();
        unsafe {
            let _ = LoadLibraryW(PCWSTR(w.as_ptr()));
        }
    }
    Some(ort_dll)
}

#[cfg(not(windows))]
pub fn prepare_cuda_dlls() -> Option<PathBuf> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_hashes_valid() {
        assert!(!ARCHIVES.is_empty());
        assert!(ARCHIVES.iter().all(|a| a.sha256.len() == 64 && a.size > 0));
    }
}
