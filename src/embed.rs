//! 검색어·문서·사진 임베딩 (EmbeddingGemma 2).
//!
//! 기존 색인은 PyTorch로 만들었으므로 같은 결과가 나오도록 맞춰 두었다.
//!
//! * 기본: `models/onnx/`의 ONNX 변환본(fp32)을 이 프로세스 안에서 ONNX Runtime(CPU)으로 돌린다.
//!   글자 모델(`model.onnx`)은 검색어·문서, 시각 인코더(`vision_encoder.onnx`)는 사진에 쓴다.
//!   PyTorch와 코사인 0.998 이상으로 일치한다. (fp16은 NaN/품질 저하, q4는 순위가 바뀌어 안 쓴다.
//!   DirectML GPU는 이 모델에서 느리고 시각 인코더가 돌지 않아 `LOCALL_ONNX_DEVICE=gpu`일 때만.)
//! * 예비: ONNX 파일이 없으면 파이썬 작업자(`python -m locall.embed_server`)를 띄워 파이프로
//!   JSON을 주고받는다. 작업자는 이 프로세스가 끝나면 같이 끝난다.

use crate::config::{base_dir, Config};
use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

const QUERY_PREFIX: &str = "task: search result | query: ";
/// 아주 긴 검색어가 들어와도 메모리가 터지지 않게 자르는 길이 (토큰)
const MAX_TOKENS: usize = 2048;

// ---------------------------------------------------------------------------
// 파이썬 작업자
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Reply {
    ok: bool,
    #[serde(default)]
    error: String,
    #[serde(default)]
    vec: Vec<f32>,
    #[serde(default)]
    device: String,
    #[serde(default)]
    dim: usize,
}

struct Worker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    dim: usize,
    device: String,
}

/// 함께 배포되는 파이썬(포터블 → 가상환경 순). 없으면 None.
pub fn find_python() -> Option<PathBuf> {
    let base = base_dir();
    [
        base.join("python").join("python.exe"),
        base.join(".venv").join("Scripts").join("python.exe"),
    ]
    .into_iter()
    .find(|p| p.is_file())
}

impl Worker {
    /// 작업자를 띄우고 모델이 올라올 때까지 기다린다 (수 초~수십 초).
    fn spawn(cfg: &Config) -> Result<Self> {
        let base = base_dir();
        let python = find_python().unwrap_or_else(|| PathBuf::from("python"));

        let log_dir = base.join("data");
        let _ = std::fs::create_dir_all(&log_dir);
        let stderr = std::fs::File::create(log_dir.join("embed.log"))
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null());

        let mut cmd = Command::new(&python);
        cmd.args(["-W", "ignore", "-m", "locall.embed_server"])
            .current_dir(&base)
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr);

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("임베딩 작업자를 실행할 수 없습니다: {:?}", python))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("작업자 stdin 없음"))?;
        let stdout = BufReader::new(child.stdout.take().ok_or_else(|| anyhow!("작업자 stdout 없음"))?);

        let mut w = Self { child, stdin, stdout, dim: 0, device: String::new() };
        let ready = w.read_reply()?;
        if ready.dim != cfg.dim {
            bail!("모델 차원({})이 설정({})과 다릅니다", ready.dim, cfg.dim);
        }
        w.dim = ready.dim;
        w.device = ready.device;
        Ok(w)
    }

    fn read_reply(&mut self) -> Result<Reply> {
        let mut line = String::new();
        if self.stdout.read_line(&mut line)? == 0 {
            bail!("임베딩 작업자가 종료되었습니다 (data/embed.log 참고)");
        }
        let reply: Reply = serde_json::from_str(&line).context("작업자 응답 해석 실패")?;
        if !reply.ok {
            bail!("{}", reply.error);
        }
        Ok(reply)
    }

    fn call(&mut self, req: serde_json::Value) -> Result<Vec<f32>> {
        let mut line = serde_json::to_string(&req)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.flush()?;
        let reply = self.read_reply()?;
        if reply.vec.len() != self.dim {
            bail!("벡터 길이가 맞지 않습니다: {}", reply.vec.len());
        }
        Ok(reply.vec)
    }

    fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ---------------------------------------------------------------------------
// ONNX (검색어와 사진)
// ---------------------------------------------------------------------------

/// 문서 조각 한 개의 최대 토큰 (1200자 조각은 보통 1000토큰 아래)
const MAX_DOC_TOKENS: usize = 4096;
/// 문서 임베딩 한 묶음의 최대 문장 수와 (문장 수 × 가장 긴 길이) 토큰 예산
const MAX_DOC_BATCH: usize = 32;
const DOC_TOKEN_BUDGET: usize = 8192;
const PAD: i64 = 0;
/// GPU에서는 문장 길이를 이 단위로 올려 묶는다 (같은 모양이 되풀이되게)
const GPU_LEN_BUCKET: usize = 64;
/// 사진을 자를 조각 크기, 조각 3×3을 한 토큰으로 묶는 크기, 토큰 최대 개수 (processor 설정과 같음)
const PATCH: usize = 16;
const POOL: usize = 3;
const MAX_SOFT_TOKENS: usize = 280;
/// 모델에 넣기 전에 사진을 이 크기 안으로 줄인다 (파이썬 `embed_server`와 같음)
const QUERY_IMAGE_MAX: u32 = 896;
const BOS: i64 = 2;
const EOS: i64 = 1;
const BOI: i64 = 255_999; // <|image>
const IMAGE_TOKEN: i64 = 258_880; // <|image|>
const EOI: i64 = 258_882; // <image|>

struct Onnx {
    env: ort::environment::Environment,
    text: ort::session::Session,
    tokenizer: tokenizers::Tokenizer,
    /// 시각 인코더: 파일이 있으면 첫 사진 검색 때 올린다 (약 700MB라 글자만 쓸 때는 안 올림)
    vision: Option<ort::session::Session>,
    vision_path: Option<PathBuf>,
    /// DirectML판 onnxruntime을 불러와 GPU를 쓸 수 있는지
    dml: bool,
    text_gpu: bool,
    vision_gpu: bool,
    dim: usize,
}

/// 어떤 모델을 GPU(DirectML)로 돌릴지
#[derive(Clone, Copy)]
pub struct GpuUse {
    pub text: bool,
    pub vision: bool,
}

/// GPU(DirectML)는 LOCALL_ONNX_DEVICE=gpu 일 때만 쓴다. 이 모델에서는 DirectML이 느리고
/// (GPU 사용률 10~15%, 글자 임베딩이 CPU와 비슷) 시각 인코더는 Reshape 오류로 돌지 않는다.
fn gpu_allowed() -> bool {
    std::env::var("LOCALL_ONNX_DEVICE").map_or(false, |v| v.eq_ignore_ascii_case("gpu"))
}

/// DirectML판 onnxruntime과 DirectML.dll을 두는 곳 (`models/onnx/dml/`)
fn dml_dir() -> PathBuf {
    onnx_dir().join("dml")
}

/// 이 DLL을 전체 경로로 먼저 올려 둔다. onnxruntime.dll이 DirectML.dll을 찾을 때 Windows의
/// System32에 있는 옛 판이 아니라 이것을 쓰게 하려고.
#[cfg(windows)]
fn preload_dll(path: &Path) -> bool {
    use std::os::windows::ffi::OsStrExt;
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    unsafe { windows::Win32::System::LibraryLoader::LoadLibraryW(windows::core::PCWSTR(wide.as_ptr())).is_ok() }
}

#[cfg(not(windows))]
fn preload_dll(_path: &Path) -> bool {
    false
}

/// ONNX 모델 폴더 (`models/onnx/`)
fn onnx_dir() -> PathBuf {
    base_dir().join("models").join("onnx")
}

/// onnxruntime.dll 위치와 DirectML판인지. GPU를 쓸 수 있으면 models/onnx/dml의 DirectML판
/// (CPU로도 돌릴 수 있다), 아니면 실행 파일 옆 → models/onnx → 포터블 파이썬의 onnxruntime 순.
fn find_onnxruntime() -> Option<(PathBuf, bool)> {
    let dml = dml_dir();
    if gpu_allowed() && dml.join("onnxruntime.dll").is_file() && preload_dll(&dml.join("DirectML.dll")) {
        return Some((dml.join("onnxruntime.dll"), true));
    }
    find_cpu_onnxruntime().map(|p| (p, false))
}

fn find_cpu_onnxruntime() -> Option<PathBuf> {
    const DLL: &str = if cfg!(windows) { "onnxruntime.dll" } else { "libonnxruntime.so" };
    let base = base_dir();
    let mut cands = Vec::new();
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        cands.push(dir.join(DLL));
    }
    cands.push(base.join(DLL));
    cands.push(onnx_dir().join(DLL));
    cands.push(base.join("python").join("Lib").join("site-packages").join("onnxruntime").join("capi").join(DLL));
    cands.push(
        base.join(".venv").join("Lib").join("site-packages").join("onnxruntime").join("capi").join(DLL),
    );
    cands.into_iter().find(|p| p.is_file())
}

/// `이름.onnx`와 그 가중치 `이름.onnx_data`가 둘 다 있으면 `이름.onnx` 경로
fn onnx_file(name: &str) -> Option<PathBuf> {
    let dir = onnx_dir().join("onnx");
    let model = dir.join(format!("{}.onnx", name));
    (model.is_file() && dir.join(format!("{}.onnx_data", name)).is_file()).then_some(model)
}

/// 세션 만들기. gpu면 DirectML로 해 보고, 안 되면 CPU로. (세션, 실제로 GPU인지)
fn make_session(env: &ort::environment::Environment, model: &Path, gpu: bool) -> Result<(ort::session::Session, bool)> {
    if gpu {
        match make_gpu_session(env, model) {
            Ok(s) => return Ok((s, true)),
            Err(e) => eprintln!("[모델] GPU(DirectML)로 열지 못해 CPU로 엽니다: {:#}", e),
        }
    }
    make_cpu_session(env, model).map(|s| (s, false))
}

fn make_gpu_session(env: &ort::environment::Environment, model: &Path) -> Result<ort::session::Session> {
    use ort::ep::directml::{DirectML, PerformancePreference};
    let fail = |e: &dyn std::fmt::Display| anyhow!("{}", e);
    // DirectML은 메모리 패턴 최적화와 병렬 실행을 지원하지 않는다 (ONNX Runtime 문서)
    let mut builder = ort::session::Session::builder(env)
        .map_err(|e| fail(&e))?;
    if std::env::var("LOCALL_DML_NOFUSION").is_ok() {
        builder = builder.with_config_entry("ep.dml.disable_graph_fusion", "1").map_err(|e| fail(&e))?;
    }
    if let Ok(level) = std::env::var("LOCALL_DML_OPT") {
        use ort::session::builder::GraphOptimizationLevel as L;
        let l = match level.as_str() { "0" => L::Disable, "1" => L::Level1, "2" => L::Level2, _ => L::Level3 };
        builder = builder.with_optimization_level(l).map_err(|e| fail(&e))?;
    }
    let mut builder = builder
        .with_memory_pattern(false)
        .map_err(|e| fail(&e))?
        .with_parallel_execution(false)
        .map_err(|e| fail(&e))?
        .with_execution_providers([DirectML::default()
            // 노트북의 내장 그래픽이 아니라 성능이 높은 쪽(외장 GPU)을 고른다
            .with_performance_preference(PerformancePreference::HighPerformance)
            .build()
            .error_on_failure()])
        .map_err(|e| fail(&e))?;
    builder.commit_from_file(model).map_err(|e| fail(&e))
}

fn make_cpu_session(env: &ort::environment::Environment, model: &Path) -> Result<ort::session::Session> {
    let fail = |e: &dyn std::fmt::Display| anyhow!("ONNX 모델 적재 실패 ({}): {}", model.display(), e);
    // 질의는 띄엄띄엄 오므로, 추론 뒤 스레드가 헛돌며(spinning) CPU를 붙잡아
    // 바로 이어지는 내적 검색(rayon)을 느리게 만들지 않게 한다.
    let mut builder = ort::session::Session::builder(env)
        .map_err(|e| fail(&e))?
        .with_intra_op_spinning(false)
        .map_err(|e| fail(&e))?
        .with_inter_op_spinning(false)
        .map_err(|e| fail(&e))?;
    builder.commit_from_file(model).map_err(|e| fail(&e))
}

/// 파이썬 쪽 `_sanitize`와 같이, 모델이 특수 토큰으로 읽는 표기를 평범한 글자로 바꾼다.
fn sanitize(text: &str) -> String {
    const SPECIAL: [(&str, &str); 7] = [
        ("<|image|>", "<image>"),
        ("<|video|>", "<video>"),
        ("<|audio|>", "<audio>"),
        ("<|image>", "<image>"),
        ("<image|>", "<image>"),
        ("<|audio>", "<audio>"),
        ("<audio|>", "<audio>"),
    ];
    let mut s = text.to_string();
    for (tok, repl) in SPECIAL {
        if s.contains(tok) {
            s = s.replace(tok, repl);
        }
    }
    s
}

/// 앞 dim차원만 잘라 다시 정규화
fn normalize_head(emb: &[f32], dim: usize) -> Result<Vec<f32>> {
    if emb.len() < dim {
        bail!("벡터 길이가 맞지 않습니다: {}", emb.len());
    }
    let mut v = emb[..dim].to_vec();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter_mut().for_each(|x| *x /= norm);
    Ok(v)
}

// ----- 사진 전처리 (PIL + Gemma4ImageProcessor를 그대로 따라 한다) -----

/// 사진 파일 → EXIF 방향대로 돌린 RGB, 긴 변 896 이하 (파이썬 `_load_query_image`)
fn load_query_image(path: &Path) -> Result<image::RgbImage> {
    use image::{DynamicImage, ImageDecoder, ImageReader};
    let mut decoder = ImageReader::open(path)
        .with_context(|| format!("사진을 열 수 없습니다: {}", path.display()))?
        .with_guessed_format()?
        .into_decoder()
        .with_context(|| format!("사진 형식을 읽을 수 없습니다: {}", path.display()))?;
    let orientation = decoder.orientation().ok();
    let mut img = DynamicImage::from_decoder(decoder).context("사진 해독 실패")?;
    if let Some(o) = orientation {
        img.apply_orientation(o);
    }
    Ok(pil_thumbnail(img.to_rgb8(), QUERY_IMAGE_MAX))
}

/// PIL `Image.thumbnail((max, max))`의 목표 크기 (비율을 지키는 PIL의 round_aspect). 이미 작으면 None.
fn pil_thumbnail_size(w: u32, h: u32, max: u32) -> Option<(u32, u32)> {
    if w <= max && h <= max {
        return None;
    }
    let aspect = w as f64 / h as f64;
    let (mut x, mut y) = (max as f64, max as f64);
    // floor와 ceil 중 비율이 더 가까운 쪽
    let pick = |n: f64, err: &dyn Fn(f64) -> f64| -> f64 {
        let (f, c) = (n.floor(), n.ceil());
        (if err(c) < err(f) { c } else { f }).max(1.0)
    };
    if x / y >= aspect {
        x = pick(y * aspect, &|n| (aspect - n / y).abs());
    } else {
        y = pick(x / aspect, &|n| if n == 0.0 { 0.0 } else { (aspect - x / n).abs() });
    }
    Some((x as u32, y as u32))
}

/// PIL `Image.thumbnail((max, max))`: 비율을 지켜 줄이되, 먼저 정수배 평균으로 줄이고(reducing_gap=2)
/// 나머지를 bicubic으로 줄인다.
fn pil_thumbnail<P>(img: image::ImageBuffer<P, Vec<u8>>, max: u32) -> image::ImageBuffer<P, Vec<u8>>
where
    P: image::Pixel<Subpixel = u8> + 'static,
{
    let (w, h) = img.dimensions();
    let Some((tx, ty)) = pil_thumbnail_size(w, h, max) else {
        return img;
    };
    let fx = ((w as f64 / tx as f64 / 2.0) as u32).max(1);
    let fy = ((h as f64 / ty as f64 / 2.0) as u32).max(1);
    let reduced = if fx > 1 || fy > 1 { box_reduce(&img, fx, fy) } else { img };
    image::imageops::resize(&reduced, tx, ty, image::imageops::FilterType::CatmullRom)
}

/// PIL `Image.reduce`: fx×fy 칸의 평균 (가장자리의 덜 찬 칸은 있는 픽셀만으로)
fn box_reduce<P>(img: &image::ImageBuffer<P, Vec<u8>>, fx: u32, fy: u32) -> image::ImageBuffer<P, Vec<u8>>
where
    P: image::Pixel<Subpixel = u8> + 'static,
{
    let ch = P::CHANNEL_COUNT as usize;
    let (w, h) = img.dimensions();
    let (ow, oh) = (w.div_ceil(fx), h.div_ceil(fy));
    let src = img.as_raw();
    let mut out = vec![0u8; (ow * oh) as usize * ch];
    for oy in 0..oh {
        for ox in 0..ow {
            let mut sum = [0u32; 4];
            let mut n = 0u32;
            for y in oy * fy..((oy + 1) * fy).min(h) {
                let row = (y * w) as usize * ch;
                for x in ox * fx..((ox + 1) * fx).min(w) {
                    let i = row + x as usize * ch;
                    for c in 0..ch {
                        sum[c] += src[i + c] as u32;
                    }
                    n += 1;
                }
            }
            let o = ((oy * ow + ox) as usize) * ch;
            for c in 0..ch {
                out[o + c] = ((sum[c] + n / 2) / n) as u8;
            }
        }
    }
    image::ImageBuffer::from_raw(ow, oh, out).expect("크기가 맞는 버퍼")
}

/// 색인용 사진 불러오기 (파이썬 `extract.load_image(path, 896)`):
/// JPEG는 PIL draft처럼 1/2·1/4·1/8로 줄여 읽고, 짧은 변이 96 미만이면 None(아이콘 등),
/// EXIF 방향대로 돌린 뒤 896 안으로 줄여 RGB로. 투명한 그림은 PIL처럼 알파를 곱한 채로 줄인다.
pub fn load_index_image(path: &Path) -> Result<Option<image::RgbImage>> {
    use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};
    let reader = ImageReader::open(path)
        .with_context(|| format!("사진을 열 수 없습니다: {}", path.display()))?
        .with_guessed_format()?;
    let is_jpeg = reader.format() == Some(ImageFormat::Jpeg);
    let mut decoder = reader.into_decoder().with_context(|| format!("사진 형식을 읽을 수 없습니다: {}", path.display()))?;
    let orientation = decoder.orientation().ok();
    let img = DynamicImage::from_decoder(decoder).context("사진 해독 실패")?;
    let max = QUERY_IMAGE_MAX;

    // PIL draft: scale = min(w // 896, h // 896) 이하의 가장 큰 8·4·2·1
    let (w, h) = (img.width(), img.height());
    let scale = if is_jpeg { (w / max).min(h / max) } else { 1 };
    let s = [8u32, 4, 2, 1].into_iter().find(|&s| scale >= s).unwrap_or(1);
    let small_side = if s > 1 { w.div_ceil(s).min(h.div_ceil(s)) } else { w.min(h) };
    if small_side < 96 {
        return Ok(None);
    }

    let rgb = if img.color().has_alpha() && pil_thumbnail_size(w, h, max).is_some() {
        // RGBa(알파를 곱한 값)로 줄였다가 되돌리고 RGB로: 완전히 투명한 곳은 검정이 된다
        let mut rgba = img.to_rgba8();
        if let Some(o) = orientation {
            let mut d = DynamicImage::ImageRgba8(rgba);
            d.apply_orientation(o);
            rgba = d.to_rgba8();
        }
        for p in rgba.pixels_mut() {
            let a = p[3] as u32;
            for c in 0..3 {
                p[c] = ((p[c] as u32 * a + 127) / 255) as u8;
            }
        }
        let mut small = pil_thumbnail(rgba, max);
        for p in small.pixels_mut() {
            let a = p[3] as u32;
            for c in 0..3 {
                p[c] = if a == 0 { 0 } else { ((p[c] as u32 * 255 + a / 2) / a).min(255) as u8 };
            }
        }
        DynamicImage::ImageRgba8(small).to_rgb8()
    } else {
        let mut rgb = img.to_rgb8();
        if s > 1 {
            rgb = box_reduce(&rgb, s, s);
        }
        if let Some(o) = orientation {
            let mut d = DynamicImage::ImageRgb8(rgb);
            d.apply_orientation(o);
            rgb = d.to_rgb8();
        }
        pil_thumbnail(rgb, max)
    };
    Ok(Some(rgb))
}

/// 조각 개수 한도 안에서 비율을 지키며 48(=16×3)의 배수로 맞춘 크기 (높이, 너비)
fn target_size(h: usize, w: usize) -> Result<(usize, usize)> {
    let max_patches = MAX_SOFT_TOKENS * POOL * POOL;
    let side = POOL * PATCH;
    let factor = ((max_patches * PATCH * PATCH) as f64 / (h * w) as f64).sqrt();
    let mut th = (factor * h as f64 / side as f64).floor() as usize * side;
    let mut tw = (factor * w as f64 / side as f64).floor() as usize * side;
    let max_side = (max_patches / (POOL * POOL)) * side;
    if th == 0 && tw == 0 {
        bail!("사진이 너무 작습니다");
    }
    if th == 0 {
        th = side;
        tw = ((w / h) * side).min(max_side);
    } else if tw == 0 {
        tw = side;
        th = ((h / w) * side).min(max_side);
    }
    Ok((th, tw))
}

/// RGB 사진 → (조각 픽셀 [n×768], 조각 위치 [n×2] (x, y), 조각 수 n, 이미지 토큰 수)
fn patchify(img: &image::RgbImage) -> Result<(Vec<f32>, Vec<i64>, usize, usize)> {
    let (th, tw) = target_size(img.height() as usize, img.width() as usize)?;
    let resized = image::imageops::resize(img, tw as u32, th as u32, image::imageops::FilterType::CatmullRom);
    let (ph, pw) = (th / PATCH, tw / PATCH);
    let n = ph * pw;
    let mut pixels = Vec::with_capacity(n * PATCH * PATCH * 3);
    let mut pos = Vec::with_capacity(n * 2);
    for py in 0..ph {
        for px in 0..pw {
            for y in 0..PATCH {
                for x in 0..PATCH {
                    let p = resized.get_pixel((px * PATCH + x) as u32, (py * PATCH + y) as u32).0;
                    pixels.extend(p.iter().map(|&c| c as f32 / 255.0));
                }
            }
            pos.push(px as i64);
            pos.push(py as i64);
        }
    }
    Ok((pixels, pos, n, n / (POOL * POOL)))
}

impl Onnx {
    /// 글자 모델 파일이 다 있으면 적재, 하나라도 없으면 Ok(None). 시각 인코더는 나중에.
    fn load(cfg: &Config, gpu: GpuUse) -> Result<Option<Self>> {
        let tok_path = onnx_dir().join("tokenizer.json");
        let (Some(model), true) = (onnx_file("model"), tok_path.is_file()) else {
            return Ok(None);
        };
        let Some((dll, dml)) = find_onnxruntime() else {
            return Ok(None);
        };

        let env = ort::init_from(&dll)
            .map_err(|e| anyhow!("onnxruntime을 불러올 수 없습니다 ({}): {}", dll.display(), e))?
            .with_name("locall")
            .build()
            .map_err(|e| anyhow!("onnxruntime 초기화 실패: {}", e))?;
        let (text, text_gpu) = make_session(&env, &model, dml && gpu.text)?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow!("토크나이저 적재 실패: {}", e))?;

        let vision_path = onnx_file("vision_encoder");
        let mut me = Self {
            env,
            text,
            tokenizer,
            vision: None,
            vision_path,
            dml,
            text_gpu,
            vision_gpu: dml && gpu.vision,
            dim: cfg.dim,
        };
        let v = me.query("예열")?; // 첫 추론은 느리므로 미리 한 번
        if v.len() != cfg.dim {
            bail!("모델 차원({})이 설정({})과 다릅니다", v.len(), cfg.dim);
        }
        Ok(Some(me))
    }

    fn has_vision(&self) -> bool {
        self.vision_path.is_some()
    }

    /// 화면·기록용 장치 설명 (예: "글자 cpu · 사진 gpu")
    fn device(&self) -> String {
        let d = |gpu: bool| if gpu { "gpu" } else { "cpu" };
        let mut s = format!("onnx 글자 {}", d(self.text_gpu));
        if self.has_vision() {
            s.push_str(&format!(" · 사진 {}", d(self.vision_gpu)));
        }
        if self.dml {
            s.push_str(" (DirectML)");
        }
        s
    }

    /// 글자 모델 한 번 돌리기 → 앞 dim차원 정규화 벡터
    fn run_text(&mut self, ids: Vec<i64>, image_features: Option<(usize, Vec<f32>)>) -> Result<Vec<f32>> {
        use ort::value::Tensor;
        let n = ids.len();
        let mask = vec![1i64; n];
        let empty = || Tensor::from_array(([0usize, 512], Vec::<f32>::new()));
        let image = match image_features {
            Some((tokens, feats)) => Tensor::from_array(([tokens, feats.len() / tokens.max(1)], feats))?,
            None => empty()?,
        };
        let outputs = self.text.run(ort::inputs![
            "input_ids" => Tensor::from_array(([1usize, n], ids))?,
            "attention_mask" => Tensor::from_array(([1usize, n], mask))?,
            "image_features" => image,
            "video_features" => empty()?,
            "audio_features" => empty()?,
        ])?;
        // sentence_embedding = attention mask 평균 풀링 + 정규화 (768차원). 앞 dim개만 쓰고 다시 정규화.
        let (_, emb) = outputs["sentence_embedding"].try_extract_tensor::<f32>()?;
        normalize_head(emb, self.dim)
    }

    fn query(&mut self, text: &str) -> Result<Vec<f32>> {
        let text = format!("{}{}", QUERY_PREFIX, sanitize(text));
        let enc = self.tokenizer.encode(text, true).map_err(|e| anyhow!("토큰화 실패: {}", e))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
        if ids.len() > MAX_TOKENS {
            let eos = *ids.last().unwrap();
            ids.truncate(MAX_TOKENS - 1);
            ids.push(eos);
        }
        self.run_text(ids, None)
    }

    /// 문서 문장 여러 개 → 벡터들 (순서 그대로). 길이가 비슷한 것끼리 묶어 패딩을 줄인다.
    fn docs(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        use ort::value::Tensor;
        let mut ids: Vec<Vec<i64>> = Vec::with_capacity(texts.len());
        for t in texts {
            let enc = self.tokenizer.encode(sanitize(t), true).map_err(|e| anyhow!("토큰화 실패: {}", e))?;
            let mut v: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
            if v.len() > MAX_DOC_TOKENS {
                let eos = *v.last().unwrap();
                v.truncate(MAX_DOC_TOKENS - 1);
                v.push(eos);
            }
            ids.push(v);
        }
        let gpu = self.text_gpu;
        let bucket = |n: usize| if gpu { n.div_ceil(GPU_LEN_BUCKET) * GPU_LEN_BUCKET } else { n };
        let mut order: Vec<usize> = (0..ids.len()).collect();
        order.sort_by_key(|&i| ids[i].len());

        let mut out: Vec<Vec<f32>> = vec![Vec::new(); ids.len()];
        let mut start = 0;
        while start < order.len() {
            // 짧은 것부터: 묶음 크기 × 가장 긴 길이가 토큰 예산을 넘지 않게
            let mut end = start + 1;
            while end < order.len()
                && end - start < MAX_DOC_BATCH
                && (end - start + 1) * bucket(ids[order[end]].len()) <= DOC_TOKEN_BUDGET
            {
                end += 1;
            }
            let batch = &order[start..end];
            let mut len = ids[batch[batch.len() - 1]].len();
            if self.text_gpu {
                len = len.div_ceil(GPU_LEN_BUCKET) * GPU_LEN_BUCKET;
            }
            let b = batch.len();
            let mut flat = Vec::with_capacity(b * len);
            let mut mask = Vec::with_capacity(b * len);
            for &i in batch {
                let v = &ids[i];
                flat.extend_from_slice(v);
                flat.extend(std::iter::repeat(PAD).take(len - v.len()));
                mask.extend(std::iter::repeat(1i64).take(v.len()));
                mask.extend(std::iter::repeat(0i64).take(len - v.len()));
            }
            let empty = || Tensor::from_array(([0usize, 512], Vec::<f32>::new()));
            let outputs = self.text.run(ort::inputs![
                "input_ids" => Tensor::from_array(([b, len], flat))?,
                "attention_mask" => Tensor::from_array(([b, len], mask))?,
                "image_features" => empty()?,
                "video_features" => empty()?,
                "audio_features" => empty()?,
            ])?;
            let (shape, emb) = outputs["sentence_embedding"].try_extract_tensor::<f32>()?;
            let width = *shape.last().unwrap_or(&0) as usize;
            for (k, &i) in batch.iter().enumerate() {
                out[i] = normalize_head(&emb[k * width..(k + 1) * width], self.dim)?;
            }
            start = end;
        }
        Ok(out)
    }

    /// 사진 → 시각 인코더(조각 → 이미지 토큰 특징) → 글자 모델(<bos><|image> 토큰들 <image|><eos>)
    fn image(&mut self, path: &Path) -> Result<Vec<f32>> {
        let img = load_query_image(path)?;
        self.image_rgb(&img)
    }

    /// 이미 불러온 RGB 그림 → 벡터 (896 축소 없이 바로 모델 입력 크기로; 색인기의 스캔 PDF용)
    fn image_rgb(&mut self, img: &image::RgbImage) -> Result<Vec<f32>> {
        use ort::value::Tensor;
        if self.vision.is_none() {
            let p = self.vision_path.clone().ok_or_else(|| anyhow!("시각 인코더 파일이 없습니다"))?;
            let (session, on_gpu) = make_session(&self.env, &p, self.vision_gpu)?;
            self.vision_gpu = on_gpu;
            self.vision = Some(session);
        }
        let (pixels, pos, n, tokens) = patchify(img)?;
        let feats = {
            let vision = self.vision.as_mut().unwrap();
            let out = vision.run(ort::inputs![
                "pixel_values" => Tensor::from_array(([1usize, n, PATCH * PATCH * 3], pixels))?,
                "pixel_position_ids" => Tensor::from_array(([1usize, n, 2], pos))?,
            ])?;
            let (shape, f) = out["image_features"].try_extract_tensor::<f32>()?;
            if shape.len() != 2 || shape[0] as usize != tokens {
                bail!("시각 인코더 출력 모양이 예상과 다릅니다: {:?} (토큰 {})", shape, tokens);
            }
            f.to_vec()
        };
        let mut ids = Vec::with_capacity(tokens + 4);
        ids.extend([BOS, BOI]);
        ids.extend(std::iter::repeat(IMAGE_TOKEN).take(tokens));
        ids.extend([EOI, EOS]);
        self.run_text(ids, Some((tokens, feats)))
    }
}

// ---------------------------------------------------------------------------
// 공개 API
// ---------------------------------------------------------------------------

/// 사진 임베딩용 파이썬 작업자. 다른 스레드에서 띄운 뒤 `Embedder::set_image_worker`로 넘긴다.
pub struct ImageWorker(Worker);

impl ImageWorker {
    /// 작업자를 띄우고 모델이 올라올 때까지 기다린다 (수 초~수십 초).
    pub fn spawn(cfg: &Config) -> Result<Self> {
        Worker::spawn(cfg).map(Self)
    }
}

pub struct Embedder {
    onnx: Option<Onnx>,
    worker: Option<Worker>,
    cfg: Config,
    pub dim: usize,
    pub device: String,
}

impl Embedder {
    /// 검색어 임베딩을 준비한다. ONNX 모델이 있으면 바로(1~2초), 없으면 파이썬 작업자를 띄워
    /// 모델이 올라올 때까지 기다린다 (수 초~수십 초).
    pub fn spawn(cfg: &Config) -> Result<Self> {
        // LOCALL_EMBED=python 이면 ONNX를 건너뛴다 (비교·문제 확인용)
        let force_python = std::env::var("LOCALL_EMBED").map_or(false, |v| v.eq_ignore_ascii_case("python"));
        // 검색 앱: 검색어는 CPU로도 20ms라 GPU 메모리를 아끼고, 무거운 사진만 GPU로
        let gpu = GpuUse { text: false, vision: true };
        let onnx = match if force_python { Ok(None) } else { Onnx::load(cfg, gpu) } {
            Ok(o) => o,
            Err(e) => {
                // ONNX가 깨져 있어도 앱은 돌아가야 하므로 파이썬 작업자로 넘어간다
                eprintln!("[모델] ONNX 사용 불가, 파이썬 작업자로 대신합니다: {:#}", e);
                None
            }
        };
        if onnx.is_some() {
            let device = onnx.as_ref().map(|o| o.device()).unwrap_or_default();
            return Ok(Self { onnx, worker: None, cfg: cfg.clone(), dim: cfg.dim, device });
        }
        let w = Worker::spawn(cfg)?;
        Ok(Self { dim: w.dim, device: w.device.clone(), onnx: None, worker: Some(w), cfg: cfg.clone() })
    }

    /// 파이썬 작업자 (없거나 죽었으면 새로 띄운다)
    fn worker(&mut self) -> Result<&mut Worker> {
        if self.worker.as_mut().map_or(false, |w| !w.is_alive()) {
            self.worker = None;
        }
        if self.worker.is_none() {
            self.worker = Some(Worker::spawn(&self.cfg)?);
        }
        Ok(self.worker.as_mut().unwrap())
    }

    /// 검색어를 임베딩할 수 있는 상태인지 (ONNX면 늘 참, 작업자면 프로세스가 살아 있는지)
    pub fn is_alive(&mut self) -> bool {
        if self.onnx.is_some() {
            return true;
        }
        self.worker.as_mut().map_or(false, |w| w.is_alive())
    }

    /// 검색어 → 정규화된 dim 차원 벡터
    pub fn query(&mut self, text: &str) -> Result<Vec<f32>> {
        if let Some(o) = self.onnx.as_mut() {
            return o.query(text);
        }
        match self.worker.as_mut() {
            Some(w) => w.call(serde_json::json!({ "op": "query", "text": text })),
            None => bail!("임베딩 작업자가 종료되었습니다 (data/embed.log 참고)"),
        }
    }

    /// 색인기용: ONNX 모델로만 연다 (파이썬 작업자로 넘어가지 않는다)
    pub fn spawn_onnx(cfg: &Config, gpu: GpuUse) -> Result<Self> {
        let onnx = Onnx::load(cfg, gpu)?.ok_or_else(|| {
            anyhow!("models/onnx의 ONNX 모델 또는 onnxruntime.dll을 찾지 못했습니다")
        })?;
        let device = onnx.device();
        Ok(Self { onnx: Some(onnx), worker: None, cfg: cfg.clone(), dim: cfg.dim, device })
    }

    /// 문서 문장들(`title: … | text: …`) → 정규화된 dim 차원 벡터들. ONNX가 있을 때만.
    pub fn docs(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match self.onnx.as_mut() {
            Some(o) => o.docs(texts),
            None => bail!("문서 임베딩은 ONNX 모델이 있어야 합니다"),
        }
    }

    /// 시각 인코더 ONNX가 있어 그림을 임베딩할 수 있는지
    pub fn has_vision(&self) -> bool {
        self.onnx.as_ref().map_or(false, |o| o.has_vision())
    }

    /// 불러온 RGB 그림 → 벡터 (시각 인코더 ONNX가 있을 때만)
    pub fn image_rgb(&mut self, img: &image::RgbImage) -> Result<Vec<f32>> {
        match self.onnx.as_mut().filter(|o| o.has_vision()) {
            Some(o) => o.image_rgb(img),
            None => bail!("그림 임베딩은 시각 인코더 ONNX가 있어야 합니다"),
        }
    }

    /// 사진 검색 전에 작업자를 따로 띄워야 하는지 (ONNX 사용 중인데 작업자가 없거나 죽었을 때).
    /// 앱은 이때 `ImageWorker::spawn`을 다른 스레드에서 돌려, 검색 스레드가 멈추지 않게 한다.
    pub fn needs_image_worker(&mut self) -> bool {
        // 시각 인코더 ONNX가 있으면 작업자가 필요 없다
        self.onnx.as_ref().map_or(false, |o| !o.has_vision())
            && !self.worker.as_mut().map_or(false, |w| w.is_alive())
    }

    /// 다른 스레드에서 띄운 사진 작업자를 넘겨받는다
    pub fn set_image_worker(&mut self, w: ImageWorker) {
        self.worker = Some(w.0);
    }

    /// 사진 파일 → 정규화된 dim 차원 벡터. ONNX 사용 중이고 작업자가 아직 없으면 여기서 띄우고
    /// 기다린다 (수 초~수십 초). 기다리면 안 되는 곳에서는 `needs_image_worker`를 먼저 볼 것.
    pub fn image(&mut self, path: &Path) -> Result<Vec<f32>> {
        if let Some(o) = self.onnx.as_mut().filter(|o| o.has_vision()) {
            return o.image(path);
        }
        let req = serde_json::json!({ "op": "image", "path": path.to_string_lossy() });
        if self.onnx.is_none() {
            return match self.worker.as_mut() {
                Some(w) => w.call(req),
                None => bail!("임베딩 작업자가 종료되었습니다 (data/embed.log 참고)"),
            };
        }
        self.worker()?.call(req)
    }
}
