//! 검색 결과에 보여 줄 사진 미리보기 (가운데를 정사각형으로 잘라 줄인 PNG).
//!
//! 만든 것은 경로·수정 시각별로 메모리에 담아 두어, 같은 결과를 다시 볼 때는 바로 돌려준다.

use crate::config::is_image_content_ext;
use image::{DynamicImage, ImageDecoder, ImageReader};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

/// 미리보기 한 변 (화면에는 38px로 보이므로 고해상도 화면까지 넉넉하게)
pub const THUMB_PX: u32 = 96;
/// 메모리에 담아 둘 최대 개수 (하나에 수십 KB)
const CACHE_MAX: usize = 600;

type Key = (String, Option<SystemTime>);

fn cache() -> &'static Mutex<HashMap<Key, Option<Vec<u8>>>> {
    static CACHE: OnceLock<Mutex<HashMap<Key, Option<Vec<u8>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 사진 파일 → 미리보기 PNG. 사진이 아니거나 읽지 못하면 None.
pub fn thumbnail_png(path: &Path) -> Option<Vec<u8>> {
    let ext = path.extension()?.to_str()?.to_lowercase();
    if !is_image_content_ext(&ext) {
        return None;
    }
    let meta = std::fs::metadata(path).ok().filter(|m| m.is_file())?;
    let key = (path.to_string_lossy().into_owned(), meta.modified().ok());
    if let Some(hit) = cache().lock().ok()?.get(&key) {
        return hit.clone();
    }

    let made = make(path);
    if let Ok(mut c) = cache().lock() {
        if c.len() >= CACHE_MAX {
            c.clear();
        }
        c.insert(key, made.clone());
    }
    made
}

fn make(path: &Path) -> Option<Vec<u8>> {
    let mut decoder = ImageReader::open(path).ok()?.with_guessed_format().ok()?.into_decoder().ok()?;
    // 휴대폰 사진은 EXIF 방향대로 돌려야 똑바로 보인다
    let orientation = decoder.orientation().ok();
    let mut img = DynamicImage::from_decoder(decoder).ok()?;
    if let Some(o) = orientation {
        img.apply_orientation(o);
    }
    // 가운데 정사각형만 잘라 줄인다 (thumbnail은 빠른 축소)
    let (w, h) = (img.width(), img.height());
    let side = w.min(h);
    if side == 0 {
        return None;
    }
    let square = img.crop_imm((w - side) / 2, (h - side) / 2, side, side);
    let small = square.thumbnail(THUMB_PX, THUMB_PX);
    let mut out = std::io::Cursor::new(Vec::new());
    small.to_rgba8().write_to(&mut out, image::ImageFormat::Png).ok()?;
    Some(out.into_inner())
}
