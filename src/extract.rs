//! 파일에서 본문 글자를 꺼낸다. 파이썬 `locall/extract.py`와 같은 글자가 나오도록 맞춰 두었다
//! (같아야 기존 색인의 벡터·키워드 검색 내용과 어긋나지 않는다).
//! PDF는 PDFium(models/pdfium/pdfium.dll)으로 읽는다. 파이썬은 PyMuPDF라 글자가 조금 다를 수 있다.

use anyhow::{bail, Result};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

pub const MAX_CHARS: usize = 400_000;

pub static SPECIAL_TOKENS: &[(&str, &str)] = &[
    ("<|image|>", "<image>"),
    ("<|video|>", "<video>"),
    ("<|audio|>", "<audio>"),
    ("<|image>", "<image>"),
    ("<image|>", "<image>"),
    ("<|audio>", "<audio>"),
    ("<audio|>", "<audio>"),
];

/// 파이썬 `str.isspace()` / 정규식 `\s`와 같은 공백 판정
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// 파이썬 `_clean`: \0→공백, [ \t\r\f\v]+→공백, `\n\s*\n+`→\n, 앞뒤 공백 제거, 특수 토큰 치환
pub fn clean(text: &str) -> String {
    let mut s1 = String::with_capacity(text.len());
    let mut in_run = false;
    for c in text.chars() {
        let c = if c == '\0' { ' ' } else { c };
        if matches!(c, ' ' | '\t' | '\r' | '\x0b' | '\x0c') {
            if !in_run {
                s1.push(' ');
                in_run = true;
            }
        } else {
            s1.push(c);
            in_run = false;
        }
    }

    // `\n\s*\n+` → `\n`: 줄바꿈부터 이어진 공백 덩어리 안의 마지막 줄바꿈까지를 줄바꿈 하나로
    let chars: Vec<char> = s1.chars().collect();
    let mut s2 = String::with_capacity(s1.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\n' {
            let mut j = i + 1;
            let mut last_nl = None;
            while j < chars.len() && is_py_space(chars[j]) {
                if chars[j] == '\n' {
                    last_nl = Some(j);
                }
                j += 1;
            }
            if let Some(l) = last_nl {
                s2.push('\n');
                i = l + 1;
                continue;
            }
        }
        s2.push(chars[i]);
        i += 1;
    }

    let mut s3 = s2.trim_matches(is_py_space).to_string();
    for (token, repl) in SPECIAL_TOKENS {
        if s3.contains(token) {
            s3 = s3.replace(token, repl);
        }
    }
    s3
}

/// UTF-16 코드 단위 → 글자 (짝 없는 대리 쌍은 버림, 파이썬 errors="ignore"와 같음)
fn decode_utf16_ignore(units: impl IntoIterator<Item = u16>) -> String {
    char::decode_utf16(units).filter_map(|r| r.ok()).collect()
}

pub fn read_plain(path: &Path) -> Result<String> {
    let mut buf = Vec::new();
    File::open(path)?.take(MAX_CHARS as u64).read_to_end(&mut buf)?;

    if buf.starts_with(&[0xff, 0xfe]) {
        return Ok(decode_utf16_ignore(buf[2..].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]))));
    }
    if buf.starts_with(&[0xfe, 0xff]) {
        return Ok(decode_utf16_ignore(buf[2..].chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]]))));
    }
    if let Ok(s) = std::str::from_utf8(&buf) {
        return Ok(s.strip_prefix('\u{feff}').unwrap_or(s).to_string());
    }
    // CP949 (errors="ignore"이므로 못 읽은 바이트는 버린다)
    let (cow, _, _) = encoding_rs::EUC_KR.decode(&buf);
    Ok(cow.chars().filter(|&c| c != '\u{fffd}').collect())
}

/// zip 열기. 열지 못하면 NotSupported (파이썬 zipfile은 끝이 조금 망가진 파일도 여는 경우가 있다)
fn open_zip(path: &Path) -> Result<zip::ZipArchive<BufReader<File>>> {
    let f = File::open(path)?;
    zip::ZipArchive::new(BufReader::new(f)).map_err(|e| NotSupported(format!("zip을 열지 못함: {}", e)).into())
}

/// zip 안의 XML 하나를 글자로 (깨진 UTF-8은 무시)
fn zip_text<R: Read + std::io::Seek>(archive: &mut zip::ZipArchive<R>, name: &str) -> Option<String> {
    let mut f = archive.by_name(name).ok()?;
    let mut raw = Vec::new();
    f.read_to_end(&mut raw).ok()?;
    Some(String::from_utf8_lossy(&raw).chars().filter(|&c| c != '\u{fffd}').collect())
}

/// `<w:p>`/`<a:p>` 한 문단의 글자 (python-docx / python-pptx의 paragraph.text와 비슷하게).
/// 줄바꿈 태그는 비어 있든(`<a:br/>`) 안에 속성 태그가 있든(`<a:br>…</a:br>`) 같게 친다.
fn ooxml_paragraph(
    reader: &mut quick_xml::Reader<&[u8]>,
    text_tag: &[u8],
    para_tag: &[u8],
    tab: &[u8],
    breaks: &[&[u8]],
    br_char: char,
) -> String {
    use quick_xml::events::Event;
    let mut out = String::new();
    let mut in_text = false;
    // 문단 속성(w:pPr / a:pPr) 안의 탭은 탭 위치 정의라 글자가 아니다
    let ppr: &[u8] = if para_tag == b"w:p" { b"w:pPr" } else { b"a:pPr" };
    let mut in_ppr = 0usize;
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) if e.name().as_ref() == ppr => in_ppr += 1,
            Ok(Event::End(e)) if e.name().as_ref() == ppr => in_ppr = in_ppr.saturating_sub(1),
            Ok(Event::Start(_)) | Ok(Event::Empty(_)) if in_ppr > 0 => {}
            Ok(Event::Start(e)) => {
                let n = e.name();
                if n.as_ref() == text_tag {
                    in_text = true;
                } else if n.as_ref() == tab {
                    out.push('\t');
                } else if breaks.contains(&n.as_ref()) {
                    out.push(br_char);
                }
            }
            Ok(Event::End(e)) if e.name().as_ref() == text_tag => in_text = false,
            Ok(Event::End(e)) if e.name().as_ref() == para_tag => break,
            Ok(Event::Text(t)) if in_text => out.push_str(&t.unescape().unwrap_or_default()),
            Ok(Event::Empty(e)) => {
                let n = e.name();
                if n.as_ref() == tab {
                    out.push('\t');
                } else if breaks.contains(&n.as_ref()) {
                    out.push(br_char);
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    out
}

/// 태그의 속성값 하나 (예: w:val)
fn xml_attr(e: &quick_xml::events::BytesStart, key: &[u8]) -> Option<String> {
    e.attributes()
        .flatten()
        .find(|a| a.key.as_ref() == key)
        .map(|a| String::from_utf8_lossy(&a.value).into_owned())
}

/// python-docx: 본문 문단들(줄바꿈으로) + 표의 각 행(" | "로 칸을 이어서).
/// python-docx의 row.cells처럼 가로로 합친 칸(gridSpan)은 칸 수만큼, 세로로 합친 칸(vMerge)은
/// 위 칸의 글자로 되풀이한다.
pub fn read_docx(path: &Path) -> Result<String> {
    use quick_xml::events::Event;
    let mut archive = open_zip(path)?;
    let xml = zip_text(&mut archive, "word/document.xml").unwrap_or_default();
    let mut reader = quick_xml::Reader::from_str(&xml);
    let mut paras = Vec::new();
    let mut rows = Vec::new();
    let mut tbl_depth = 0usize;
    let mut row_cells: Vec<String> = Vec::new(); // 격자 칸 단위
    let mut cell_paras: Vec<String> = Vec::new();
    let mut span = 1usize;
    let mut vmerge: Option<bool> = None; // Some(true)=새로 시작, Some(false)=위 칸에 이어짐
    let mut above: Vec<String> = Vec::new(); // 격자 열마다 세로 합치기의 시작 칸 글자
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => match e.name().as_ref() {
                b"w:tbl" => {
                    tbl_depth += 1;
                    if tbl_depth == 1 {
                        above.clear();
                    }
                }
                b"w:tr" if tbl_depth == 1 => row_cells.clear(),
                b"w:tc" if tbl_depth == 1 => {
                    cell_paras.clear();
                    span = 1;
                    vmerge = None;
                }
                b"w:p" => {
                    let p = ooxml_paragraph(&mut reader, b"w:t", b"w:p", b"w:tab", &[b"w:br", b"w:cr"], '\n');
                    if tbl_depth == 0 {
                        paras.push(p);
                    } else if tbl_depth == 1 {
                        cell_paras.push(p);
                    }
                }
                _ => {}
            },
            Ok(Event::Empty(e)) => match e.name().as_ref() {
                b"w:p" if tbl_depth == 0 => paras.push(String::new()),
                b"w:p" if tbl_depth == 1 => cell_paras.push(String::new()),
                b"w:gridSpan" if tbl_depth == 1 => {
                    span = xml_attr(&e, b"w:val").and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
                }
                b"w:vMerge" if tbl_depth == 1 => {
                    vmerge = Some(xml_attr(&e, b"w:val").as_deref() == Some("restart"));
                }
                _ => {}
            },
            Ok(Event::End(e)) => match e.name().as_ref() {
                b"w:tbl" => tbl_depth = tbl_depth.saturating_sub(1),
                b"w:tc" if tbl_depth == 1 => {
                    let col = row_cells.len();
                    let text = if vmerge == Some(false) {
                        above.get(col).cloned().unwrap_or_default()
                    } else {
                        cell_paras.join("\n")
                    };
                    for k in 0..span {
                        if above.len() <= col + k {
                            above.resize(col + k + 1, String::new());
                        }
                        above[col + k] = text.clone();
                        row_cells.push(text.clone());
                    }
                }
                b"w:tr" if tbl_depth == 1 => rows.push(row_cells.join(" | ")),
                _ => {}
            },
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    paras.extend(rows);
    Ok(paras.join("\n"))
}

/// 프레젠테이션에 실린 순서대로의 슬라이드 파일 이름 (presentation.xml의 sldIdLst → rels)
fn pptx_slide_order<R: Read + std::io::Seek>(archive: &mut zip::ZipArchive<R>) -> Vec<String> {
    use quick_xml::events::Event;
    let mut rids = Vec::new();
    if let Some(xml) = zip_text(archive, "ppt/presentation.xml") {
        let mut reader = quick_xml::Reader::from_str(&xml);
        let mut buf = Vec::new();
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Empty(e)) | Ok(Event::Start(e)) if e.name().as_ref() == b"p:sldId" => {
                    if let Some(r) = xml_attr(&e, b"r:id") {
                        rids.push(r);
                    }
                }
                Ok(Event::Eof) | Err(_) => break,
                _ => {}
            }
            buf.clear();
        }
    }
    let mut targets = std::collections::HashMap::new();
    if let Some(xml) = zip_text(archive, "ppt/_rels/presentation.xml.rels") {
        let mut reader = quick_xml::Reader::from_str(&xml);
        let mut buf = Vec::new();
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Empty(e)) | Ok(Event::Start(e)) if e.name().as_ref() == b"Relationship" => {
                    if let (Some(id), Some(t)) = (xml_attr(&e, b"Id"), xml_attr(&e, b"Target")) {
                        let t = t.trim_start_matches('/');
                        let full = if t.starts_with("ppt/") { t.to_string() } else { format!("ppt/{}", t) };
                        targets.insert(id, full);
                    }
                }
                Ok(Event::Eof) | Err(_) => break,
                _ => {}
            }
            buf.clear();
        }
    }
    let ordered: Vec<String> = rids.iter().filter_map(|r| targets.get(r).cloned()).collect();
    if !ordered.is_empty() {
        return ordered;
    }
    // 목록을 못 읽으면 파일 번호 순서로
    let mut slides: Vec<(u32, String)> = archive
        .file_names()
        .filter_map(|n| {
            let num = n.strip_prefix("ppt/slides/slide")?.strip_suffix(".xml")?.parse().ok()?;
            Some((num, n.to_string()))
        })
        .collect();
    slides.sort();
    slides.into_iter().map(|(_, n)| n).collect()
}

/// python-pptx: 슬라이드 순서대로, 맨 위 도형 중 글상자가 있는 것의 글자 (문단은 줄바꿈)
pub fn read_pptx(path: &Path) -> Result<String> {
    use quick_xml::events::Event;
    let mut archive = open_zip(path)?;
    let slides = pptx_slide_order(&mut archive);

    let mut parts = Vec::new();
    for name in slides {
        let Some(xml) = zip_text(&mut archive, &name) else { continue };
        let mut reader = quick_xml::Reader::from_str(&xml);
        // spTree 바로 아래의 p:sp 안 p:txBody만 (묶음 도형 안은 python-pptx도 안 본다)
        let mut stack: Vec<Vec<u8>> = Vec::new();
        let mut frame: Option<Vec<String>> = None;
        let mut buf = Vec::new();
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    let n = e.name().as_ref().to_vec();
                    let top_shape = stack.len() >= 2
                        && stack[stack.len() - 1] == b"p:sp"
                        && stack[stack.len() - 2] == b"p:spTree";
                    if n == b"p:txBody" && top_shape {
                        frame = Some(Vec::new());
                    } else if n == b"a:p" && frame.is_some() {
                        let p = ooxml_paragraph(&mut reader, b"a:t", b"a:p", b"a:tab", &[b"a:br"], '\x0b');
                        frame.as_mut().unwrap().push(p);
                        buf.clear();
                        continue;
                    }
                    stack.push(n);
                }
                Ok(Event::Empty(e)) if e.name().as_ref() == b"a:p" && frame.is_some() => {
                    frame.as_mut().unwrap().push(String::new());
                }
                Ok(Event::End(e)) => {
                    if e.name().as_ref() == b"p:txBody" {
                        if let Some(f) = frame.take() {
                            parts.push(f.join("\n"));
                        }
                    }
                    stack.pop();
                }
                Ok(Event::Eof) | Err(_) => break,
                _ => {}
            }
            buf.clear();
        }
    }
    Ok(parts.join("\n"))
}

/// HWPX: `<hp:t>` 하나마다 한 줄 (안쪽 태그는 버리고 글자만, 공백뿐인 것은 뺀다)
pub fn read_hwpx(path: &Path) -> Result<String> {
    use quick_xml::events::Event;
    let mut archive = open_zip(path)?;
    let mut sections: Vec<String> = archive
        .file_names()
        .filter(|n| {
            n.strip_prefix("Contents/section")
                .and_then(|r| r.strip_suffix(".xml"))
                .map_or(false, |d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
        })
        .map(str::to_string)
        .collect();
    sections.sort();

    let mut parts = Vec::new();
    for name in sections {
        let Some(xml) = zip_text(&mut archive, &name) else { continue };
        let mut reader = quick_xml::Reader::from_str(&xml);
        let mut depth = 0usize; // hp:t 안이면 1 이상
        let mut cur = String::new();
        let mut buf = Vec::new();
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    if e.name().as_ref() == b"hp:t" && depth == 0 {
                        depth = 1;
                        cur.clear();
                    } else if depth > 0 {
                        depth += 1;
                    }
                }
                Ok(Event::End(e)) if depth > 0 => {
                    depth -= 1;
                    if depth == 0 && e.name().as_ref() == b"hp:t" && !cur.trim().is_empty() {
                        parts.push(std::mem::take(&mut cur));
                    }
                }
                Ok(Event::Text(t)) if depth > 0 => cur.push_str(&t.unescape().unwrap_or_default()),
                Ok(Event::CData(t)) if depth > 0 => cur.push_str(&String::from_utf8_lossy(&t)),
                Ok(Event::Eof) | Err(_) => break,
                _ => {}
            }
            buf.clear();
        }
    }
    Ok(parts.join("\n"))
}

/// 파이썬 float의 str()과 같은 모양 (1e-05, 0.1, 12345.6)
fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let a = f.abs();
    if a != 0.0 && (a < 1e-4 || a >= 1e16) {
        let s = format!("{:e}", f); // 1e-5, 1.5e20
        let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
        let (sign, digits) = if let Some(d) = e.strip_prefix('-') { ("-", d) } else { ("+", e) };
        return format!("{}e{}{:0>2}", m, sign, digits);
    }
    let s = format!("{}", f);
    if s.contains('.') { s } else { format!("{}.0", s) }
}

/// 엑셀 일련번호 → "YYYY-MM-DD HH:MM:SS" (시간만이면 "HH:MM:SS"), openpyxl의 str(datetime)과 같게
fn excel_datetime(serial: f64) -> String {
    let total_secs = (serial * 86_400.0).round() as i64;
    let (days, secs) = (total_secs.div_euclid(86_400), total_secs.rem_euclid(86_400));
    let hms = format!("{:02}:{:02}:{:02}", secs / 3600, secs % 3600 / 60, secs % 60);
    if days == 0 {
        return hms;
    }
    // 1899-12-30을 0일로 (엑셀의 1900년 2월 29일 버그 때문에 60일 전은 하루 당긴다)
    let d = if days < 60 { days + 1 } else { days };
    // 1970-01-01 기준 일수 → 그레고리력 날짜 (Howard Hinnant의 civil_from_days)
    let z = d - 25_569 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    format!("{:04}-{:02}-{:02} {}", year, month, day, hms)
}

/// openpyxl: 워크시트 앞 6개, 각 시트 1~200행, 값이 있는 칸을 " | "로 (값은 str()로)
pub fn read_xlsx(path: &Path) -> Result<String> {
    use calamine::{open_workbook_auto, Data, Reader, SheetType};

    let mut workbook = open_workbook_auto(path)?;
    let names: Vec<String> = workbook
        .sheets_metadata()
        .iter()
        .filter(|s| s.typ == SheetType::WorkSheet)
        .take(6)
        .map(|s| s.name.clone())
        .collect();
    let mut parts = Vec::new();
    for name in names {
        parts.push(format!("[시트] {}", name));
        let Ok(range) = workbook.worksheet_range(&name) else { continue };
        let first_row = range.start().map_or(0, |(r, _)| r as usize);
        for (i, row) in range.rows().enumerate() {
            if first_row + i >= 200 {
                break;
            }
            let cells: Vec<String> = row
                .iter()
                .filter_map(|c| match c {
                    Data::Empty => None,
                    Data::String(s) if s.is_empty() => None,
                    Data::String(s) => Some(s.clone()),
                    Data::Int(n) => Some(n.to_string()),
                    // XML에 소수점이 없는 수는 openpyxl이 int로 읽는다
                    Data::Float(f) if f.fract() == 0.0 && f.abs() < 1e16 => Some(format!("{}", *f as i64)),
                    Data::Float(f) => Some(py_float(*f)),
                    Data::Bool(b) => Some(if *b { "True".into() } else { "False".into() }),
                    Data::DateTime(d) if d.is_datetime() => Some(excel_datetime(d.as_f64())),
                    Data::DateTime(d) => Some(py_float(d.as_f64())),
                    Data::DateTimeIso(s) | Data::DurationIso(s) => Some(s.clone()),
                    Data::Error(e) => Some(e.to_string()),
                })
                .collect();
            if !cells.is_empty() {
                parts.push(cells.join(" | "));
            }
        }
    }
    Ok(parts.join("\n"))
}

/// HWP 5.0 PARA_TEXT 레코드에서 글자만 (UTF-16 단위로 모아 대리 쌍도 살린다)
fn hwp_para_text(payload: &[u8]) -> String {
    let n = payload.len() / 2;
    let mut units: Vec<u16> = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let code = u16::from_le_bytes([payload[i * 2], payload[i * 2 + 1]]);
        if code >= 32 {
            units.push(code);
            i += 1;
        } else if code == 0 || code == 10 || code == 13 || (24..=31).contains(&code) {
            units.push(if code == 10 || code == 13 { b'\n' as u16 } else { b' ' as u16 });
            i += 1;
        } else {
            // 확장/인라인 컨트롤은 8글자 폭
            if code == 9 {
                units.push(b' ' as u16);
            }
            i += 8;
        }
    }
    decode_utf16_ignore(units)
}

fn read_hwp_body(cfb: &mut cfb::CompoundFile<File>) -> Result<String> {
    let mut header = Vec::new();
    cfb.open_stream("/FileHeader")?.read_to_end(&mut header)?;
    if header.len() < 40 {
        bail!("FileHeader가 짧습니다");
    }
    let flags = u32::from_le_bytes([header[36], header[37], header[38], header[39]]);
    if flags & 0x2 != 0 {
        return Ok(String::new()); // 암호 문서
    }
    let compressed = flags & 0x1 != 0;

    // BodyText/Section0, Section1, … 을 번호 순서대로
    let mut sections: Vec<(u64, String)> = cfb
        .walk()
        .filter(|e| e.is_stream())
        .filter_map(|e| {
            let p = e.path().to_string_lossy().replace('\\', "/");
            let rest = p.strip_prefix("/BodyText/")?;
            let digits: String = rest.chars().filter(|c| c.is_ascii_digit()).collect();
            Some((digits.parse().unwrap_or(0), p))
        })
        .collect();
    sections.sort();

    let mut parts = Vec::new();
    let mut total = 0usize;
    for (_, sec) in sections {
        let mut raw = Vec::new();
        cfb.open_stream(&sec)?.read_to_end(&mut raw)?;
        let data = if compressed {
            let mut out = Vec::new();
            flate2::read::DeflateDecoder::new(&raw[..]).read_to_end(&mut out)?;
            out
        } else {
            raw
        };
        let mut pos = 0;
        while pos + 4 <= data.len() {
            let head = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            pos += 4;
            let tag = head & 0x3FF;
            let mut size = (head >> 20) as usize;
            if size == 0xFFF {
                if pos + 4 > data.len() {
                    break;
                }
                size = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
                pos += 4;
            }
            if tag == 67 {
                // HWPTAG_PARA_TEXT (파이썬처럼 잘린 레코드도 남은 만큼 읽는다)
                let t = hwp_para_text(&data[pos.min(data.len())..(pos + size).min(data.len())]);
                total += t.chars().count();
                parts.push(t);
            }
            pos += size;
        }
        if total > MAX_CHARS {
            break;
        }
    }
    Ok(parts.join("\n"))
}

pub fn read_hwp(path: &Path) -> Result<String> {
    let mut cfb = cfb::CompoundFile::open(File::open(path)?).map_err(|e| NotSupported(format!("HWP 구조를 읽지 못함: {}", e)))?;
    if let Ok(text) = read_hwp_body(&mut cfb) {
        if !text.trim_matches(is_py_space).is_empty() {
            return Ok(text);
        }
    }
    // 본문을 못 읽으면 미리보기 글자로
    if let Ok(mut stream) = cfb.open_stream("/PrvText") {
        let mut prv = Vec::new();
        stream.read_to_end(&mut prv)?;
        return Ok(decode_utf16_ignore(prv.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]))));
    }
    Ok(String::new())
}

// ----- PDF (PDFium) -----

/// pdfium.dll: 실행 파일 옆 → 기준 폴더 → models/pdfium 순
fn find_pdfium() -> Option<std::path::PathBuf> {
    let base = crate::config::base_dir();
    let mut cands = Vec::new();
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        cands.push(dir.join("pdfium.dll"));
    }
    cands.push(base.join("pdfium.dll"));
    cands.push(base.join("models").join("pdfium").join("pdfium.dll"));
    cands.into_iter().find(|p| p.is_file())
}

/// 프로세스에 하나만 올리는 PDFium (thread_safe 기능이라 여러 스레드에서 불러도 차례로 처리된다)
fn pdfium() -> Option<&'static pdfium_render::prelude::Pdfium> {
    use pdfium_render::prelude::Pdfium;
    static PDFIUM: std::sync::OnceLock<Option<Pdfium>> = std::sync::OnceLock::new();
    PDFIUM
        .get_or_init(|| {
            let path = find_pdfium()?;
            Pdfium::bind_to_library(&path).ok().map(Pdfium::new)
        })
        .as_ref()
}

fn is_password_error(e: &pdfium_render::prelude::PdfiumError) -> bool {
    use pdfium_render::prelude::{PdfiumError, PdfiumInternalError};
    matches!(e, PdfiumError::PdfiumLibraryInternalError(PdfiumInternalError::PasswordError))
}

/// 파이썬 `_read_pdf`(PyMuPDF)처럼 쪽마다 글자를 꺼내 줄바꿈으로 잇는다. 암호 문서는 빈 글자.
/// (MuPDF와 PDFium은 글자 순서·공백이 조금씩 달라 기존 색인과 글자가 똑같지는 않다)
pub fn read_pdf(path: &Path) -> Result<String> {
    let pdfium = pdfium().ok_or_else(|| NotSupported("pdfium.dll을 찾지 못했습니다".into()))?;
    let doc = match pdfium.load_pdf_from_file(path, None) {
        Ok(d) => d,
        Err(e) if is_password_error(&e) => return Ok(String::new()),
        Err(e) => bail!("PDF를 열지 못함: {}", e),
    };
    let mut parts = Vec::new();
    let mut total = 0usize;
    for page in doc.pages().iter() {
        let t = page.text().map(|t| t.all()).unwrap_or_default();
        // PDFium은 줄 끝이 \r\n이다 (MuPDF는 \n)
        let t = t.replace("\r\n", "\n").replace('\r', "\n");
        total += t.chars().count();
        parts.push(t);
        if total > MAX_CHARS {
            break;
        }
    }
    Ok(parts.join("\n"))
}

/// 글자가 없는 스캔 PDF용: 첫 쪽을 80dpi 그림으로 (파이썬 `pdf_first_page_image`).
/// 암호 문서나 빈 문서는 None.
pub fn pdf_first_page_image(path: &Path) -> Result<Option<image::RgbImage>> {
    use pdfium_render::prelude::PdfRenderConfig;
    let pdfium = pdfium().ok_or_else(|| NotSupported("pdfium.dll을 찾지 못했습니다".into()))?;
    let doc = match pdfium.load_pdf_from_file(path, None) {
        Ok(d) => d,
        Err(e) if is_password_error(&e) => return Ok(None),
        Err(e) => bail!("PDF를 열지 못함: {}", e),
    };
    if doc.pages().len() == 0 {
        return Ok(None);
    }
    let page = doc.pages().get(0)?;
    // MuPDF처럼 (포인트 × 80/72)를 올림해서 픽셀 크기로
    let zoom = 80.0 / 72.0;
    let w = ((page.width().value * zoom) - 0.001).ceil().max(1.0) as i32;
    let h = ((page.height().value * zoom) - 0.001).ceil().max(1.0) as i32;
    let bitmap = page.render_with_config(&PdfRenderConfig::new().set_target_size(w, h))?;
    Ok(Some(bitmap.as_image()?.to_rgb8()))
}

/// Rust로는 아직 못 읽는 파일 (pdfium.dll이 없을 때의 PDF, cfb가 못 여는 HWP). 색인기는 이런 파일을 실패로 치지 않고
/// 대기로 남겨 파이썬 색인기가 마저 하게 한다.
#[derive(Debug)]
pub struct NotSupported(pub String);

impl std::fmt::Display for NotSupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NotSupported {}

pub fn extract_text(path: &Path, ext: &str) -> Result<String> {
    let raw = match ext {
        "pdf" => read_pdf(path)?,
        "docx" => read_docx(path)?,
        "pptx" => read_pptx(path)?,
        "hwpx" => read_hwpx(path)?,
        "xlsx" => read_xlsx(path)?,
        "hwp" => read_hwp(path)?,
        _ => read_plain(path)?,
    };
    Ok(clean(&raw).chars().take(MAX_CHARS).collect())
}

/// 파이썬 `chunk_text`: 앞에서부터 size 글자씩, 가능하면 줄/문장 끝에서 자른다
pub fn chunk_text(text: &str, size: usize, max_chunks: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut chunks = Vec::new();
    let mut pos = 0;
    while pos < n && chunks.len() < max_chunks {
        let mut end = (pos + size).min(n);
        if end < n {
            let w = &chars[pos..end];
            let rfind = |pat: &[char]| -> isize {
                if w.len() < pat.len() {
                    return -1;
                }
                (0..=w.len() - pat.len()).rev().find(|&i| &w[i..i + pat.len()] == pat).map_or(-1, |i| i as isize)
            };
            let cut = rfind(&['\n']).max(rfind(&['.', ' '])).max(rfind(&['다', '.', ' ']));
            if cut as f64 > size as f64 * 0.5 {
                end = pos + cut as usize + 1;
            }
        }
        let piece: String = chars[pos..end].iter().collect();
        let piece = piece.trim_matches(is_py_space);
        if !piece.is_empty() {
            chunks.push(piece.to_string());
        }
        pos = end;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_matches_python() {
        // 파이썬: _clean("Hello\0 World!  \n\n\nThis is <|image|> test.\n<|video|>")
        assert_eq!(
            clean("Hello\0 World!  \n\n\nThis is <|image|> test.\n<|video|>"),
            "Hello World! \nThis is <image> test.\n<video>"
        );
        // 줄 앞뒤 공백은 빈 줄 덩어리가 아니면 그대로 둔다
        assert_eq!(clean("  a \n b  \n \n  c "), "a \n b \n c");
    }

    #[test]
    fn chunk_prefers_last_break() {
        let text = "가나다라마바사.\n아자차카타파하. 끝";
        // 창(12글자) 안에서 가장 뒤에 있는 끊는 자리를 고른다
        let chunks = chunk_text(text, 12, 5);
        assert_eq!(chunks[0], "가나다라마바사.");
    }

    #[test]
    fn excel_dates() {
        assert_eq!(excel_datetime(45292.0), "2024-01-01 00:00:00");
        assert_eq!(excel_datetime(45292.5), "2024-01-01 12:00:00");
        assert_eq!(excel_datetime(0.25), "06:00:00");
    }

    #[test]
    fn python_floats() {
        assert_eq!(py_float(0.1), "0.1");
        assert_eq!(py_float(1e-5), "1e-05");
        assert_eq!(py_float(12345.5), "12345.5");
        assert_eq!(py_float(2.5e20), "2.5e+20");
    }
}
