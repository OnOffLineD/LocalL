"""파일에서 텍스트/이미지를 꺼낸다. 실패하면 예외를 던지고, 호출 쪽에서 건너뛴다."""
import html
import re
import struct
import zipfile
import zlib

MAX_CHARS = 400_000


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore").encode("utf-8", "ignore").decode("utf-8")


def _read_plain(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read(MAX_CHARS)
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="ignore")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp949", errors="ignore")


def _read_pdf(path: str) -> str:
    import pymupdf

    parts, total = [], 0
    with pymupdf.open(path) as doc:
        if doc.needs_pass:
            return ""
        for page in doc:
            t = page.get_text()
            parts.append(t)
            total += len(t)
            if total > MAX_CHARS:
                break
    return "\n".join(parts)


def pdf_first_page_image(path: str):
    """글자가 없는 스캔 PDF용: 첫 쪽을 이미지로."""
    import pymupdf
    from PIL import Image

    with pymupdf.open(path) as doc:
        if doc.needs_pass or doc.page_count == 0:
            return None
        pix = doc[0].get_pixmap(dpi=80)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _read_docx(path: str) -> str:
    import docx

    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs]
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text for c in row.cells))
    return "\n".join(parts)


def _read_pptx(path: str) -> str:
    from pptx import Presentation

    parts = []
    for slide in Presentation(path).slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
    return "\n".join(parts)


def _read_xlsx(path: str) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    parts = []
    try:
        for ws in wb.worksheets[:6]:
            parts.append(f"[시트] {ws.title}")
            for row in ws.iter_rows(max_row=200, values_only=True):
                cells = [str(c) for c in row if c is not None]
                if cells:
                    parts.append(" | ".join(cells))
    finally:
        wb.close()
    return "\n".join(parts)


def _read_hwpx(path: str) -> str:
    parts = []
    with zipfile.ZipFile(path) as z:
        sections = sorted(n for n in z.namelist() if re.match(r"Contents/section\d+\.xml$", n))
        for name in sections:
            xml = z.read(name).decode("utf-8", errors="ignore")
            for run in re.findall(r"<hp:t(?:\s[^>]*)?>(.*?)</hp:t>", xml, re.S):
                line = html.unescape(re.sub(r"<[^>]+>", "", run))
                if line.strip():
                    parts.append(line)
    return "\n".join(parts)


def _hwp_para_text(payload: bytes) -> str:
    """HWP 5.0 PARA_TEXT 레코드에서 글자만 뽑는다."""
    out = []
    n = len(payload) // 2
    i = 0
    while i < n:
        code = struct.unpack_from("<H", payload, i * 2)[0]
        if code >= 32:
            out.append(chr(code))
            i += 1
        elif code in (0, 10, 13) or 24 <= code <= 31:
            out.append("\n" if code in (10, 13) else " ")
            i += 1
        else:  # 확장/인라인 컨트롤은 8글자 폭
            if code == 9:
                out.append(" ")
            i += 8
    return "".join(out)


def _read_hwp(path: str) -> str:
    import olefile

    ole = olefile.OleFileIO(path)
    try:
        try:
            header = ole.openstream("FileHeader").read()
            flags = struct.unpack_from("<I", header, 36)[0]
            if flags & 0x2:  # 암호 문서
                return ""
            compressed = bool(flags & 0x1)
            parts, total = [], 0
            sections = sorted(
                (s for s in ole.listdir() if s[0] == "BodyText"),
                key=lambda s: int(re.sub(r"\D", "", s[1]) or 0),
            )
            for sec in sections:
                data = ole.openstream(sec).read()
                if compressed:
                    data = zlib.decompress(data, -15)
                pos = 0
                while pos + 4 <= len(data):
                    head = struct.unpack_from("<I", data, pos)[0]
                    pos += 4
                    tag, size = head & 0x3FF, head >> 20
                    if size == 0xFFF:
                        size = struct.unpack_from("<I", data, pos)[0]
                        pos += 4
                    if tag == 67:  # HWPTAG_PARA_TEXT
                        t = _hwp_para_text(data[pos:pos + size])
                        parts.append(t)
                        total += len(t)
                    pos += size
                if total > MAX_CHARS:
                    break
            text = "\n".join(parts)
            if text.strip():
                return text
        except Exception:
            pass
        if ole.exists("PrvText"):  # 미리보기 텍스트로 대체
            return ole.openstream("PrvText").read().decode("utf-16-le", errors="ignore")
        return ""
    finally:
        ole.close()


_READERS = {
    "pdf": _read_pdf, "docx": _read_docx, "pptx": _read_pptx,
    "xlsx": _read_xlsx, "hwpx": _read_hwpx, "hwp": _read_hwp,
}


def extract_text(path: str, ext: str) -> str:
    reader = _READERS.get(ext, _read_plain)
    return _clean(reader(path))[:MAX_CHARS]


def chunk_text(text: str, size: int, max_chunks: int) -> list[str]:
    """앞에서부터 size 글자씩, 가능하면 줄/문장 끝에서 자른다."""
    chunks, pos, n = [], 0, len(text)
    while pos < n and len(chunks) < max_chunks:
        end = min(pos + size, n)
        if end < n:
            window = text[pos:end]
            cut = max(window.rfind("\n"), window.rfind(". "), window.rfind("다. "))
            if cut > size * 0.5:
                end = pos + cut + 1
        piece = text[pos:end].strip()
        if piece:
            chunks.append(piece)
        pos = end
    return chunks


def load_image(path: str, max_side: int = 896):
    """작게 줄인 RGB 이미지. 아이콘처럼 너무 작은 것은 None."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im.draft("RGB", (max_side, max_side))  # JPEG는 줄여서 디코딩
        if min(im.size) < 96:
            return None
        im = ImageOps.exif_transpose(im)
        im.thumbnail((max_side, max_side))
        return im.convert("RGB")
