"""설정 로딩과 파일 종류 분류."""
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE / "config.json"
DEFAULT_DB = BASE / "data" / "index.db"
DEFAULT_MODELS = BASE / "models"

os.environ.setdefault("HF_HOME", str(DEFAULT_MODELS))

KIND_EXTS = {
    "doc": "pdf docx doc pptx ppt xlsx xls hwp hwpx txt md rtf csv odt epub",
    "image": "jpg jpeg png webp bmp gif tif tiff heic psd",
    "video": "mp4 mkv avi mov wmv webm m4v",
    "audio": "mp3 wav flac m4a ogg aac wma",
    "archive": "zip 7z rar tar gz iso",
    "app": "exe msi lnk bat cmd url apk",
    "code": "py js ts tsx jsx java c cpp h hpp cs go rs rb php html css json yaml yml toml xml sql sh ipynb",
}
EXT_KIND = {e: k for k, exts in KIND_EXTS.items() for e in exts.split()}

KIND_LABEL = {
    "dir": "폴더", "doc": "문서", "image": "이미지", "video": "동영상",
    "audio": "오디오", "archive": "압축 파일", "app": "프로그램", "code": "코드",
}

# 내용을 읽을 수 있는 확장자 (나머지는 파일명과 위치만 색인)
TEXT_EXTS = set("txt md csv".split()) | set(KIND_EXTS["code"].split())
PARSED_DOC_EXTS = set("pdf docx pptx xlsx hwp hwpx".split())
IMAGE_CONTENT_EXTS = set("jpg jpeg png webp bmp gif tif tiff".split())
RUNNABLE_EXTS = set("exe msi lnk bat cmd url".split())


def load(path=None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cur_drive = (BASE.drive + "\\") if BASE.drive else ""
    resolved_roots = []
    for r in cfg.get("roots", []):
        if r in ("<current_drive>", "<drive>", "%DRIVE%"):
            if cur_drive and cur_drive not in resolved_roots:
                resolved_roots.append(cur_drive)
        elif r in ("<project>", "."):
            resolved_roots.append(str(BASE))
        else:
            resolved_roots.append(os.path.normpath(os.path.expanduser(r)))
    cfg["roots"] = resolved_roots
    for key in ("exclude_dir_names", "cloud_dir_names"):
        cfg[key] = [s.lower() for s in cfg[key]]
    for key in ("exclude_paths", "cloud_paths"):
        cfg[key] = [os.path.normpath(os.path.expanduser(p)) for p in cfg[key]]
    return cfg


def has_content(kind: str, ext: str, cfg: dict) -> bool:
    """이 파일의 내용까지 임베딩할 수 있는지."""
    if kind not in cfg["content_kinds"]:
        return False
    if kind == "image":
        return ext in IMAGE_CONTENT_EXTS
    return ext in TEXT_EXTS or ext in PARSED_DOC_EXTS
