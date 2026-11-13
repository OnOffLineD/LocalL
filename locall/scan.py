"""폴더 순회. 클라우드 드라이브와 시스템/개발 부산물 폴더는 건너뛴다."""
import ctypes
import json
import os
import string

from .config import EXT_KIND

ATTR_HIDDEN = 0x2
ATTR_SYSTEM = 0x4
ATTR_REPARSE = 0x400
# 내용이 로컬에 없는 파일(읽으면 다운로드가 일어남)
ATTR_CLOUD = 0x1000 | 0x40000 | 0x400000  # OFFLINE | RECALL_ON_OPEN | RECALL_ON_DATA_ACCESS

UNITY_JUNK = {"library", "temp", "obj", "logs"}


def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))


def volume_label(drive_root: str) -> str:
    buf = ctypes.create_unicode_buffer(261)
    ok = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(drive_root), buf, 261, None, None, None, None, 0
    )
    return buf.value if ok else ""


def detect_cloud_paths(cfg: dict) -> list[str]:
    """이 PC에서 클라우드 동기화에 쓰이는 폴더/드라이브를 찾는다."""
    found = list(cfg.get("cloud_paths", []))
    for var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        if os.environ.get(var):
            found.append(os.environ[var])
    # Google Drive 데스크톱 앱은 가상 드라이브(라벨 "Google Drive")로 붙는다
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.exists(root) and "google drive" in volume_label(root).lower():
            found.append(root)
    for base in (os.environ.get("LOCALAPPDATA"), os.environ.get("APPDATA")):
        info = os.path.join(base or "", "Dropbox", "info.json")
        if os.path.isfile(info):
            try:
                with open(info, encoding="utf-8") as f:
                    for account in json.load(f).values():
                        if account.get("path"):
                            found.append(account["path"])
            except (OSError, ValueError):
                pass
    return sorted({_norm(p) for p in found})


def _is_under(path_n: str, parents: list[str]) -> bool:
    return any(path_n == p or path_n.startswith(p.rstrip("\\") + "\\") for p in parents)


def walk(roots: list[str], cfg: dict, report: dict):
    """(path, name, is_dir, ext, kind, size, mtime)를 내놓는다.

    report["cloud"]에는 건너뛴 클라우드 폴더, report["denied"]에는 접근 거부 수가 쌓인다.
    """
    cloud = detect_cloud_paths(cfg)
    excluded = [_norm(p) for p in cfg["exclude_paths"]]
    exclude_names = set(cfg["exclude_dir_names"])
    cloud_names = cfg["cloud_dir_names"]
    kinds = set(cfg["include_kinds"])
    skip_hidden = cfg.get("skip_hidden", True)
    report.setdefault("cloud", [])
    report.setdefault("denied", 0)

    stack = []
    # 다른 루트 안에 든 루트(예: D:\ 와 D:\사진)는 그 루트로서 한 번만 훑는다
    root_keys = set()
    for root in roots:
        root_n = _norm(root)
        if root_n.rstrip("\\") in root_keys:
            continue  # 같은 루트가 두 번 적힘
        root_keys.add(root_n.rstrip("\\"))
        if _is_under(root_n, excluded):
            continue  # 루트 자체(또는 그 위 폴더)를 제외했다
        if _is_under(root_n, cloud):
            report["cloud"].append(root)
        elif os.path.isdir(root):
            stack.append(root)

    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                entries = list(it)
        except OSError:
            report["denied"] += 1
            continue

        names = {e.name.lower() for e in entries}
        if "pyvenv.cfg" in names:  # 파이썬 가상환경
            continue
        skip_here = UNITY_JUNK if {"assets", "projectsettings"} <= names else ()

        for e in entries:
            try:
                st = e.stat(follow_symlinks=False)
            except OSError:
                continue
            attrs = st.st_file_attributes
            if attrs & (ATTR_CLOUD | ATTR_REPARSE):
                continue
            if skip_hidden and attrs & (ATTR_HIDDEN | ATTR_SYSTEM):
                continue
            lname = e.name.lower()

            if e.is_dir(follow_symlinks=False):
                if lname.startswith(".") or lname in exclude_names or lname in skip_here:
                    continue
                path_n = _norm(e.path)
                if _is_under(path_n, excluded):
                    continue
                if _is_under(path_n, cloud) or lname in cloud_names or lname.startswith("onedrive - "):
                    report["cloud"].append(e.path)
                    continue
                if path_n not in root_keys:
                    stack.append(e.path)
                if "dir" in kinds:
                    yield e.path, e.name, True, "", "dir", 0, st.st_mtime
            else:
                ext = os.path.splitext(lname)[1][1:]
                kind = EXT_KIND.get(ext)
                if kind in kinds and not lname.startswith("~$"):
                    yield e.path, e.name, False, ext, kind, st.st_size, st.st_mtime
