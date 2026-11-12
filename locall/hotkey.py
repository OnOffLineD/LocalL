"""전역 단축키 (Win32 RegisterHotKey). 추가 패키지 없이 ctypes만 쓴다."""
import ctypes
import threading
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

MODS = {"alt": 0x1, "ctrl": 0x2, "shift": 0x4, "win": 0x8}
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_APP_SET = 0x8001

KEYS = {
    "space": 0x20, "enter": 0x0D, "tab": 0x09, "backspace": 0x08, "insert": 0x2D, "delete": 0x2E,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "`": 0xC0, "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD, "\\": 0xDC,
    ";": 0xBA, "'": 0xDE, ",": 0xBC, ".": 0xBE, "/": 0xBF,
}
KEYS.update({chr(c).lower(): c for c in range(0x41, 0x5B)})   # a-z
KEYS.update({chr(c): c for c in range(0x30, 0x3A)})           # 0-9
KEYS.update({f"f{i}": 0x6F + i for i in range(1, 25)})        # f1-f24
VK_NAME = {vk: name for name, vk in KEYS.items()}
MODIFIER_VKS = {0x10, 0x11, 0x12, 0x5B, 0x5C, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5}


def parse(text: str) -> tuple[int, int]:
    """'ctrl+shift+space' → (조합 키 비트, 가상 키 코드). 잘못된 형식이면 ValueError."""
    parts = [p.strip().lower() for p in text.split("+") if p.strip()]
    if not parts or parts[-1] not in KEYS:
        raise ValueError(f"알 수 없는 단축키: {text!r}")
    mods = 0
    for p in parts[:-1]:
        if p not in MODS:
            raise ValueError(f"알 수 없는 조합 키: {p!r}")
        mods |= MODS[p]
    return mods, KEYS[parts[-1]]


def to_text(mods: int, vk: int):
    """(조합 키 비트, 가상 키 코드) → 'ctrl+shift+space'. 지원하지 않는 키면 None."""
    if vk not in VK_NAME:
        return None
    names = [n for n in ("ctrl", "alt", "shift", "win") if mods & MODS[n]]
    return "+".join(names + [VK_NAME[vk]])


def display(text: str) -> str:
    return "+".join(p.upper() if len(p) <= 3 and p[0] == "f" or len(p) == 1 else p.capitalize()
                    for p in text.split("+"))


def pressed_mods() -> int:
    """지금 눌려 있는 조합 키."""
    def down(vk):
        return user32.GetAsyncKeyState(vk) & 0x8000

    mods = 0
    if down(0x11):
        mods |= MODS["ctrl"]
    if down(0x12):
        mods |= MODS["alt"]
    if down(0x10):
        mods |= MODS["shift"]
    if down(0x5B) or down(0x5C):
        mods |= MODS["win"]
    return mods


class HotkeyThread(threading.Thread):
    """단축키를 등록하고, 눌릴 때마다 on_press()를 (이 스레드에서) 부른다."""

    def __init__(self, on_press):
        super().__init__(daemon=True)
        self.on_press = on_press
        self.current = None
        self._ready = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._req, self._ok = None, False

    def run(self):
        self.tid = kernel32.GetCurrentThreadId()
        msg = wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)  # 메시지 큐 생성
        self._ready.set()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY:
                self.on_press()
            elif msg.message == WM_APP_SET:
                self._apply()

    def _apply(self):
        mods, vk = self._req
        if self.current:
            user32.UnregisterHotKey(None, 1)
        ok = bool(user32.RegisterHotKey(None, 1, mods | MOD_NOREPEAT, vk))
        if ok:
            self.current = (mods, vk)
        elif self.current:  # 실패하면 이전 단축키로 되돌린다
            user32.RegisterHotKey(None, 1, self.current[0] | MOD_NOREPEAT, self.current[1])
        self._ok = ok
        self._done.set()

    def set(self, text: str) -> bool:
        """단축키를 바꾼다. 다른 프로그램이 이미 쓰는 조합이면 False."""
        with self._lock:
            self._req = parse(text)
            self._ready.wait()
            self._done.clear()
            user32.PostThreadMessageW(self.tid, WM_APP_SET, 0, 0)
            return self._done.wait(3) and self._ok
