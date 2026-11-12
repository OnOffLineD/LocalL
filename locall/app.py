"""단축키로 뜨는 검색 팝업:  pythonw -m locall.app

백그라운드에 머물다가 전역 단축키를 누르면 작은 검색창이 뜬다.
  Enter 열기/실행 · Ctrl+Enter 위치 열기 · ↑↓ 선택 · Tab 종류 바꾸기 · Esc 닫기
단축키는 팝업 오른쪽 아래 ⚙ 메뉴에서 바꾼다 (config.json의 "hotkey"에 저장).
"""
import argparse
import ctypes
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
import winreg

from . import actions, config, hotkey

ROWS = 7           # 한 번에 보이는 줄 수
FETCH = 28         # 검색해 오는 결과 수
WIDTH = 720
FONT = "Malgun Gothic"
DEFAULT_HOTKEY = "ctrl+shift+space"
KINDS = [("all", "전체"), ("doc", "문서"), ("image", "이미지"), ("dir", "폴더"),
         ("video", "동영상"), ("audio", "오디오"), ("app", "프로그램"), ("archive", "압축")]
MATCH = {"name": "이름·위치", "text": "문서 내용", "image": "이미지 내용"}

LIGHT = dict(bg="#ffffff", border="#b9b8b3", line="#e6e5e1", fg="#1c1c1a", dim="#6d6c67",
             sel="#e4eefd", icon="#eeede9")
DARK = dict(bg="#202124", border="#4a4c51", line="#33353a", fg="#ececea", dim="#9d9c96",
            sel="#2c3a50", icon="#2e3035")

log = logging.getLogger("locall")
user32 = ctypes.WinDLL("user32", use_last_error=True)


def system_theme() -> dict:
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        return LIGHT if winreg.QueryValueEx(key, "AppsUseLightTheme")[0] else DARK
    except OSError:
        return LIGHT


def shorten(text: str, limit: int) -> str:
    """길면 가운데를 줄인다 (경로의 앞과 끝이 모두 보이도록)."""
    if len(text) <= limit:
        return text
    head = limit // 3
    return text[:head] + " … " + text[-(limit - head - 3):]


def save_hotkey(text: str):
    """config.json의 다른 내용과 서식은 그대로 두고 hotkey 값만 바꾼다."""
    raw = config.CONFIG_PATH.read_text(encoding="utf-8")
    new, n = re.subn(r'("hotkey"\s*:\s*)"[^"]*"', lambda m: f'{m.group(1)}"{text}"', raw, count=1)
    if not n:
        new = raw.replace("{", '{\n  "hotkey": "' + text + '",', 1)
    config.CONFIG_PATH.write_text(new, encoding="utf-8")


class App:
    def __init__(self, cfg: dict, db_path: str, show_on_start: bool = True):
        self.cfg, self.db_path = cfg, db_path
        self.c = system_theme()
        self.events = queue.Queue()     # 다른 스레드 → 화면 스레드
        self.requests = queue.Queue()   # 검색 요청
        self.thumb_jobs = queue.Queue()
        self.searcher = None
        self.results, self.sel, self.top = [], 0, 0
        self.kind_i = 0
        self.seq = 0
        self.shown_query = None         # 지금 화면의 결과가 어떤 (검색어, 종류)의 것인지
        self.visible = self.modal = False
        self.thumbs = {}                # 파일 id → PhotoImage (없으면 None)
        self.timer = None
        self.status = "모델과 색인을 불러오는 중…"

        self._build()
        self.hk = hotkey.HotkeyThread(lambda: self.events.put(("toggle",)))
        self.hk.start()
        self.hotkey_text = str(cfg.get("hotkey") or DEFAULT_HOTKEY).lower()
        try:
            self.hotkey_ok = self.hk.set(self.hotkey_text)
        except ValueError:
            self.hotkey_text, self.hotkey_ok = DEFAULT_HOTKEY, self.hk.set(DEFAULT_HOTKEY)

        threading.Thread(target=self._load, daemon=True).start()
        threading.Thread(target=self._search_worker, daemon=True).start()
        threading.Thread(target=self._thumb_worker, daemon=True).start()
        self.root.after(40, self._pump)
        self._refresh_text()
        if show_on_start or not self.hotkey_ok:
            self.show()

    # ---------------- 화면 구성 ----------------
    def px(self, n):
        return int(round(n * self.scale))

    def _build(self):
        c = self.c
        self.root = root = tk.Tk()
        root.withdraw()
        root.title("LocalL")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.report_callback_exception = lambda *exc: log.error("화면 오류", exc_info=exc)
        self.scale = root.winfo_fpixels("1i") / 96
        px = self.px

        self.outer = tk.Frame(root, bg=c["border"], padx=1, pady=1)
        self.outer.pack(fill="both", expand=True)
        box = tk.Frame(self.outer, bg=c["bg"])
        box.pack(fill="both", expand=True)

        self.var = tk.StringVar()
        self.entry = tk.Entry(box, textvariable=self.var, font=(FONT, 15), bd=0, relief="flat",
                              bg=c["bg"], fg=c["fg"], insertbackground=c["fg"],
                              highlightthickness=0, selectbackground=c["sel"], selectforeground=c["fg"])
        self.entry.pack(fill="x", padx=px(16), pady=(px(12), px(10)))
        self.var.trace_add("write", lambda *_: self._schedule())

        tk.Frame(box, bg=c["line"], height=1).pack(fill="x")
        self.list = tk.Frame(box, bg=c["bg"])
        self.list.pack(fill="x")
        self.rows = [self._make_row(i) for i in range(ROWS)]

        self.detail = tk.Label(box, bg=c["bg"], fg=c["dim"], font=(FONT, 9), anchor="w", justify="left",
                               wraplength=px(WIDTH - 34))
        self.detail.pack(fill="x", padx=px(16), pady=(px(6), px(2)))

        tk.Frame(box, bg=c["line"], height=1).pack(fill="x", pady=(px(4), 0))
        foot = tk.Frame(box, bg=c["bg"])
        foot.pack(fill="x", padx=px(12), pady=px(5))
        self.hint = tk.Label(foot, bg=c["bg"], fg=c["dim"], font=(FONT, 9), anchor="w")
        self.hint.pack(side="left")
        gear = tk.Label(foot, text="⚙ 설정", bg=c["bg"], fg=c["dim"], font=(FONT, 9), cursor="hand2")
        gear.pack(side="right")
        gear.bind("<Button-1>", self._menu)

        e = self.entry
        e.bind("<Escape>", lambda ev: self.hide())
        e.bind("<Return>", lambda ev: self._enter(False))
        e.bind("<Control-Return>", lambda ev: self._enter(True))
        e.bind("<Down>", lambda ev: self._move(1))
        e.bind("<Up>", lambda ev: self._move(-1))
        e.bind("<Next>", lambda ev: self._move(ROWS))
        e.bind("<Prior>", lambda ev: self._move(-ROWS))
        e.bind("<Tab>", lambda ev: self._cycle_kind(1))
        e.bind("<Shift-Tab>", lambda ev: self._cycle_kind(-1))
        root.bind("<FocusOut>", self._focus_out)
        root.bind("<MouseWheel>", lambda ev: self._move(-1 if ev.delta > 0 else 1))

    def _make_row(self, i):
        c, px = self.c, self.px
        frame = tk.Frame(self.list, bg=c["bg"])
        holder = tk.Frame(frame, bg=c["icon"], width=px(40), height=px(40))
        holder.pack(side="left", padx=(px(14), px(10)), pady=px(5))
        holder.pack_propagate(False)
        icon = tk.Label(holder, bg=c["icon"], fg=c["dim"], font=(FONT, 8, "bold"))
        icon.pack(fill="both", expand=True)
        meta = tk.Label(frame, bg=c["bg"], fg=c["dim"], font=(FONT, 9), anchor="e")
        meta.pack(side="right", padx=(px(8), px(14)))
        text = tk.Frame(frame, bg=c["bg"])
        text.pack(side="left", fill="x", expand=True)
        name = tk.Label(text, bg=c["bg"], fg=c["fg"], font=(FONT, 11, "bold"), anchor="w")
        name.pack(fill="x")
        path = tk.Label(text, bg=c["bg"], fg=c["dim"], font=(FONT, 9), anchor="w")
        path.pack(fill="x")
        row = dict(frame=frame, icon=icon, name=name, path=path, meta=meta, text=text, shown=False)
        for w in (frame, holder, icon, meta, text, name, path):
            w.bind("<Button-1>", lambda ev, i=i: self._click(i))
            w.bind("<Double-Button-1>", lambda ev, i=i: (self._click(i), self._enter(False)))
        return row

    # ---------------- 백그라운드 작업 ----------------
    def _load(self):
        try:
            from .search import Searcher

            self.searcher = Searcher(self.cfg, self.db_path)
            self.events.put(("ready",))
        except Exception as e:
            log.exception("불러오기 실패")
            self.events.put(("error", f"불러오기 실패: {e}"))

    def _search_worker(self):
        while True:
            req = self.requests.get()
            while not self.requests.empty():  # 밀린 요청은 버리고 최신 것만
                req = self.requests.get()
            seq, text, kind = req
            try:
                results = self.searcher.search(text, kind, n=FETCH)
                self.events.put(("results", seq, text, kind, results))
            except Exception as e:
                log.exception("검색 실패")
                self.events.put(("error", f"검색 실패: {e}"))

    def _thumb_worker(self):
        from PIL import ImageOps

        from .extract import load_image

        while True:
            fid, path = self.thumb_jobs.get()
            try:
                im = load_image(path, 160)
                im = ImageOps.fit(im, (self.px(40), self.px(40))) if im else None
            except Exception:
                im = None
            self.events.put(("thumb", fid, im))

    def _pump(self):
        """다른 스레드가 보낸 일을 화면 스레드에서 처리한다."""
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "toggle":
                    if not self.modal:
                        self.hide() if self.visible else self.show()
                elif kind == "ready":
                    n = self.searcher.stats()["files"]
                    self.status = "" if n else "아직 색인이 없습니다. ⚙ 설정 → 색인 만들기/갱신을 실행하세요."
                    self._refresh_text()
                    self._schedule(0)
                elif kind == "error":
                    self.status = ev[1]
                    self._refresh_text()
                elif kind == "results":
                    _, seq, text, k, results = ev
                    if seq == self.seq:
                        self.results, self.sel, self.top = results, 0, 0
                        self.shown_query = (text, k)
                        self._render()
                elif kind == "thumb":
                    from PIL import ImageTk

                    self.thumbs[ev[1]] = ImageTk.PhotoImage(ev[2]) if ev[2] else None
                    self._render()
        except queue.Empty:
            pass
        self.root.after(40, self._pump)

    # ---------------- 검색 ----------------
    def _schedule(self, delay=180):
        if self.timer:
            self.root.after_cancel(self.timer)
        self.timer = self.root.after(delay, self._search_now)

    def _search_now(self):
        self.timer = None
        text = self.var.get().strip()
        self.seq += 1
        if not text or self.searcher is None:
            self.results, self.shown_query = [], (text, KINDS[self.kind_i][0])
            self._render()
            return
        self.requests.put((self.seq, text, KINDS[self.kind_i][0]))

    def _cycle_kind(self, step):
        self.kind_i = (self.kind_i + step) % len(KINDS)
        self._refresh_text()
        self._schedule(0)
        return "break"

    # ---------------- 그리기 ----------------
    def _render(self):
        c = self.c
        visible = self.results[self.top:self.top + ROWS]
        for i, row in enumerate(self.rows):
            if i >= len(visible):
                if row["shown"]:
                    row["frame"].pack_forget()
                    row["shown"] = False
                continue
            r = visible[i]
            bg = c["sel"] if self.top + i == self.sel else c["bg"]
            for key in ("frame", "name", "path", "meta", "text"):
                row[key].configure(bg=bg)
            row["name"].configure(text=shorten(r["name"], 60))
            row["path"].configure(text=shorten(r["dir"], 78))
            label = "실행 파일" if r["runnable"] else MATCH.get(r["match"], "")
            row["meta"].configure(text=label)
            photo = self.thumbs.get(r["id"]) if r["thumb"] else None
            if photo:
                row["icon"].configure(image=photo, text="")
            else:
                row["icon"].configure(image="", text="폴더" if r["kind"] == "dir" else (r["ext"] or "파일").upper()[:5])
                if r["thumb"] and r["id"] not in self.thumbs:
                    self.thumbs[r["id"]] = None
                    self.thumb_jobs.put((r["id"], r["path"]))
            if not row["shown"]:
                row["frame"].pack(fill="x")
                row["shown"] = True
        self._refresh_text()
        self._resize()

    def _refresh_text(self):
        kind = KINDS[self.kind_i][1]
        if self.results:
            r = self.results[self.sel]
            detail = r["path"] + (f"\n{r['snippet']}" if r["snippet"] else "")
            hint = f"{self.sel + 1}/{len(self.results)} · Enter {'실행' if r['runnable'] else '열기'} · Ctrl+Enter 위치 열기 · Tab 종류: {kind}"
        else:
            typed = bool(self.var.get().strip())
            if self.status:
                detail = self.status
            elif typed and self.shown_query == (self.var.get().strip(), KINDS[self.kind_i][0]):
                detail = "찾지 못했습니다. 다른 표현으로 적어 보세요."
            else:
                detail = "찾는 파일을 말하듯이 적어 보세요. 예: 작년 연말정산 서류, 바닷가에서 찍은 사진"
            hint = f"Tab 종류: {kind} · Esc 닫기 · 단축키 {hotkey.display(self.hotkey_text)}"
            if not self.hotkey_ok:
                hint = f"단축키 {hotkey.display(self.hotkey_text)} 등록 실패 — ⚙ 설정에서 바꿔 주세요"
        self.detail.configure(text=detail)
        self.hint.configure(text=hint)

    def _resize(self):
        self.root.update_idletasks()
        w, h = self.px(WIDTH), self.outer.winfo_reqheight()
        x = (self.root.winfo_screenwidth() - w) // 2
        y = int(self.root.winfo_screenheight() * 0.2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    # ---------------- 조작 ----------------
    def _move(self, step):
        if self.results:
            self.sel = max(0, min(len(self.results) - 1, self.sel + step))
            if self.sel < self.top:
                self.top = self.sel
            elif self.sel >= self.top + ROWS:
                self.top = self.sel - ROWS + 1
            self._render()
        return "break"

    def _click(self, i):
        if self.top + i < len(self.results):
            self.sel = self.top + i
            self._render()
        self.entry.focus_set()

    def _enter(self, reveal):
        current = (self.var.get().strip(), KINDS[self.kind_i][0])
        if current != self.shown_query:   # 아직 이 검색어의 결과가 아니면 열지 않고 검색부터
            self._schedule(0)
            return "break"
        if not self.results:
            return "break"
        r = self.results[self.sel]
        try:
            if not os.path.exists(r["path"]):
                raise OSError("파일이 더 이상 그 위치에 없습니다. 색인을 갱신하세요.")
            (actions.reveal if reveal else actions.open_path)(r["path"])
        except OSError as e:
            self.detail.configure(text=f"열지 못했습니다: {e.strerror or e}")
            return "break"
        self.hide()
        return "break"

    def show(self):
        self.visible = True
        self._resize()
        self.root.deiconify()
        self.root.lift()
        self.root.update_idletasks()
        hwnd = user32.GetAncestor(self.root.winfo_id(), 2) or self.root.winfo_id()
        if not user32.SetForegroundWindow(hwnd) or user32.GetForegroundWindow() != hwnd:
            # 윈도우가 포커스 이동을 막을 때 쓰는 관용적인 우회: Alt를 한 번 눌렀다 뗀다
            user32.keybd_event(0x12, 0, 0, 0)
            user32.keybd_event(0x12, 0, 2, 0)
            user32.SetForegroundWindow(hwnd)
        self.entry.focus_force()
        self.entry.select_range(0, "end")
        self.entry.icursor("end")

    def hide(self):
        self.visible = False
        self.root.withdraw()

    def _focus_out(self, _ev):
        def check():
            if self.visible and not self.modal and self.root.focus_displayof() is None:
                self.hide()
        self.root.after(150, check)

    # ---------------- 설정 ----------------
    def _menu(self, ev):
        menu = tk.Menu(self.root, tearoff=0, font=(FONT, 10))
        menu.add_command(label=f"단축키 바꾸기…  (현재 {hotkey.display(self.hotkey_text)})",
                         command=self._change_hotkey)
        menu.add_command(label="색인 만들기/갱신  (새 창에서 실행)", command=self._run_index)
        menu.add_separator()
        menu.add_command(label="LocalL 종료", command=self.root.destroy)
        self.modal = True
        try:
            menu.tk_popup(ev.x_root, ev.y_root)
        finally:
            menu.grab_release()
            self.root.after(300, self._end_modal)

    def _end_modal(self):
        if not any(isinstance(w, tk.Toplevel) for w in self.root.winfo_children()):
            self.modal = False
            if self.visible:
                self.entry.focus_force()

    def _run_index(self):
        python = os.path.join(os.path.dirname(sys.executable), "python.exe")
        subprocess.Popen(["cmd", "/k", python, "-W", "ignore", "-m", "locall.index", "--db", self.db_path],
                         cwd=str(config.BASE), creationflags=subprocess.CREATE_NEW_CONSOLE)

    def _change_hotkey(self):
        c, px = self.c, self.px
        self.modal = True
        dlg = tk.Toplevel(self.root)
        dlg.overrideredirect(True)
        dlg.attributes("-topmost", True)
        frame = tk.Frame(dlg, bg=c["bg"], highlightbackground=c["border"], highlightthickness=1)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text="새 단축키를 눌러 주세요", bg=c["bg"], fg=c["fg"],
                 font=(FONT, 13, "bold")).pack(padx=px(28), pady=(px(18), px(4)))
        msg = tk.Label(frame, bg=c["bg"], fg=c["dim"], font=(FONT, 10),
                       text=f"Ctrl·Alt·Shift·Win 중 하나 이상과 함께 누릅니다.\n현재: {hotkey.display(self.hotkey_text)}   ·   Esc 취소")
        msg.pack(padx=px(28), pady=(0, px(18)))
        dlg.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dlg.winfo_reqwidth()) // 2
        dlg.geometry(f"+{x}+{self.root.winfo_y() + px(40)}")

        def close():
            dlg.destroy()
            self.modal = False
            self._refresh_text()
            if self.visible:
                self.entry.focus_force()

        def on_key(ev):
            vk = ev.keycode
            if vk in hotkey.MODIFIER_VKS:
                return "break"
            # 키가 눌린 순간의 조합 키 상태 (지금 시점의 키 상태를 물으면 이미 뗀 뒤일 수 있다)
            mods = hotkey.pressed_mods() & hotkey.MODS["win"]
            mods |= hotkey.MODS["shift"] if ev.state & 0x1 else 0
            mods |= hotkey.MODS["ctrl"] if ev.state & 0x4 else 0
            mods |= hotkey.MODS["alt"] if ev.state & 0x20000 else 0
            if vk == 0x1B and not mods:
                close()
                return "break"
            text = hotkey.to_text(mods, vk)
            is_fkey = 0x70 <= vk <= 0x87
            if text is None:
                msg.configure(text="이 키는 단축키로 쓸 수 없습니다. 다른 키를 눌러 주세요.\n(한/영 상태면 영문으로 바꿔 보세요)")
            elif not mods and not is_fkey:
                msg.configure(text="Ctrl·Alt·Shift·Win 중 하나 이상을 함께 눌러 주세요.\nEsc 취소")
            elif self.hk.set(text):
                self.hotkey_text, self.hotkey_ok = text, True
                try:
                    save_hotkey(text)
                except OSError:
                    log.exception("단축키 저장 실패")
                close()
            else:
                msg.configure(text=f"{hotkey.display(text)} 은(는) 다른 프로그램이 이미 쓰고 있습니다.\n다른 조합을 눌러 주세요.   ·   Esc 취소")
            return "break"

        dlg.bind("<KeyPress>", on_key)
        dlg.focus_force()
        dlg.grab_set()

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="LocalL 검색 팝업")
    ap.add_argument("--db", default=str(config.DEFAULT_DB))
    ap.add_argument("--hidden", action="store_true", help="시작할 때 팝업을 띄우지 않음")
    args = ap.parse_args()

    os.makedirs(config.BASE / "data", exist_ok=True)
    logging.basicConfig(filename=str(config.BASE / "data" / "app.log"), level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    mutex = kernel32.CreateMutexW(None, False, "LocalL_single_instance")
    if ctypes.get_last_error() == 183:  # 이미 실행 중
        user32.MessageBoxW(None, "LocalL이 이미 실행 중입니다.\n단축키를 눌러 검색창을 여세요.", "LocalL", 0x40)
        return
    try:
        App(config.load(), args.db, show_on_start=not args.hidden).run()
    except Exception:
        log.exception("앱이 비정상 종료됨")
        raise
    finally:
        kernel32.CloseHandle(mutex)


if __name__ == "__main__":
    main()
