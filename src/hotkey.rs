use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver};
use std::sync::OnceLock;
use std::thread;

pub const MOD_ALT: u32 = 0x1;
pub const MOD_CTRL: u32 = 0x2;
pub const MOD_SHIFT: u32 = 0x4;
pub const MOD_WIN: u32 = 0x8;
pub const MOD_NOREPEAT: u32 = 0x4000;

pub fn key_map() -> &'static HashMap<&'static str, u32> {
    static MAP: OnceLock<HashMap<&'static str, u32>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("space", 0x20);
        m.insert("enter", 0x0D);
        m.insert("tab", 0x09);
        m.insert("backspace", 0x08);
        m.insert("insert", 0x2D);
        m.insert("delete", 0x2E);
        m.insert("home", 0x24);
        m.insert("end", 0x23);
        m.insert("pageup", 0x21);
        m.insert("pagedown", 0x22);
        m.insert("up", 0x26);
        m.insert("down", 0x28);
        m.insert("left", 0x25);
        m.insert("right", 0x27);
        m.insert("`", 0xC0);
        m.insert("-", 0xBD);
        m.insert("=", 0xBB);
        m.insert("[", 0xDB);
        m.insert("]", 0xDD);
        m.insert("\\", 0xDC);
        m.insert(";", 0xBA);
        m.insert("'", 0xDE);
        m.insert(",", 0xBC);
        m.insert(".", 0xBE);
        m.insert("/", 0xBF);

        // a-z
        for c in b'a'..=b'z' {
            let s: &'static str = Box::leak(String::from(c as char).into_boxed_str());
            m.insert(s, (c - b'a' + 0x41) as u32);
        }
        // 0-9
        for c in b'0'..=b'9' {
            let s: &'static str = Box::leak(String::from(c as char).into_boxed_str());
            m.insert(s, (c - b'0' + 0x30) as u32);
        }
        // f1-f24
        for i in 1..=24 {
            let s: &'static str = Box::leak(format!("f{}", i).into_boxed_str());
            m.insert(s, 0x6F + i);
        }
        m
    })
}

pub fn parse_hotkey(text: &str) -> Result<(u32, u32)> {
    let parts: Vec<&str> = text
        .split('+')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if parts.is_empty() {
        return Err(anyhow!("단축키가 비어 있습니다"));
    }

    let last = parts.last().unwrap().to_lowercase();
    let vk = key_map()
        .get(last.as_str())
        .copied()
        .ok_or_else(|| anyhow!("알 수 없는 키: {}", last))?;

    let mut mods = 0u32;
    for &p in &parts[..parts.len() - 1] {
        match p.to_lowercase().as_str() {
            "ctrl" => mods |= MOD_CTRL,
            "alt" => mods |= MOD_ALT,
            "shift" => mods |= MOD_SHIFT,
            "win" => mods |= MOD_WIN,
            _ => return Err(anyhow!("알 수 없는 조합 키: {}", p)),
        }
    }

    Ok((mods, vk))
}

#[cfg(windows)]
pub struct HotkeyManager {
    thread_id: u32,
    _worker: thread::JoinHandle<()>,
    event_rx: Receiver<()>,
}

#[cfg(windows)]
impl HotkeyManager {
    pub fn new(hotkey_text: &str) -> Result<Self> {
        let (mods, vk) = parse_hotkey(hotkey_text)?;
        let (tx, rx) = channel();
        let (init_tx, init_rx) = channel::<Result<u32, String>>();
        let hotkey_name = hotkey_text.to_string();

        let worker = thread::spawn(move || {
            use windows::Win32::System::Threading::GetCurrentThreadId;
            use windows::Win32::UI::Input::KeyboardAndMouse::{RegisterHotKey, UnregisterHotKey, HOT_KEY_MODIFIERS};
            use windows::Win32::UI::WindowsAndMessaging::{GetMessageW, MSG, WM_HOTKEY};

            unsafe {
                let tid = GetCurrentThreadId();
                let success = RegisterHotKey(
                    None,
                    1,
                    HOT_KEY_MODIFIERS(mods | MOD_NOREPEAT),
                    vk,
                );

                if success.is_err() {
                    let _ = init_tx.send(Err(format!("'{}' 등록 실패: 다른 프로그램이 이미 사용 중입니다.", hotkey_name)));
                    return;
                }

                let _ = init_tx.send(Ok(tid));

                let mut msg = MSG::default();
                while GetMessageW(&mut msg, None, 0, 0).as_bool() {
                    if msg.message == WM_HOTKEY {
                        let _ = tx.send(());
                    }
                }

                let _ = UnregisterHotKey(None, 1);
            }
        });

        match init_rx.recv() {
            Ok(Ok(tid)) => Ok(Self {
                thread_id: tid,
                _worker: worker,
                event_rx: rx,
            }),
            Ok(Err(e)) => Err(anyhow!(e)),
            Err(_) => Err(anyhow!("단축키 스레드 초기화 실패")),
        }
    }

    pub fn try_recv(&self) -> bool {
        self.event_rx.try_recv().is_ok()
    }
}

#[cfg(windows)]
impl Drop for HotkeyManager {
    fn drop(&mut self) {
        use windows::Win32::Foundation::{LPARAM, WPARAM};
        use windows::Win32::UI::WindowsAndMessaging::{PostThreadMessageW, WM_QUIT};
        unsafe {
            let _ = PostThreadMessageW(self.thread_id, WM_QUIT, WPARAM(0), LPARAM(0));
        }
    }
}
