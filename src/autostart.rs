//! PC를 켤 때 LocalL을 자동으로 시작하기.
//!
//! `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`의 `LocalL` 값에
//! `"<locall.exe 경로>" --autostart`를 적는다. 관리자 권한이 필요 없다.
//! 켜짐/꺼짐은 이 레지스트리 값으로 판단한다 (config.json에는 적지 않는다).
//! 작업 관리자의 "시작 프로그램"에서 끈 경우(StartupApproved)도 꺼진 것으로 본다.

/// 자동 시작으로 켜졌을 때 붙는 인자. 이때는 창을 띄우지 않고 단축키만 기다린다.
pub const ARG: &str = "--autostart";

#[cfg(windows)]
mod imp {
    use super::ARG;
    use anyhow::{bail, Result};
    use windows::core::{w, PCWSTR};
    use windows::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_SUCCESS};
    use windows::Win32::System::Registry::{
        RegCloseKey, RegDeleteValueW, RegOpenKeyExW, RegQueryValueExW, RegSetValueExW, HKEY, HKEY_CURRENT_USER,
        KEY_READ, KEY_SET_VALUE, REG_SAM_FLAGS, REG_SZ,
    };

    const RUN_KEY: PCWSTR = w!(r"Software\Microsoft\Windows\CurrentVersion\Run");
    const APPROVED_KEY: PCWSTR = w!(r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run");
    const VALUE: PCWSTR = w!("LocalL");

    struct Key(HKEY);

    impl Drop for Key {
        fn drop(&mut self) {
            unsafe {
                let _ = RegCloseKey(self.0);
            }
        }
    }

    fn open(path: PCWSTR, access: REG_SAM_FLAGS) -> Option<Key> {
        let mut h = HKEY::default();
        let err = unsafe { RegOpenKeyExW(HKEY_CURRENT_USER, path, 0, access, &mut h) };
        (err == ERROR_SUCCESS).then(|| Key(h))
    }

    /// 값의 날 바이트. 없으면 None.
    fn read_bytes(key: &Key) -> Option<Vec<u8>> {
        let mut size = 0u32;
        let err = unsafe { RegQueryValueExW(key.0, VALUE, None, None, None, Some(&mut size)) };
        if err != ERROR_SUCCESS {
            return None;
        }
        let mut buf = vec![0u8; size as usize];
        let err = unsafe { RegQueryValueExW(key.0, VALUE, None, None, Some(buf.as_mut_ptr()), Some(&mut size)) };
        if err != ERROR_SUCCESS {
            return None;
        }
        buf.truncate(size as usize);
        Some(buf)
    }

    fn read_command() -> Option<String> {
        let bytes = read_bytes(&open(RUN_KEY, KEY_READ)?)?;
        let units: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let len = units.iter().position(|&c| c == 0).unwrap_or(units.len());
        Some(String::from_utf16_lossy(&units[..len]))
    }

    /// 작업 관리자에서 "사용 안 함"으로 바꿨는지 (첫 바이트가 홀수면 꺼짐)
    fn disabled_in_task_manager() -> bool {
        open(APPROVED_KEY, KEY_READ)
            .and_then(|k| read_bytes(&k))
            .and_then(|b| b.first().copied())
            .map_or(false, |b| b & 1 == 1)
    }

    fn command() -> Result<String> {
        let exe = std::env::current_exe()?;
        Ok(format!("\"{}\" {}", exe.display(), ARG))
    }

    pub fn is_enabled() -> bool {
        read_command().is_some() && !disabled_in_task_manager()
    }

    fn run_key() -> Result<Key> {
        match open(RUN_KEY, KEY_READ | KEY_SET_VALUE) {
            Some(k) => Ok(k),
            None => bail!("시작 프로그램 레지스트리를 열 수 없습니다"),
        }
    }

    fn write_command(key: &Key) -> Result<()> {
        let cmd: Vec<u16> = command()?.encode_utf16().chain(Some(0)).collect();
        let bytes = unsafe { std::slice::from_raw_parts(cmd.as_ptr() as *const u8, cmd.len() * 2) };
        let err = unsafe { RegSetValueExW(key.0, VALUE, 0, REG_SZ, Some(bytes)) };
        if err != ERROR_SUCCESS {
            bail!("시작 프로그램에 등록하지 못했습니다 (오류 {})", err.0);
        }
        Ok(())
    }

    pub fn set_enabled(enabled: bool) -> Result<()> {
        let key = run_key()?;
        if enabled {
            write_command(&key)?;
            // 작업 관리자에서 꺼 둔 기록이 있으면 지워서 다시 켜지게 한다
            if let Some(approved) = open(APPROVED_KEY, KEY_SET_VALUE) {
                unsafe {
                    let _ = RegDeleteValueW(approved.0, VALUE);
                }
            }
        } else {
            let err = unsafe { RegDeleteValueW(key.0, VALUE) };
            if err != ERROR_SUCCESS && err != ERROR_FILE_NOT_FOUND {
                bail!("시작 프로그램에서 빼지 못했습니다 (오류 {})", err.0);
            }
        }
        Ok(())
    }

    /// 켜져 있는데 폴더를 옮겨 경로가 달라졌으면 지금 경로로 다시 적는다.
    pub fn refresh_path() {
        if let (Some(saved), Ok(now)) = (read_command(), command()) {
            if !saved.eq_ignore_ascii_case(&now) {
                // 작업 관리자에서 꺼 둔 상태는 건드리지 않고 경로만 고친다
                if let Ok(key) = run_key() {
                    let _ = write_command(&key);
                }
            }
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use anyhow::{bail, Result};

    pub fn is_enabled() -> bool {
        false
    }

    pub fn set_enabled(_enabled: bool) -> Result<()> {
        bail!("자동 시작은 Windows에서만 지원합니다")
    }

    pub fn refresh_path() {}
}

pub use imp::{is_enabled, refresh_path, set_enabled};
