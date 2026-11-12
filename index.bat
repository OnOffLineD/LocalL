@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 함께 있는 파이썬 (없어도 된다)
set "PY="
if exist "%~dp0python\python.exe" set "PY=%~dp0python\python.exe"
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"

rem Rust 색인기
set "RUST="
if exist "%~dp0target\release\locall-index.exe" set "RUST=%~dp0target\release\locall-index.exe"
if not defined RUST if exist "%~dp0locall-index.exe" set "RUST=%~dp0locall-index.exe"

rem 1) Rust 색인기가 파일명·본문·PDF·사진을 모두 한다.
rem    파이썬이 있으면 Rust가 못 읽은 몇 안 되는 파일(TIFF 등)은 대기로 남겨 둔다.
if defined RUST (
    if defined PY (
        "%RUST%" --leave-unsupported %*
    ) else (
        "%RUST%" %*
    )
)

rem 2) 파이썬 색인기: Rust가 남긴 것만 (Rust 색인기가 없으면 전부)
if defined PY "%PY%" -W ignore -m locall.index %*
if not defined RUST if not defined PY python -W ignore -m locall.index %*
pause
