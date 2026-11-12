@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist "%~dp0python\python.exe" (
    "%~dp0python\python.exe" -W ignore -m locall.index %*
    pause
    exit /b
)
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -W ignore -m locall.index %*
    pause
    exit /b
)

python -W ignore -m locall.index %*
pause