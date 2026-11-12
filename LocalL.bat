@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist "%~dp0python\pythonw.exe" (
    start "" "%~dp0python\pythonw.exe" -W ignore -m locall.app %*
    exit /b
)
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" -W ignore -m locall.app %*
    exit /b
)

start "" pythonw -W ignore -m locall.app %*