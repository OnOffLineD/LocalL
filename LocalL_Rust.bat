@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem 배포본은 locall.exe가 이 폴더에 있고, 직접 빌드했으면 target\release에 있다
if exist "%~dp0locall.exe" (
    start "" "%~dp0locall.exe"
) else (
    start "" target\release\locall.exe
)
