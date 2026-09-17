@echo off
rem Transkript: cift tiklandiginda arayuzu kendi masaustu penceresinde acar.
rem
rem 'uv run' proje bagimliliklarini secilen eklerle esitler (senkronlar); bu
rem betik kendiliginden baska bir araç (ffmpeg, yt-dlp, WebView2 Runtime vb.)
rem kurmaz. Saglayici anahtarlari arayuze/loglara yazilmaz.
setlocal
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

where uv >nul 2>nul
if errorlevel 1 (
  echo Hata: 'uv' bulunamadi; once uv kurulmalidir.
  echo Kurulum icin: https://docs.astral.sh/uv/ ardindan bu dosyayi yeniden acin.
  pause
  exit /b 1
)

rem .env onceligi (Transkript.command ve transkript.py ile ayni):
rem depo koku .env -> ek bayrak gerekmez; CLI varsayilani cwd'deki .env'i okur
uv run --extra api --extra cli --extra desktop subtitle-flow desktop %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Uygulama %EXIT_CODE% cikis koduyla kapandi.
  pause
)
exit /b %EXIT_CODE%
