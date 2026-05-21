@echo off
set "BASE_DIR=%~dp0"
set "BASE_DIR=%BASE_DIR:~0,-1%"

echo ==============================================
echo 1-Click Start: Auto TTS ^& Lipsync Studio
echo ==============================================

echo.
echo [PORTABLE CHECK] Verification of bundled environment...
set "PYTHON_EXE=%BASE_DIR%\GPT-SoVITS-v2pro-20250604\runtime\python.exe"
set "PIP_EXE=%BASE_DIR%\GPT-SoVITS-v2pro-20250604\runtime\Scripts\pip.exe"

REM One-time setup check for fresh Windows/New PC
if not exist "%BASE_DIR%\GPT-SoVITS-v2pro-20250604\runtime\Lib\site-packages\insightface" (
    echo [!!!] FIRST RUN DETECTED or NEW COMPUTER DETECTED.
    echo Preparing portable environment. This only happens once...
    echo.
    "%PYTHON_EXE%" -m pip install insightface gradio "numpy<2" "opencv-python<4.9" "opencv-python-headless<4.9" librosa requests numba
    echo.
    echo [PORTABLE CHECK] Environment Ready! ✅
)

echo Booting Backend GPT-SoVITS API Server...
set "PATH=%BASE_DIR%\GPT-SoVITS-v2pro-20250604\runtime;%PATH%"
call "%BASE_DIR%\run_servers.bat"

echo.
echo Booting UI Server...
"%PYTHON_EXE%" "%BASE_DIR%\app.py"

pause
