@echo off
echo Starting GPT-SoVITS-v2pro API Server...
set "GPT_DIR=%~dp0GPT-SoVITS-v2pro-20250604"
cd /d "%GPT_DIR%"
set "PATH=%GPT_DIR%\runtime;%PATH%"
start cmd /k "runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880"
echo Server started!
