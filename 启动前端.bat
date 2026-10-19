@echo off
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

netstat -ano | findstr ":8002 " | findstr LISTENING >nul
if %errorlevel%==0 (
    echo 服务已经在运行，直接打开页面...
    start "" http://127.0.0.1:8002/
    timeout /t 2 >nul
    exit /b
)

set "PORT=8002"
echo ============================================================
echo  研发费用智能体
echo  正在启动，约 30-70 秒预热，预热完浏览器会自动打开
echo  地址： http://127.0.0.1:8002/
echo  关闭本窗口即停止服务
echo ============================================================
echo.
start /b cmd /c "timeout /t 40 >nul & start "" http://127.0.0.1:8002/"
.venv\Scripts\python.exe api.py
echo.
echo 服务已停止。按任意键关闭。
pause >nul