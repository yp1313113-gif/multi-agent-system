@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ============================================================
echo  加计扣除测算 + 人工审核闭环 演示
echo ============================================================
echo.
.venv\Scripts\python.exe demo_tax_review.py
echo.
echo 演示结束，按任意键关闭。
pause >nul