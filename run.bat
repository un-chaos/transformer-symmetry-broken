@echo off
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 goto nopython
python run.py
echo.
pause
exit /b 0

:nopython
echo.
echo   找不到 python。请先安装 Python，安装时务必勾选 "Add python.exe to PATH"：
echo   https://www.python.org/downloads/
echo.
pause
exit /b 1
