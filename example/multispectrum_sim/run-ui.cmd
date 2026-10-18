@echo off
setlocal
cd /d "%~dp0"
set "YK_EXAMPLE_DATA_ROOT=%~dp0..\..\data"
set "PATH=%~dp0out\bin;%PATH%"
if not exist "%~dp0out\bin\multispectrum_sim.exe" (
    echo UI executable not found: out\bin\multispectrum_sim.exe
    pause
    exit /b 1
)
"%~dp0out\bin\multispectrum_sim.exe" --gui %*
if errorlevel 1 pause
