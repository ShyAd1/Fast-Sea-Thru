@echo off
setlocal EnableExtensions
:: Build the CUDA illuminant DLL.
:: Run from the proyect\ directory (double-click or from a terminal). Use --pause to keep the window open.
:: Equivalent to:
::   nvcc -O3 --shared -arch=native -o .\illuminant_cuda.dll .\illuminant.cu -ccbin "<MSVC Hostx64\x64>"

set "OUTDIR=%~dp0"
set "SRC=%OUTDIR%illuminant.cu"
set "OUT=%OUTDIR%illuminant_cuda.dll"

:: MSVC host compiler directory (edit if your toolset version changes).
:: Can also be overridden with the environment variable CCBIN.
if not defined CCBIN set "CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64"

echo ============================================================
echo Building CUDA illuminant DLL
echo   Source : %SRC%
echo   Output : %OUT%
echo   ccbin  : %CCBIN%
echo ============================================================

if not exist "%SRC%" goto :nosrc

:: ---- locate nvcc (PATH first, then newest installed CUDA toolkit)
set "NVCC=nvcc"
where nvcc >nul 2>&1
if not errorlevel 1 goto :nvcc_ok
set "CUDA_ROOT=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA"
for /f "delims=" %%D in ('dir /b /ad /o-n "%CUDA_ROOT%\v*" 2^>nul') do (
    if exist "%CUDA_ROOT%\%%D\bin\nvcc.exe" if "%NVCC%"=="nvcc" set "NVCC=%CUDA_ROOT%\%%D\bin\nvcc.exe"
)
where "%NVCC%" >nul 2>&1
if errorlevel 1 goto :nonvcc
:nvcc_ok

:: ---- host compiler: explicit -ccbin if the directory exists, else try vcvars64
set "CCBIN_ARG="
if exist "%CCBIN%\cl.exe" (
    set CCBIN_ARG=-ccbin "%CCBIN%"
    goto :compile
)
echo  [warn] cl.exe not found in CCBIN, trying to load the MSVC environment...
where cl.exe >nul 2>&1
if not errorlevel 1 goto :compile
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :nocl
for /f "usebackq delims=" %%I in (`""%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath"`) do set "VSINSTALL=%%I"
if not defined VSINSTALL goto :nocl
if not exist "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" goto :nocl
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
where cl.exe >nul 2>&1
if errorlevel 1 goto :nocl

:compile
echo.
echo  Compiling...
"%NVCC%" -O3 --shared -arch=native -o "%OUT%" "%SRC%" %CCBIN_ARG%
if errorlevel 1 goto :failed

echo.
echo  SUCCESS  ^>  %OUT%
echo  The DLL will be loaded automatically by proyect.py (same folder).
goto :end

:nosrc
echo.
echo  ERROR: %SRC% not found.
set "RC=1"
goto :fail_end

:nonvcc
echo.
echo  ERROR: nvcc.exe not found. Install the CUDA Toolkit or add nvcc to PATH.
set "RC=1"
goto :fail_end

:nocl
echo.
echo  ERROR: MSVC cl.exe not found. Fix the CCBIN path at the top of build.bat,
echo  install "Build Tools for Visual Studio" (Desktop development with C++),
echo  or run this from the x64 Native Tools Command Prompt.
set "RC=1"
goto :fail_end

:failed
set "RC=%ERRORLEVEL%"
echo.
echo  FAILED - nvcc returned error %RC%
echo  nvcc used: %NVCC%

:fail_end
if /I "%~1"=="--pause" pause
exit /b %RC%

:end
if /I "%~1"=="--pause" pause
exit /b 0
