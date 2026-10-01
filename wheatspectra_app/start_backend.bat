@echo off
echo ===================================================
echo Starting PhytoSpectra-VLM Backend (WheatSpectraChat)
echo ===================================================
cd /d "%~dp0backend"
if not exist ".env" copy /Y ".env.example" ".env" >nul
if defined PHYTO_PYTHON (
  set "PYTHON_EXE=%PHYTO_PYTHON%"
) else (
  set "PYTHON_EXE=E:\Habib\.ruddro\Scripts\python.exe"
)
set "PHYTOSPECTRA_ROOT=%~dp0.."
set "PHYTOSPECTRA_RUN_ID=stage42d_exact_v1"
set "PHYTOSPECTRA_SEED=2026"
set "PHYTOSPECTRA_FOLD=0"
set "PHYTOSPECTRA_RAW_DATA=%~dp0..\..\data\raw"
set "PYTHONPATH=%~dp0..\src;%~dp0backend"
if not exist "%PYTHON_EXE%" (
  echo Required project environment was not found: %PYTHON_EXE%
  exit /b 1
)
"%PYTHON_EXE%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
