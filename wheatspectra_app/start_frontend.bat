@echo off
echo ===================================================
echo Starting WheatSpectraChat Frontend
echo ===================================================
cd /d "%~dp0frontend"
if not exist "node_modules" (
  echo Frontend dependencies are missing. Run npm ci in this frontend folder first.
  exit /b 1
)
npm run dev
pause
