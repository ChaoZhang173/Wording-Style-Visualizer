@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>nul
  if not errorlevel 1 (
    ".venv\Scripts\python.exe" scripts\bootstrap.py %*
    goto :finished
  )
)

for %%V in (3.12 3.11 3.10) do (
  py -%%V -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>nul
  if not errorlevel 1 (
    py -%%V scripts\bootstrap.py %*
    goto :finished
  )
)

python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>nul
if not errorlevel 1 (
  python scripts\bootstrap.py %*
  goto :finished
)

echo Python 3.10, 3.11, or 3.12 is required.
echo Install Python 3.12 from https://www.python.org/downloads/
echo Select "Add python.exe to PATH" during installation, then try again.
pause
exit /b 1

:finished
if errorlevel 1 (
  echo.
  echo Token Atlas could not start. See the message above and README.md.
  pause
  exit /b 1
)
exit /b 0
