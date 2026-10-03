@echo off
echo ============================================
echo   DiaTwin - Starting Full Stack
echo ============================================
echo.

echo [1/2] Starting Backend API (port 8000)...
start "DiaTwin Backend" cmd /c "cd /d %~dp0 && python -m uvicorn backend.app.main:app --reload --port 8000"

echo [2/2] Starting Frontend (port 5173)...
timeout /t 3 >nul
start "DiaTwin Frontend" cmd /c "cd /d %~dp0\frontend && npm run dev"

echo.
echo ============================================
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo   API Docs: http://localhost:8000/docs
echo ============================================
echo.
echo Both servers are running in separate windows.
echo Close them to stop the application.
pause
