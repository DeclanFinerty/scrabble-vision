@echo off
REM Start Scrabble Vision mobile testing server
REM Opens two terminals: uvicorn backend + ngrok HTTPS tunnel
REM Open the ngrok URL on your phone to test with camera

echo Starting Scrabble Vision mobile server...
echo.
echo Terminal 1: uvicorn backend on 0.0.0.0:8000
echo Terminal 2: ngrok HTTPS tunnel
echo.
echo Look for the ngrok "Forwarding" URL and open it on your phone.
echo Press Ctrl+C in each terminal window to stop.
echo.

start "Scrabble Vision - Backend" cmd /k "cd /d %~dp0 && uv run uvicorn server:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 2 /nobreak >nul
start "Scrabble Vision - ngrok" cmd /k "ngrok http 8000"