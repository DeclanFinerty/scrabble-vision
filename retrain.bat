@echo off
echo ============================================
echo  Scrabble Vision - Full Retrain Pipeline
echo ============================================

echo.
echo === Phase 1: Extract real tiles ===
uv run extract_tiles.py
if errorlevel 1 goto :error

echo.
echo === Phase 2: Augment tiles ===
uv run augment_tiles.py
if errorlevel 1 goto :error

echo.
echo === Phase 3: Generate synthetic tiles ===
uv run generate_tiles.py
if errorlevel 1 goto :error

echo.
echo === Phase 4: Train classifier ===
uv run train.py
if errorlevel 1 goto :error

echo.
echo === Phase 5: Evaluate ===
uv run evaluate.py --no-interactive
if errorlevel 1 goto :error

echo.
echo ============================================
echo  Done! Check results above.
echo ============================================
goto :end

:error
echo.
echo ERROR: Pipeline failed at the step above.
pause

:end
