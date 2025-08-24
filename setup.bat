@echo off
echo Setting up environment with uv...
uv init
uv sync
echo Environment ready! 
echo.
echo To activate environment manually: .venv\Scripts\activate.bat
cmd /k .venv\Scripts\activate.bat