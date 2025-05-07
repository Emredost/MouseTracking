@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Setting environment variables...
set MOUSE_TRACKER_DATA_DIR=%CD%\mouse_data
set MOUSE_TRACKER_DEBUG=false

echo.
echo Environment setup complete!
echo Virtual environment has been created and dependencies installed.
echo.
echo Environment variables set:
echo MOUSE_TRACKER_DATA_DIR: %MOUSE_TRACKER_DATA_DIR%
echo MOUSE_TRACKER_DEBUG: %MOUSE_TRACKER_DEBUG%
echo.
echo To activate this environment in the future, run:
echo venv\Scripts\activate.bat
echo.
echo To run the mouse tracker GUI, run:
echo python mouse_tracker_gui.py 