@echo off
REM Set environment variable for TensorFlow optimizations
set TF_ENABLE_ONEDNN_OPTS=0

REM Activate the Python virtual environment
call venv\Scripts\activate

REM Run the Flask application
python app.py

pause

REM Deactivate the virtual environment after exiting
deactivate
