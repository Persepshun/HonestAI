@echo off
:loop
echo Starting training script at %date% %time%
python train_model.py
python train_keywords.py
echo Training completed.

echo.
echo Waiting 5 seconds for the next run...

:: Initialize progress bar
set bar=

:: Create a progress bar that fills up over 5 seconds with "X" symbols
for /L %%i in (1,1,5) do (
    set "bar=%bar%X"
    <nul set /p =[%bar%] %%i%% of 5 seconds complete.
    timeout /t 1 >nul
    <nul set /p ="`r"
)

echo.
echo Restarting training...
goto loop
