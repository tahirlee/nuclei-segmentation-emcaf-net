@echo off
REM Evaluate the dual-decoder U-Net on all 5 NuInsSeg test folds (same test_instance.py and watershed as EMCAF-Net).
REM Usage: run_eval_dual.bat                 (evaluates dual_decoder_unet)
REM        run_eval_dual.bat my_other_run    (another run name)

set MODEL_NAME=%1
if "%MODEL_NAME%"=="" set MODEL_NAME=dual_decoder_unet

for %%F in (1 2 3 4 5) do (
    echo --- Evaluating %MODEL_NAME% fold %%F ---
    python test_instance.py --fold %%F --dataset_path .\dataset --split test --arch dual_decoder_unet --model_path trained_models\%MODEL_NAME%\%MODEL_NAME%_fold%%F_lr3.pth
    if errorlevel 1 (
        echo ERROR on fold %%F. Stopping.
        exit /b 1
    )
)
echo.
echo All folds evaluated for %MODEL_NAME%.
