@echo off
REM Train and evaluate on the ORGAN-DISJOINT NuInsSeg folds (dataset_organ), all 5 folds.
REM Usage:  run_organ.bat emcaf        EMCAF-Net     -> trained_models\emcaf_organ\emcaf_organ_fold<N>_lr3.pth
REM         run_organ.bat dual         Dual decoder U-Net -> trained_models\dual_organ\dual_organ_fold<N>_lr3.pth
REM Same train.py / train_dual.py / test_instance.py, same settings (lr 1e-4, seed 12393) as the official-split runs;
REM only --dataset_path changes. Folds that already have a checkpoint are skipped, so it can be restarted.
REM Optional 2nd argument = full path of dataset_organ, e.g. run_organ.bat emcaf "E:\...\set_dataset\dataset_organ"
REM Run from the project folder (where train.py, test_instance.py and the folder dataset_organ are).

set WHICH=%1
if /i "%WHICH%"=="emcaf" goto emcaf
if /i "%WHICH%"=="dual" goto dual
echo Usage: run_organ.bat emcaf   or   run_organ.bat dual
exit /b 1

:emcaf
set MODEL_NAME=emcaf_organ
set TRAIN=train.py
set ARCH=nucleinet
goto start

:dual
set MODEL_NAME=dual_organ
set TRAIN=train_dual.py
set ARCH=dual_decoder_unet

:start
set DATA=%~2
if "%DATA%"=="" set DATA=.\dataset_organ

for %%F in (1 2 3 4 5) do (
    if exist trained_models\%MODEL_NAME%\%MODEL_NAME%_fold%%F_lr3.pth (
        echo --- fold %%F already trained, skipping ---
    ) else (
        echo ============================================
        echo  Training fold %%F of 5 - %MODEL_NAME%
        echo ============================================
        python %TRAIN% --fold %%F --dataset_path %DATA% --train_split train_aug --val_split val --arch %ARCH% --model_name %MODEL_NAME%
        if errorlevel 1 (
            echo ERROR training fold %%F. Stopping.
            exit /b 1
        )
    )
)

for %%F in (1 2 3 4 5) do (
    echo --- Evaluating %MODEL_NAME% fold %%F ---
    python test_instance.py --fold %%F --dataset_path %DATA% --split test --arch %ARCH% --model_path trained_models\%MODEL_NAME%\%MODEL_NAME%_fold%%F_lr3.pth
    if errorlevel 1 (
        echo ERROR evaluating fold %%F. Stopping.
        exit /b 1
    )
)
echo.
echo Done: %MODEL_NAME%. Per-image scores are in per_image_scores\%MODEL_NAME%_fold^<N^>_lr3__test.csv
