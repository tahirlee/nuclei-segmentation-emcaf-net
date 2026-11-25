@echo off
REM Train and evaluate one EMCAF-Net variant on all 5 NuInsSeg folds (official split by default).
REM Usage:  run_variant.bat ARCH MODEL_NAME [DATASET] [TRAIN_SPLIT] [ONLINE_AUG]
REM   ARCH        model file in model\ without .py, e.g. nucleinet, nucleinet_no_edge, nucleinet_aff_caf
REM   MODEL_NAME  name of the run; checkpoints go to trained_models\MODEL_NAME\MODEL_NAME_fold<N>_lr3.pth
REM   DATASET     default .\dataset
REM   TRAIN_SPLIT default train_aug (offline-augmented); use "train" for the no-augmentation ablation
REM   ONLINE_AUG  default 1; use 0 for the no-augmentation ablation
REM Examples:
REM   run_variant.bat nucleinet nucleinet                         full model (EMCAF-Net)
REM   run_variant.bat nucleinet_no_edge no_edge                   ablation: without edge gate
REM   run_variant.bat nucleinet noaug .\dataset train 0           ablation: without augmentation
REM Folds that already have a checkpoint are not retrained (train.py never overwrites a checkpoint).

set ARCH=%1
set MODEL_NAME=%2
if "%ARCH%"=="" goto usage
if "%MODEL_NAME%"=="" goto usage
set DATA=%~3
if "%DATA%"=="" set DATA=.\dataset
set SPLIT=%4
if "%SPLIT%"=="" set SPLIT=train_aug
set OAUG=%5
if "%OAUG%"=="" set OAUG=1

for %%F in (1 2 3 4 5) do (
    if exist trained_models\%MODEL_NAME%\%MODEL_NAME%_fold%%F_lr3.pth (
        echo --- fold %%F already trained, skipping ---
    ) else (
        echo ============================================
        echo  Training fold %%F of 5 - %MODEL_NAME% ^(%ARCH%^)
        echo ============================================
        python train.py --fold %%F --dataset_path %DATA% --train_split %SPLIT% --online_aug %OAUG% --val_split val --arch %ARCH% --model_name %MODEL_NAME%
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
echo Done: %MODEL_NAME%. Per-image scores: per_image_scores\%MODEL_NAME%_fold^<N^>_lr3__test.csv
exit /b 0

:usage
echo Usage: run_variant.bat ARCH MODEL_NAME [DATASET] [TRAIN_SPLIT] [ONLINE_AUG]
exit /b 1
