@echo off
REM Train the dual-decoder U-Net baseline on all 5 NuInsSeg folds (same data and protocol as EMCAF-Net).
REM Usage: run_train_dual.bat            (learning rate 1e-4, same as EMCAF-Net)
REM        run_train_dual.bat 1e-3       (another learning rate, e.g. chosen in the fold-1 pilot)
REM Output: trained_models\dual_decoder_unet\dual_decoder_unet_fold<N>_lr3.pth
REM Folds that already have a checkpoint are skipped.

set MODEL_NAME=dual_decoder_unet
set LR=%1
if "%LR%"=="" set LR=1e-4

for %%F in (1 2 3 4 5) do (
    if exist trained_models\%MODEL_NAME%\%MODEL_NAME%_fold%%F_lr3.pth (
        echo --- fold %%F already trained, skipping ---
    ) else (
        echo ============================================
        echo  Training fold %%F of 5 - %MODEL_NAME% - lr %LR%
        echo ============================================
        python train_dual.py --fold %%F --lr %LR% --dataset_path .\dataset --train_split train_aug --val_split val --arch dual_decoder_unet --model_name %MODEL_NAME%
        if errorlevel 1 (
            echo ERROR on fold %%F. Stopping.
            exit /b 1
        )
    )
)
echo All folds done. Checkpoints in trained_models\%MODEL_NAME%\
