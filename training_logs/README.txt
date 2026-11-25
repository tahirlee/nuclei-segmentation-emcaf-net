Organ-disjoint NuInsSeg runs (five folds): EMCAF-Net (emcaf_organ) and Dual decoder U-Net (dual_organ)

raw/console_organ_disjoint_full.txt   complete console output, unchanged
emcaf_organ_train.txt                 training log of EMCAF-Net, folds 1-5 (per-step and per-epoch losses)
emcaf_organ_test.txt                  test evaluation of EMCAF-Net, folds 1-5 (Dice, AJI, PQ)
dual_organ_train.txt                  training log of the Dual decoder U-Net, folds 1-5
dual_organ_test.txt                   test evaluation of the Dual decoder U-Net, folds 1-5
curves/<model>_fold<N>.csv            epoch, train_loss, val_loss, val_dice, lr
curves/<model>.png                    per-fold curves (dotted line = selected checkpoint)
curves/all_folds.png                  all folds of both models (dots = selected checkpoints)
curves/summary.csv                    epochs run, best epoch, best validation loss, stop reason per fold
fold_level_results.csv                fold means, image-level means, paired differences, corrected t-test (Nadeau & Bengio), Holm

Note: the first evaluation of emcaf_organ in the raw log reports Dice only, because the instance masks
had not yet been copied to test\foldN\GT_instance; the evaluation was repeated afterwards (emcaf_organ_test.txt).
Curves and summary were made with analysis/plot_training_log.py --overlay.
