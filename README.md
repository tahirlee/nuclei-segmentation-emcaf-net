# EMCAF-Net on NuInsSeg

Code for training and evaluating EMCAF-Net (PVTv2-B2 encoder with edge-aware multi-scale fusion (EMF) and
contextual attention fusion (CAF) in the decoder) for nuclei segmentation. 


## Repository structure

```
data.py                  data loading (online augmentation; masks are only resampled with nearest neighbour)
train.py                 training of EMCAF-Net and its variants (--arch selects the file in model/)
train_dual.py            training of the Dual decoder U-Net baseline (segmentation + distance-map loss)
test_instance.py         evaluation: Dice, AJI, PQ (fixed watershed), per-image CSVs, saved predictions
boundary_metrics.py      Boundary F1, surface Dice, HD95 and instance Boundary F1 from the saved predictions
run_variant.bat          train + evaluate one EMCAF-Net variant on all 5 folds
run_train_dual.bat       train the Dual decoder U-Net on all 5 folds
run_eval_dual.bat        evaluate the Dual decoder U-Net on all 5 folds
run_organ.bat            train + evaluate EMCAF-Net or the Dual decoder U-Net on the organ-disjoint folds
model/                   network definitions (see model/README.md)
data_prep/
  check_original_folds.py   checks the original folds (pairing, masks, leakage, fold coverage)
  check_dataset.py          checks a built dataset (pairing, masks, alignment, leakage)
  make_organ_folds.py       builds organ-disjoint folds from flat image / GT / instance folders
  check_organ_disjoint.py   verifies that folds are organ-disjoint
  build_dataset.py          builds train / val / test / train_aug (8 augmented versions per training image)
  add_instance_masks.py     copies the instance masks into test\foldN\GT_instance under the test image names
analysis/
  paired_stats.py           paired comparison of two runs (Nadeau-Bengio corrected t-test, Holm, bootstrap)
  organ_macro.py            image-level and organ-macro results for the organ-disjoint evaluation
  profile_all_variants.py   parameters, GMACs and FPS of all variants
```

## Installation

```
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version first. Download the ImageNet weights of PVTv2-B2
(`pvt_v2_b2.pth`) from the official PVTv2 repository (whai362/PVT) and place them in `model/`.

## Data

Download NuInsSeg from its official release (665 H&E images, 31 human and mouse organs, with binary,
instance and other masks). The data are not part of this repository.

### Official five-fold split

Place the five official folds as

```
dataset_original\Fold_1\train\images, GT
dataset_original\Fold_1\test\images, GT
...  Fold_5
```

and build the training data:

```
python data_prep\check_original_folds.py --root .\dataset_original --expected_total 665
python data_prep\build_dataset.py --src .\dataset_original --dst .\dataset
python data_prep\add_instance_masks.py --dataset_path .\dataset --instance_dir <folder with the 665 instance masks>
python data_prep\check_dataset.py --dataset_path .\dataset
```

Per fold this gives 478 training images (3,824 after augmentation), 54 validation images taken at random
from the training images of the fold (stratified by organ, seed 12393) and 133 test images. The test images
are copied unchanged and are never augmented. `dataset\split_manifest.csv` lists the split of every image.

### Organ-disjoint five-fold split

With all images, binary masks and instance masks in three flat folders:

```
python data_prep\make_organ_folds.py --images .\original_dataset\images --gt .\original_dataset\GT --inst .\original_dataset\inst --dst .\dataset_original_organ
python data_prep\check_organ_disjoint.py --root .\dataset_original_organ
python data_prep\build_dataset.py --src .\dataset_original_organ --dst .\dataset_organ --val_by_organ
python data_prep\add_instance_masks.py --dataset_path .\dataset_organ --instance_dir .\original_dataset\inst
python data_prep\check_organ_disjoint.py --manifest .\dataset_organ\split_manifest.csv
```

Whole organs are assigned to the test folds (seed 12393) and the validation set of each fold consists of whole
organs from the training organs (about 54 images), so no organ is shared between the training, validation and
test sets of a fold. `organ_folds.csv` lists the test fold of every organ.

## Training and evaluation

All runs use the same settings: input 512 x 512, AdamW (lr 1e-4, weight decay 1e-4), batch 4 with 2
accumulation steps, BCE + soft Dice loss, ReduceLROnPlateau (factor 0.5, patience 2), early stopping on the
validation loss (patience 5, at least 15 epochs, at most 100), gradient clipping at norm 1, seed 12393.
Evaluation uses threshold 0.5 and the marker-controlled watershed with min_distance 7 and min_size 10;
PQ uses IoU >= 0.5 with Hungarian matching.

```
REM EMCAF-Net, official split
run_variant.bat nucleinet nucleinet

REM ablations (examples)
run_variant.bat nucleinet_no_edge  no_edge
run_variant.bat nucleinet_aff_caf  aff_caf
run_variant.bat nucleinet          noaug  .\dataset train 0

REM Dual decoder U-Net baseline
run_train_dual.bat
run_eval_dual.bat

REM organ-disjoint split
run_organ.bat emcaf .\dataset_organ
run_organ.bat dual  .\dataset_organ
```

On Linux, run the Python commands inside the `.bat` files directly, for example

```
python train.py --fold 1 --arch nucleinet --model_name nucleinet
python test_instance.py --fold 1 --arch nucleinet --model_path trained_models/nucleinet/nucleinet_fold1_lr3.pth
```

`test_instance.py` prints Dice, AJI and PQ per fold, writes per-image scores to
`per_image_scores/<run>_fold<N>_lr3__test.csv` and saves the binary predictions to
`MatlabEvaluationTools/<run>_fold<N>_lr3/fold<N>/salmap/`.

## Analysis

```
REM paired comparison of two runs (per-fold and image-level)
python analysis\paired_stats.py --a nucleinet --b dual_decoder_unet --n_train 478 --n_test 133 --out stats_emcaf_vs_dual

REM image-level and organ-macro results on the organ-disjoint split
python analysis\organ_macro.py --runs emcaf_organ dual_organ

REM boundary metrics from the saved predictions, then paired statistics
python boundary_metrics.py --run nucleinet --dataset .\dataset
python boundary_metrics.py --run no_edge --dataset .\dataset
python analysis\paired_stats.py --a nucleinet --b no_edge --suffix __boundary --metrics bf1,ibf1,sdice,hd95 --n_train 478 --n_test 133 --out stats_boundary

REM parameters, GMACs and FPS
python analysis\profile_all_variants.py
```

## License

MIT (see `LICENSE`). NuInsSeg and the PVTv2 weights are distributed under their own licenses.
