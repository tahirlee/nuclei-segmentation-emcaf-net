# EMCAF-Net

Official implementation of **EMCAF-Net**, *******************************************.

The model predicts a binary nucleus mask. Individual nucleus instances are recovered
from that mask by marker-controlled watershed post-processing.


## Installation

```bash
git clone https://github.com/<user>/EMCAF-Net.git
cd EMCAF-Net
pip install -r requirements.txt
```

Install PyTorch for your own CUDA version first if the default wheel does not match
your driver: <https://pytorch.org/get-started/locally/>

## Pretrained encoder weights

Download `pvt_v2_b2.pth` from the
[official PVTv2 repository](https://github.com/whai362/PVT/tree/v2/classification)
and place it at `model/pvt_v2_b2.pth`. Without it the encoder starts from random
weights, and the model prints a warning at start-up saying so.

## Data layout

`train.py` and `test.py` expect this layout, and nothing else:

```
dataset/
├── train_aug/foldN/{images,GT}/          offline-augmented training images
├── train/foldN/{images,GT}/              the same images, not augmented
├── val/foldN/{images,GT}/                validation images, never augmented
└── test/foldN/{images,GT[,GT_instance]}/ test images
```

`images` holds the RGB patches and `GT` the binary masks; any non-zero pixel is
foreground, so both 0/1 and 0/255 masks work. Each image is paired with the mask that
has the same file name.

`GT_instance` holds integer instance-label masks, one per test image, with the same
file name and a `.tif` extension. It is needed only for AJI and PQ; without it
`test.py` reports Dice and warns. Most datasets distribute these instance
annotations separately from the binary masks — copy them into
`dataset/test/foldN/GT_instance/` and rename them to match the test image names.


## Training

```bash
python train.py --fold 1 --dataset_path ./dataset --model_name nucleinet
```

Repeat for each fold. Checkpoints are written to
`trained_models/<model_name>/<model_name>_fold<N>_lr3.pth`, with the settings of the
run beside them.


## Evaluation

```bash
python test.py --fold 1 --dataset_path ./dataset \
    --model_path trained_models/nucleinet/nucleinet_fold1_lr3.pth
```

Reports Dice, AJI and PQ, writes the thresholded predictions to
`predictions/<run>/fold<N>/`, and per-image scores to
`per_image_scores/<run>__test.csv`.




## Citation

```bibtex
@article{emcafnet,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO},
  doi     = {TODO}
}
```

## License

MIT, see [LICENSE](LICENSE). `model/pvtv2.py` is adapted from the official PVTv2
implementation (Apache-2.0).
