r"""
boundary_metrics.py - boundary-sensitive scores from the predicted masks that test_instance.py already saved.
No model is run: it reads  MatlabEvaluationTools\<run>_fold<k>_lr3\fold<k>\salmap\<image>  (binary prediction, 0/255)
and compares it with  <dataset>\test\fold<k>\GT  and  GT_instance.

Per image (tolerance --tol pixels, default 2):
  bf1      Boundary F1 of the binary mask (contour of predicted foreground vs contour of GT foreground)
  sdice    surface Dice at the same tolerance
  hd95     95th-percentile symmetric Hausdorff distance between the two contours (pixels; LOWER is better)
  ibf1     instance Boundary F1: contours of the INSTANCES, i.e. after the fixed watershed (min_distance 7,
           min_size 10) on the prediction and from GT_instance - this includes the split lines between touching
           nuclei, so it measures the separation that AJI/PQ depend on

Output: per_image_scores\<run>_fold<k>_lr3__test__boundary.csv   (columns image,bf1,sdice,hd95,ibf1)
        -> paired statistics with the existing script:
           python analysis\paired_stats.py --a nucleinet --b nucleinet_no_emf --suffix __boundary ^
                  --metrics bf1,ibf1,sdice,hd95 --n_train 478 --n_test 133 --out stats_boundary_emf

    python boundary_metrics.py --run nucleinet --folds 1 2 3 4 5 --dataset .\dataset
    python boundary_metrics.py --run nucleinet_no_emf --folds 1 2 3 4 5 --dataset .\dataset
"""
import argparse
import csv
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from test_instance import binary_to_instances, load_instance_gt


def contour(mask_or_labels):
    return find_boundaries(mask_or_labels, mode='inner')


def boundary_scores(pb, gb, tol):
    """pb, gb: boolean contour maps. Returns precision, recall, F1, surface Dice, HD95."""
    npb, ngb = int(pb.sum()), int(gb.sum())
    if npb == 0 and ngb == 0:
        return 1.0, 1.0, 1.0, 1.0, 0.0
    if npb == 0 or ngb == 0:
        return 0.0, 0.0, 0.0, 0.0, float(np.hypot(*pb.shape))
    d_to_g = ndi.distance_transform_edt(~gb)      # distance of every pixel to the nearest GT contour pixel
    d_to_p = ndi.distance_transform_edt(~pb)
    dp, dg = d_to_g[pb], d_to_p[gb]                # pred contour -> GT, GT contour -> pred
    prec = float((dp <= tol).mean())
    rec = float((dg <= tol).mean())
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    sdice = float(((dp <= tol).sum() + (dg <= tol).sum()) / (npb + ngb))
    hd95 = float(np.percentile(np.concatenate([dp, dg]), 95))
    return prec, rec, f1, sdice, hd95


def find_file(folder, stem, exts):
    for e in exts:
        p = os.path.join(folder, stem + e)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='checkpoint prefix, e.g. nucleinet or nucleinet_no_emf')
    ap.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--dataset', default='./dataset')
    ap.add_argument('--split', default='test')
    ap.add_argument('--pred_root', default='MatlabEvaluationTools')
    ap.add_argument('--run_pattern', default='{run}_fold{k}_lr3', help='name of the folder test_instance.py made')
    ap.add_argument('--tol', type=float, default=2.0)
    ap.add_argument('--min_distance', type=int, default=7)
    ap.add_argument('--min_size', type=int, default=10)
    ap.add_argument('--out_dir', default='per_image_scores')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    for k in a.folds:
        name = a.run_pattern.format(run=a.run, k=k)
        pred_dir = os.path.join(a.pred_root, name, f'fold{k}', 'salmap')
        root = os.path.join(a.dataset, a.split, f'fold{k}')
        gt_dir, inst_dir = os.path.join(root, 'GT'), os.path.join(root, 'GT_instance')
        if not os.path.isdir(pred_dir):
            raise SystemExit(f'No predictions in {pred_dir}. Run test_instance.py for this checkpoint first.')
        rows = []
        for f in sorted(os.listdir(pred_dir)):
            stem = os.path.splitext(f)[0]
            pred = np.asarray(Image.open(os.path.join(pred_dir, f)).convert('L')) > 127
            gpath = find_file(gt_dir, stem, ('.png', '.tif', '.bmp', '.jpg'))
            if gpath is None:
                continue
            gt = np.asarray(Image.open(gpath).convert('L')) > 0
            if gt.shape != pred.shape:
                raise SystemExit(f'size mismatch for {stem}: prediction {pred.shape}, GT {gt.shape}')
            _, _, bf1, sdice, hd95 = boundary_scores(contour(pred), contour(gt), a.tol)
            ibf1 = float('nan')
            ipath = find_file(inst_dir, stem, ('.tif', '.tiff')) if os.path.isdir(inst_dir) else None
            if ipath:
                pi = binary_to_instances(pred.astype(np.uint8), a.min_distance, a.min_size)
                gi = load_instance_gt(ipath)
                ibf1 = boundary_scores(contour(pi), contour(gi), a.tol)[2]
            rows.append((stem, bf1, sdice, hd95, ibf1))
        out = os.path.join(a.out_dir, f'{name}__{a.split}__boundary.csv')
        with open(out, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['image', 'bf1', 'sdice', 'hd95', 'ibf1'])
            w.writerows(rows)
        arr = np.array([r[1:] for r in rows], float)
        m = np.nanmean(arr, axis=0)
        print(f'{name}: {len(rows)} images  BF1 {m[0]:.4f}  surface Dice {m[1]:.4f}  HD95 {m[2]:.2f} px  '
              f'instance BF1 {m[3]:.4f}  -> {out}')


if __name__ == '__main__':
    main()
