"""
Evaluation script for EMCAF-Net — computes Dice, AJI, and PQ.

  Dice: pixel-level binary segmentation (no watershed)
  AJI:  instance-level, watershed applied to PREDICTION only
  PQ:   instance-level, SQ × DQ, IoU-threshold matching

Usage:
    python test.py --fold 1 \\
        --model_path ./trained_models/nucleinet/nucleinet_fold1_lr3.pth

    python test.py --fold 3 \\
        --model_path ./trained_models/nucleinet/nucleinet_fold3_lr3.pth \\
        --dataset_path ./dataset/cryonuseg

Dataset layout expected:
    <dataset_path>/test/fold<N>/images/
    <dataset_path>/test/fold<N>/GT/            (binary masks  -> Dice)
    <dataset_path>/test/fold<N>/GT_instance/   (label masks   -> AJI, PQ)

GT_instance/ is optional: without it the script reports Dice only and warns.
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)   # timm deprecations + autocast
warnings.filterwarnings('ignore', category=UserWarning)     # pvtv2 registry conflicts
import os
import argparse
import csv
import time

import imageio
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm import tqdm

import importlib

from data import test_dataset


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_dice(gt_binary, pred_binary):
    """Pixel-level Dice. No watershed."""
    gt   = (gt_binary   > 0).astype(np.float32)
    pred = (pred_binary > 0).astype(np.float32)
    return (2 * np.sum(gt * pred)) / (np.sum(gt) + np.sum(pred) + 1e-5)


def binary_to_instances(binary_mask, min_distance=7, min_size=10):
    """
    Marker-controlled watershed on a binary prediction map.
    Returns an integer instance label map (0 = background).
    Applied to PREDICTIONS only; real GT instance labels come from GT_instance/.
    """
    binary_mask = (binary_mask > 0)
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=np.int32)

    distance  = ndi.distance_transform_edt(binary_mask)
    coords    = peak_local_max(distance, min_distance=min_distance, labels=binary_mask)
    seed_mask = np.zeros(distance.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(seed_mask)
    labels = watershed(-distance, markers, mask=binary_mask)

    # remove tiny fragments
    for lbl in np.unique(labels):
        if lbl > 0 and np.sum(labels == lbl) < min_size:
            labels[labels == lbl] = 0

    return labels.astype(np.int32)


def load_instance_gt(path):
    """Load a real instance-labeled GT mask (NuInsSeg / CryoNuSeg label_mask)."""
    return tifffile.imread(path).astype(np.int32)


def compute_aji(gt, pred):
    """
    Aggregated Jaccard Index.
    Both gt and pred must be integer instance maps (0 = background).
    """
    gt_ids   = np.unique(gt);   gt_ids   = gt_ids[gt_ids   > 0]
    pred_ids = np.unique(pred); pred_ids = pred_ids[pred_ids > 0]

    correct_count = 0
    union_count   = 0
    used_pred     = set()

    for g in gt_ids:
        gt_mask    = (gt == g)
        candidates = np.unique(pred[gt_mask])
        candidates = candidates[candidates > 0]

        if len(candidates) == 0:
            union_count += np.count_nonzero(gt_mask)
            continue

        best_ji, best_p = 0, None
        for p in candidates:
            pm = (pred == p)
            ji = np.count_nonzero(gt_mask & pm) / np.count_nonzero(gt_mask | pm)
            if ji > best_ji:
                best_ji, best_p = ji, p

        pm = (pred == best_p)
        correct_count += np.count_nonzero(gt_mask & pm)
        union_count   += np.count_nonzero(gt_mask | pm)
        used_pred.add(best_p)

    # unmatched predicted instances → false positives
    for p in pred_ids:
        if p not in used_pred:
            union_count += np.count_nonzero(pred == p)

    return correct_count / (union_count + np.finfo(float).eps)


def compute_pq(gt, pred, iou_thresh=0.5):
    """
    Panoptic Quality = SQ × DQ.
    Optimal one-to-one matching via Hungarian algorithm; match threshold IoU ≥ iou_thresh.
    Returns (PQ, SQ, DQ).
    """
    gt_ids   = np.unique(gt);   gt_ids   = gt_ids[gt_ids   > 0]
    pred_ids = np.unique(pred); pred_ids = pred_ids[pred_ids > 0]

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0, 1.0, 1.0
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0, 0.0, 0.0

    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for i, g in enumerate(gt_ids):
        gm = (gt == g)
        for p_val in np.unique(pred[gm]):
            if p_val == 0:
                continue
            j  = np.where(pred_ids == p_val)[0][0]
            pm = (pred == p_val)
            inter = np.logical_and(gm, pm).sum()
            if inter > 0:
                iou_matrix[i, j] = inter / np.logical_or(gm, pm).sum()

    rows, cols = linear_sum_assignment(1.0 - iou_matrix)
    matched    = [iou_matrix[r, c] for r, c in zip(rows, cols)
                  if iou_matrix[r, c] >= iou_thresh]

    tp = len(matched)
    fp = len(pred_ids) - tp
    fn = len(gt_ids)   - tp

    sq = float(np.mean(matched)) if tp > 0 else 0.0
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-8)
    return sq * dq, sq, dq


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate EMCAF-Net: Dice, AJI, PQ')
    p.add_argument('--model_path',     type=str, required=True,
                   help='path to trained .pth checkpoint')
    p.add_argument('--fold',           type=int, required=True,
                   help='test fold index')
    p.add_argument('--dataset_path',   type=str, default='./dataset',
                   help='root dataset directory')
    p.add_argument('--arch',           type=str, default='nucleinet',
                   help='module in model/ providing the architecture')
    p.add_argument('--split',          type=str, default='test',
                   help="sub-folder to evaluate (normally 'test')")
    p.add_argument('--testsize',       type=int, default=512)
    p.add_argument('--min_distance',   type=int, default=7,
                   help='watershed seed min-separation (px), prediction only')
    p.add_argument('--min_size',       type=int, default=10,
                   help='min instance area (px²) to keep after watershed')
    p.add_argument('--iou_thresh',     type=float, default=0.5,
                   help='IoU threshold for PQ matching')
    p.add_argument('--label_mask_ext', type=str, default='.tif',
                   help='file extension in GT_instance/ folder')
    p.add_argument('--save_dir',       type=str, default='predictions',
                   help='directory for the saved binary prediction maps')
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # Load model
    Net = importlib.import_module(f'model.{args.arch}').nucleinet
    model = Net()
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model = model.to(device)
    model.eval()
    print(f'Model loaded from: {args.model_path}  (architecture: model/{args.arch}.py)')

    fold_tag         = f'fold{args.fold}'
    test_root        = os.path.join(args.dataset_path, args.split, fold_tag)
    image_root       = os.path.join(test_root, 'images') + '/'
    gt_root          = os.path.join(test_root, 'GT')     + '/'
    label_mask_root  = os.path.join(test_root, 'GT_instance')

    run_name = os.path.splitext(os.path.basename(args.model_path))[0]
    save_dir = os.path.join(args.save_dir, run_name, fold_tag)
    os.makedirs(save_dir, exist_ok=True)

    loader = test_dataset(image_root, gt_root, args.testsize)

    dice_scores = []
    aji_scores  = []
    pq_scores   = []
    sq_scores   = []
    dq_scores   = []
    missing_gt  = []
    rows        = []          # per-image scores for paired statistics
    time_sum    = 0.0

    # GPU warm-up so the first image does not distort the timing
    with torch.no_grad(), torch.cuda.amp.autocast():
        warm = torch.zeros(1, 3, args.testsize, args.testsize, device=device)
        for _ in range(5):
            model(warm)

    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(loader.size), desc=fold_tag):
            image, gt, name = loader.load_data()
            gt_np = np.asarray(gt, np.float32)

            image = image.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0    = time.time()
            res   = model(image)
            if device.type == 'cuda':
                torch.cuda.synchronize()     # wait for the GPU so the timing is real
            time_sum += time.time() - t0

            res       = F.interpolate(res.float(), size=gt_np.shape,
                                      mode='bilinear', align_corners=False)
            res_prob  = res.sigmoid().cpu().numpy().squeeze()
            res_bin   = (res_prob > 0.5).astype(np.uint8)

            imageio.imwrite(os.path.join(save_dir, name),
                            (res_bin * 255).astype(np.uint8))

            # Dice: binary pixel-level (no watershed)
            gt_bin = (gt_np > 0).astype(np.uint8)
            dice_scores.append(compute_dice(gt_bin, res_bin))
            cur_dice = dice_scores[-1]

            # AJI / PQ: instance-level (watershed on prediction)
            pred_inst = binary_to_instances(res_bin, args.min_distance, args.min_size)

            base          = os.path.splitext(name)[0]
            lm_path       = os.path.join(label_mask_root, base + args.label_mask_ext)

            if not os.path.exists(lm_path):
                missing_gt.append(name)
                continue

            gt_inst = load_instance_gt(lm_path)
            aji_scores.append(compute_aji(gt_inst, pred_inst))
            pq, sq, dq = compute_pq(gt_inst, pred_inst, args.iou_thresh)
            pq_scores.append(pq); sq_scores.append(sq); dq_scores.append(dq)
            rows.append((base, cur_dice, aji_scores[-1], pq))

    # ── Summary ────────────────────────────────────────────────────────────────
    n = loader.size
    print(f'\nRunning time: {time_sum / n:.5f} s/image  |  FPS: {n / time_sum:.2f}')

    print(f'\nResults for {fold_tag}  ({len(dice_scores)} images, model {run_name}):')
    print(f'  Dice : {np.mean(dice_scores):.4f}')
    if aji_scores:
        print(f'  AJI  : {np.mean(aji_scores):.4f}')
        print(f'  PQ   : {np.mean(pq_scores):.4f}  '
              f'(SQ: {np.mean(sq_scores):.4f}, DQ: {np.mean(dq_scores):.4f})')
    else:
        print('  AJI/PQ: no GT_instance/ files found')

    if missing_gt:
        print(f'\nWARNING: {len(missing_gt)} image(s) skipped for AJI/PQ '
              f'(no matching label mask found):')
        for nm in missing_gt[:5]:
            print(f'  {nm}')
        if len(missing_gt) > 5:
            print(f'  ...and {len(missing_gt) - 5} more')

    if rows:
        os.makedirs('per_image_scores', exist_ok=True)
        csv_path = os.path.join('per_image_scores', f'{run_name}__{args.split}.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['image', 'dice', 'aji', 'pq'])
            w.writerows(rows)
        print(f'Per-image scores written to {csv_path}')

    print(f'\nWatershed params (prediction only): '
          f'min_distance={args.min_distance}, '
          f'min_size={args.min_size}, '
          f'iou_thresh={args.iou_thresh}')


if __name__ == '__main__':
    main()
