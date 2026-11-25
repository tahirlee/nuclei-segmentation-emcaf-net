r"""
build_dataset.py - build  train / val / test / train_aug  from the verified original folds

Input  (already checked with check_original_folds.py):
    dataset_original\Fold_N\train\images, GT        (532 images)
    dataset_original\Fold_N\test\images,  GT        (133 images)  [+ GT_instance if present]

Output (a NEW folder, it refuses to write into a non-empty one):
    dataset\test\foldN\images, GT [, GT_instance]   copy of the original test fold, untouched
    dataset\val\foldN\images, GT                    54 images taken from the original train, stratified by organ
    dataset\train\foldN\images, GT                  the remaining 478 images, NOT augmented
    dataset\train_aug\foldN\images, GT              the 478 images x 8 versions = 3,824
    dataset\split_manifest.csv                      fold, image, split, organ

Offline augmentation (8 versions per training image, file name = original name + suffix):
    __orig           the unchanged image
    __tcr0/1/2       translate + crop + resize back to 512 (crop 70-95 % of the side, random position;
                     crops with no nucleus are re-drawn)
    __fliph/__flipv  horizontal / vertical flip
    __color0/1       brightness, contrast, saturation, hue jitter and Gaussian noise (image only)
Masks follow every geometric change with nearest-neighbour resampling and stay strictly {0, 255}.
Validation and test images are never augmented and never appear in train / train_aug.

With --val_by_organ the validation images are whole organs (no organ shared with train / train_aug),
about --val_n images; use it on organ-disjoint folds from make_organ_folds.py.

Optional mask clean-up (train and val only, the test masks are never modified):
    --clean_masks         removes foreground specks and fills background holes of <= 3 pixels
                          (annotation rasterisation artefacts); larger structures are untouched
    --exclude names.txt   text file, one image name per line (as in the original folder, without
                          extension); these images are dropped from train and val (never from test)
Everything that was changed or dropped is written to cleaning_log.csv.

    python data_prep\build_dataset.py --src .\dataset_original --dst .\dataset
Afterwards:
    python data_prep\check_dataset.py --dataset_path .\dataset
"""
import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageEnhance
from scipy import ndimage as ndi

EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
IMG_NAMES = ('images', 'image', 'img', 'imgs')
GT_NAMES = ('gt', 'mask', 'masks', 'label', 'labels', 'groundtruth')
INST_NAMES = ('gt_instance', 'instance', 'gt_inst')
VARIANTS = ['orig', 'tcr0', 'tcr1', 'tcr2', 'fliph', 'flipv', 'color0', 'color1']


def child(parent, names):
    for d in os.listdir(parent):
        if d.lower() in names and os.path.isdir(os.path.join(parent, d)):
            return os.path.join(parent, d)
    return None


def files(d):
    out = {}
    for f in sorted(os.listdir(d)):
        s, e = os.path.splitext(f)
        if e.lower() in EXTS:
            out[s] = os.path.join(d, f)
    return out


def organ(stem):
    return stem.split('_')[0]


def rng_for(seed, *keys):
    h = hashlib.md5('|'.join(map(str, (seed,) + keys)).encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def clean_binary(fg, max_px=3):
    """remove foreground components and fill background holes of <= max_px pixels"""
    fg = fg.copy()
    lab, n = ndi.label(fg)
    removed = filled = 0
    if n:
        sz = np.bincount(lab.ravel())
        small = np.where(sz[1:] <= max_px)[0] + 1
        removed = len(small)
        fg[np.isin(lab, small)] = False
    lab, n = ndi.label(~fg)
    if n:
        sz = np.bincount(lab.ravel())
        small = np.where(sz[1:] <= max_px)[0] + 1
        filled = len(small)
        fg[np.isin(lab, small)] = True
    return fg, removed, filled


def load_pair(ip, mp, clean=False, stats=None):
    img = Image.open(ip).convert('RGB')
    m = np.asarray(Image.open(mp))
    m = m[..., 0] if m.ndim == 3 else m
    fg = m > 0
    if clean:
        fg, r, f = clean_binary(fg)
        if stats is not None and (r or f):
            stats.append((os.path.basename(ip), r, f))
    mask = Image.fromarray((fg * 255).astype(np.uint8))
    return img, mask


def tcr(img, mask, rng, size):
    w, h = img.size
    m = np.asarray(mask) > 0
    for _ in range(12):
        s = rng.uniform(0.70, 0.95)
        cw, ch = int(w * s), int(h * s)
        x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
        if m[y0:y0 + ch, x0:x0 + cw].sum() >= 50 or m.sum() < 50:
            break
    box = (x0, y0, x0 + cw, y0 + ch)
    return (img.crop(box).resize((size, size), Image.BILINEAR),
            mask.crop(box).resize((size, size), Image.NEAREST))


def color(img, rng):
    for enh, lo, hi in ((ImageEnhance.Brightness, 0.8, 1.2), (ImageEnhance.Contrast, 0.8, 1.2),
                        (ImageEnhance.Color, 0.8, 1.2)):
        img = enh(img).enhance(float(rng.uniform(lo, hi)))
    hsv = np.asarray(img.convert('HSV')).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-8, 9))) % 256          # small hue shift
    img = Image.fromarray(hsv.astype(np.uint8), 'HSV').convert('RGB')
    a = np.asarray(img).astype(np.float32) + rng.normal(0, float(rng.uniform(2, 8)), (img.size[1], img.size[0], 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def make(variant, img, mask, rng, size):
    if variant == 'orig':
        return img.resize((size, size), Image.BILINEAR) if img.size != (size, size) else img, \
               mask.resize((size, size), Image.NEAREST) if mask.size != (size, size) else mask
    if variant.startswith('tcr'):
        return tcr(img, mask, rng, size)
    if variant == 'fliph':
        return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
    if variant == 'flipv':
        return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
    return color(img, rng), mask


def choose_val(names, n_val, rng):
    """stratified by organ, exactly n_val images (largest-remainder allocation)"""
    groups = defaultdict(list)
    for s in names:
        groups[organ(s)].append(s)
    tot = len(names)
    quota = {g: len(v) * n_val / tot for g, v in groups.items()}
    take = {g: int(q) for g, q in quota.items()}
    rest = n_val - sum(take.values())
    for g in sorted(groups, key=lambda g: quota[g] - take[g], reverse=True)[:rest]:
        take[g] += 1
    val = []
    for g, v in sorted(groups.items()):
        v = sorted(v)
        idx = rng.permutation(len(v))[:take[g]]
        val += [v[i] for i in idx]
    return set(val)


def choose_val_organs(names, n_val, rng):
    """validation = whole organs (no organ shared with the training images), total closest to n_val,
    at least 2 organs"""
    groups = defaultdict(list)
    for s in names:
        groups[organ(s)].append(s)
    keys = sorted(groups)
    best = None
    for _ in range(300):
        order = [keys[i] for i in rng.permutation(len(keys))]
        pick, tot = [], 0
        for o in order:
            pick.append(o)
            tot += len(groups[o])
            if tot >= n_val and len(pick) >= 2:
                break
        score = abs(tot - n_val)
        if len(pick) >= 2 and (best is None or score < best[0]):
            best = (score, pick)
    return {s for o in best[1] for s in groups[o]}


def save_img(im, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='./dataset_original')
    ap.add_argument('--dst', default='./dataset')
    ap.add_argument('--val_n', type=int, default=54)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=12393)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--val_by_organ', action='store_true',
                    help='validation = whole organs taken from the training pool (for organ-disjoint folds)')
    ap.add_argument('--clean_masks', action='store_true',
                    help='remove specks / fill holes of <= 3 px in train and val masks (never test)')
    ap.add_argument('--exclude', default='', help='text file with image names to drop from train and val')
    a = ap.parse_args()
    excl = set()
    if a.exclude:
        excl = {l.strip() for l in open(a.exclude, encoding='utf-8') if l.strip()}
        excl = {os.path.splitext(x)[0] for x in excl}
    log = []

    if os.path.isdir(a.dst) and os.listdir(a.dst):
        sys.exit(f'{a.dst} exists and is not empty. Use a new folder name or remove it first.')
    folds = {}
    for d in os.listdir(a.src):
        m = re.fullmatch(r'fold[_\- ]?(\d+)', d, re.I)
        if m:
            folds[int(m.group(1))] = os.path.join(a.src, d)
    if not folds:
        sys.exit(f'No Fold_N folders in {a.src}')

    manifest = []
    for n, fdir in sorted(folds.items()):
        subs = {d.lower(): os.path.join(fdir, d) for d in os.listdir(fdir)}
        tr_i, tr_g = files(child(subs['train'], IMG_NAMES)), files(child(subs['train'], GT_NAMES))
        te_i, te_g = files(child(subs['test'], IMG_NAMES)), files(child(subs['test'], GT_NAMES))
        te_x = child(subs['test'], INST_NAMES)
        te_x = files(te_x) if te_x else {}
        assert set(tr_i) == set(tr_g) and set(te_i) == set(te_g), f'fold {n}: image/mask names differ'
        assert not (set(tr_i) & set(te_i)), f'fold {n}: train and test share names'
        fo = f'fold{n}'

        val = (choose_val_organs if a.val_by_organ else choose_val)(sorted(tr_i), a.val_n, rng_for(a.seed, 'val', n))
        train = sorted(set(tr_i) - val)
        dropped = sorted((set(train) | val) & excl)
        for s in dropped:
            log.append((fo, s, 'excluded', '', ''))
        train = [s for s in train if s not in excl]
        val = {s for s in val if s not in excl}

        # test: copied untouched
        for s in sorted(te_i):
            for sub, table in (('images', te_i), ('GT', te_g), ('GT_instance', te_x)):
                if s in table:
                    p = os.path.join(a.dst, 'test', fo, sub)
                    os.makedirs(p, exist_ok=True)
                    shutil.copy2(table[s], os.path.join(p, os.path.basename(table[s])))
            manifest.append((fo, s, 'test', organ(s)))
        # val and train (not augmented): binary masks, original size
        for s in sorted(val) + train:
            sp = 'val' if s in val else 'train'
            st = []
            img, mask = load_pair(tr_i[s], tr_g[s], a.clean_masks, st)
            log += [(fo, s, 'cleaned', r, f) for _, r, f in st]
            save_img(img, os.path.join(a.dst, sp, fo, 'images', s + '.png'))
            save_img(mask, os.path.join(a.dst, sp, fo, 'GT', s + '.png'))
            manifest.append((fo, s, sp, organ(s)))

        # train_aug
        def work(s):
            img, mask = load_pair(tr_i[s], tr_g[s], a.clean_masks)
            for v in VARIANTS:
                im2, m2 = make(v, img, mask, rng_for(a.seed, 'aug', n, s, v), a.size)
                save_img(im2, os.path.join(a.dst, 'train_aug', fo, 'images', f'{s}__{v}.png'))
                save_img(m2, os.path.join(a.dst, 'train_aug', fo, 'GT', f'{s}__{v}.png'))
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(work, train))
        print(f'{fo}: test {len(te_i)}, val {len(val)}, train {len(train)}, train_aug {len(train) * len(VARIANTS)}',
              flush=True)

    with open(os.path.join(a.dst, 'split_manifest.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['fold', 'image', 'split', 'organ'])
        w.writerows(manifest)
    with open(os.path.join(a.dst, 'cleaning_log.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['fold', 'image', 'action', 'specks_removed', 'holes_filled'])
        w.writerows(log)
    nc = sum(1 for x in log if x[2] == 'cleaned')
    print(f'\nMasks cleaned: {nc} train/val masks changed, {sum(1 for x in log if x[2] == "excluded")} images excluded '
          f'(see cleaning_log.csv). Test masks are untouched.')
    print(f'\nDone. Manifest: {os.path.join(a.dst, "split_manifest.csv")}')
    print(r'Next: python data_prep\check_dataset.py --dataset_path ' + a.dst)


if __name__ == '__main__':
    main()
