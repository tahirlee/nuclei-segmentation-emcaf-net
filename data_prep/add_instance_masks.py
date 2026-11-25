r"""
add_instance_masks.py - copy the NuInsSeg instance masks into every test fold

    dataset_new\instance_masks\human_bladder_01.tif   ... 665 files (16-bit tif, 0 = background)
    dataset_new\test\fold1\images\human bladder_human_bladder_01.png (or .tif / .jpg)

Each test image is matched to its instance mask by name (the test image name is
"<organ folder>_<file name>", the instance mask is "<file name>.tif"; spaces/underscores and
capital letters are ignored). The file is copied to

    dataset_new\test\foldN\GT_instance\<test image name>.tif

which is where test_instance.py looks for it (--label_mask_ext .tif).

Every copied mask is checked against the binary GT of the same test image: the instance foreground
(label > 0) must equal the binary foreground. Anything that does not match is reported.

    python data_prep\add_instance_masks.py --dataset_path .\dataset_new --instance_dir .\dataset_new\instance_masks
"""
import argparse
import csv
import os
import re
import shutil
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def norm(s):
    return re.sub(r'[\s_]+', '_', s.strip().lower())


def read(path):
    a = np.asarray(Image.open(path))
    return a[..., 0] if a.ndim == 3 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', default='./dataset_new')
    ap.add_argument('--instance_dir', default='./dataset_new/instance_masks')
    ap.add_argument('--tol', type=float, default=0.001,
                    help='allowed fraction of pixels where instance foreground differs from binary GT')
    a = ap.parse_args()

    # ---- index the instance masks
    inst = {}
    dup = []
    for dp, _, fs in os.walk(a.instance_dir):
        for f in fs:
            s, e = os.path.splitext(f)
            if e.lower() in EXTS:
                k = norm(s)
                if k in inst:
                    dup.append(f)
                inst[k] = os.path.join(dp, f)
    if not inst:
        sys.exit(f'No instance masks found in {a.instance_dir}')
    print(f'Instance masks found: {len(inst)}' + (f'   (duplicate names: {dup[:3]})' if dup else ''))

    tdir = os.path.join(a.dataset_path, 'test')
    folds = sorted([d for d in os.listdir(tdir) if re.fullmatch(r'fold\d+', d)], key=lambda s: int(s[4:]))
    used = defaultdict(list)
    rows, problems = [], []
    keys_by_len = sorted(inst, key=len, reverse=True)
    for fo in folds:
        idir, gdir = os.path.join(tdir, fo, 'images'), os.path.join(tdir, fo, 'GT')
        odir = os.path.join(tdir, fo, 'GT_instance')
        os.makedirs(odir, exist_ok=True)
        n_ok = 0
        for f in sorted(os.listdir(idir)):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in EXTS:
                continue
            ns = norm(stem)
            hit = next((k for k in keys_by_len if ns == k or ns.endswith('_' + k)), None)
            if hit is None:
                problems.append((fo, stem, 'no instance mask found'))
                continue
            gt_path = next((os.path.join(gdir, g) for g in os.listdir(gdir)
                            if os.path.splitext(g)[0] == stem), None)
            if gt_path is None:
                problems.append((fo, stem, 'no binary GT with this name'))
                continue
            im, gt = read(inst[hit]), read(gt_path)
            if im.shape != gt.shape:
                problems.append((fo, stem, f'size {im.shape} vs GT {gt.shape}'))
                continue
            diff = float(((im > 0) != (gt > 0)).mean())
            nlab = len(np.unique(im[im > 0]))
            if diff > a.tol:
                problems.append((fo, stem, f'instance foreground differs from binary GT in {diff * 100:.2f}% of pixels '
                                 f'(matched {os.path.basename(inst[hit])})'))
                continue
            shutil.copy2(inst[hit], os.path.join(odir, stem + '.tif'))
            used[hit].append((fo, stem))
            rows.append((fo, stem, os.path.basename(inst[hit]), nlab, f'{diff * 100:.4f}'))
            n_ok += 1
        print(f'{fo}: {n_ok} instance masks copied')

    multi = {k: v for k, v in used.items() if len(v) > 1}
    unused = sorted(set(inst) - set(used))
    with open(os.path.join(a.dataset_path, 'instance_copy_log.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['fold', 'test_image', 'instance_file', 'n_instances', 'foreground_diff_percent'])
        w.writerows(rows)

    print('\n' + '=' * 70)
    print(f'Copied {len(rows)} instance masks.  Instances per image: '
          f'median {int(np.median([r[3] for r in rows])) if rows else 0}, '
          f'max {max([r[3] for r in rows]) if rows else 0}')
    if problems:
        print(f'PROBLEMS ({len(problems)}):')
        for fo, s, m in problems[:15]:
            print(f'   {fo}/{s}: {m}')
    if multi:
        print(f'PROBLEM: {len(multi)} instance masks were used for more than one test image, e.g. '
              f'{list(multi.items())[:2]}')
    if unused:
        print(f'Note: {len(unused)} instance masks were not used by any test fold '
              f'(expected: the images that are in train/val, not test).')
    ok = not problems and not multi
    print('RESULT: ' + ('all test images have a matching instance mask.' if ok else 'fix the problems above.'))
    print('=' * 70)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
