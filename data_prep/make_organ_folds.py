r"""
make_organ_folds.py - build organ-disjoint (and therefore slide-disjoint) 5-fold data from flat folders

Input  (three flat folders, file name = organ + number):
    --images   ...\images          e.g. "human bladder_human_bladder_01.png"
    --gt       ...\GT              same stem as the image
    --inst     ...\instance        instance map, e.g. "human_bladder_01.png"
                                   (the end of the image name, e.g. human_bladder_01; the full image
                                    stem is tried first)  [optional: leave out to skip GT_instance]
Organ = the part before the repeated name: "human bladder_human_bladder_01" and
    "human_bladder_human_bladder_01" both give organ "human bladder".

Output (new, empty folder) - same layout build_dataset.py reads:
    <dst>\Fold_N\train\images, GT        images of the other 4 folds' organs
    <dst>\Fold_N\test\images, GT, GT_instance     whole organs, never in train
    <dst>\organ_folds.csv                organ, fold, n_images   (publish this file)
Every organ is the test organ of exactly one fold. Folds are balanced to ~n/5 images by greedy
assignment (largest organ first, ties and order shuffled by --seed). Files are copied, not changed.

    python data_prep\make_organ_folds.py --images .\images --gt .\GT --inst .\instance --dst .\dataset_original_organ
Then:
    python data_prep\check_organ_disjoint.py --root .\dataset_original_organ
    python data_prep\build_dataset.py --src .\dataset_original_organ --dst .\dataset_organ --val_by_organ
"""
import argparse
import csv
import os
import random
import shutil
import sys
from collections import defaultdict

EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def files(d):
    out = {}
    for f in sorted(os.listdir(d)):
        s, e = os.path.splitext(f)
        if e.lower() in EXTS:
            out[s] = os.path.join(d, f)
    return out

def parse(stem):
    """-> (organ, number-part). Names look like '<organ>_<organ with _>_<n>', with the organ written
    either with a space ('human bladder_human_bladder_01') or with underscores
    ('human_bladder_human_bladder_01'). Falls back to the text before the first underscore."""
    base, _, num = stem.rpartition('_')
    for i, ch in enumerate(base):
        if ch == '_' and base[:i].replace(' ', '_') == base[i + 1:]:
            return base[:i].replace('_', ' '), num
    return stem.split('_')[0], num


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--images', required=True)
ap.add_argument('--gt', required=True)
ap.add_argument('--inst', default='')
ap.add_argument('--dst', required=True)
ap.add_argument('--folds', type=int, default=5)
ap.add_argument('--seed', type=int, default=12393)
a = ap.parse_args()

if os.path.isdir(a.dst) and os.listdir(a.dst):
    sys.exit(f'{a.dst} exists and is not empty. Use a new folder name or remove it first.')
img, gt = files(a.images), files(a.gt)
inst = files(a.inst) if a.inst else {}
miss = sorted(set(img) ^ set(gt))
if miss:
    sys.exit(f'{len(miss)} names are not in both images and GT, e.g. {miss[:3]}')


def inst_for(s):
    """instance map whose name is the end of the image name: the longest suffix of the image stem
    (taken after an underscore) that exists in the instance folder, e.g.
    'human bladder_human_bladder_01' -> 'human_bladder_01'."""
    if s in inst:
        return inst[s]
    for i, ch in enumerate(s):
        if ch == '_' and s[i + 1:] in inst:
            return inst[s[i + 1:]]
    return None


if inst:
    bad = [s for s in img if inst_for(s) is None]
    if bad:
        sys.exit(f'{len(bad)} images have no instance map, e.g. {bad[:3]}')

by_organ = defaultdict(list)
for s in img:
    by_organ[parse(s)[0]].append(s)
organs = sorted(by_organ)
if len(organs) < a.folds:
    sys.exit(f'Only {len(organs)} organs found for {a.folds} folds. Check --images.')

rnd = random.Random(a.seed)
rnd.shuffle(organs)
organs.sort(key=lambda o: -len(by_organ[o]))          # stable: ties keep the shuffled order
load = [0] * a.folds
fold_of = {}
for o in organs:
    k = min(range(a.folds), key=lambda i: (load[i], i))
    fold_of[o] = k + 1
    load[k] += len(by_organ[o])

for k in range(1, a.folds + 1):
    for sp in ('train', 'test'):
        for sub in ('images', 'GT') + (('GT_instance',) if inst and sp == 'test' else ()):
            os.makedirs(os.path.join(a.dst, f'Fold_{k}', sp, sub), exist_ok=True)


def put(path, k, sp, sub):
    shutil.copy2(path, os.path.join(a.dst, f'Fold_{k}', sp, sub, os.path.basename(path)))


for s in sorted(img):
    tk = fold_of[parse(s)[0]]
    for k in range(1, a.folds + 1):
        sp = 'test' if k == tk else 'train'
        put(img[s], k, sp, 'images')
        put(gt[s], k, sp, 'GT')
        if inst and sp == 'test':
            put(inst_for(s), k, sp, 'GT_instance')

with open(os.path.join(a.dst, 'organ_folds.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['organ', 'fold', 'n_images'])
    for o in sorted(fold_of, key=lambda o: (fold_of[o], o)):
        w.writerow([o, fold_of[o], len(by_organ[o])])
print(f'{len(img)} images, {len(by_organ)} organs')
for k in range(1, a.folds + 1):
    print(f'Fold_{k}: test {load[k-1]} images, {sum(1 for o in fold_of if fold_of[o]==k)} organs, '
          f'train {len(img)-load[k-1]}')
print(f'\nDone. Fold table: {os.path.join(a.dst, "organ_folds.csv")}')
print(r'Next: python data_prep\check_organ_disjoint.py --root ' + a.dst)
