r"""
organ_macro.py - image-level and organ-macro results from the per-image CSVs of test_instance.py

Reads  per_image_scores\<run>_fold<k>_lr3__test.csv  (columns image,dice,aji,pq) for every run and fold.
Organ = the part of the image name before the repeated name ("human bladder_human_bladder_01" ->
"human bladder"). Organ-folds are slide-disjoint, so organ-macro stands in for WSI-macro.

Prints, per run:
  image-level   mean over all test images (all folds pooled)
  organ-macro   mean of the per-organ means (every organ counts once, whatever its image count)
  fold mean +- SD   of the per-fold image-level means
and, for two runs, the organ-level paired difference (A - B) with the number of organs where A is better.
Writes organ_map.csv (image,group), per_organ_<run>.csv and organ_macro_summary.csv.

    python analysis/organ_macro.py --runs emcaf_organ dual_organ
    python analysis/organ_macro.py --runs emcaf_organ dual_organ --scores_dir per_image_scores --out organ_results
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np


def parse(stem):
    base, _, num = stem.rpartition('_')
    for i, ch in enumerate(base):
        if ch == '_' and base[:i].replace(' ', '_') == base[i + 1:]:
            return base[:i].replace('_', ' '), num
    return stem.split('_')[0], num


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--runs', nargs='+', required=True, help='run names, e.g. emcaf_organ dual_organ')
ap.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
ap.add_argument('--metrics', nargs='+', default=['dice', 'aji', 'pq'])
ap.add_argument('--scores_dir', default='per_image_scores')
ap.add_argument('--pattern', default='{run}_fold{k}_lr3__test{suffix}.csv')
ap.add_argument('--suffix', default='')
ap.add_argument('--out', default='organ_results')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

data = {}                       # run -> {image: {metric: v}} ; fold_of[image]
fold_of = {}
for run in a.runs:
    d = {}
    for k in a.folds:
        p = os.path.join(a.scores_dir, a.pattern.format(run=run, k=k, suffix=a.suffix))
        if not os.path.exists(p):
            sys.exit(f'Missing {p}. Evaluate every fold of {run} first.')
        for r in csv.DictReader(open(p, newline='', encoding='utf-8')):
            d[r['image']] = {m: float(r[m]) for m in a.metrics}
            fold_of[r['image']] = k
    data[run] = d
imgs = sorted(set.intersection(*[set(d) for d in data.values()]))
for run, d in data.items():
    if len(d) != len(imgs):
        print(f'WARNING: {run} has {len(d)} images, common to all runs: {len(imgs)}')
organ = {s: parse(s)[0] for s in imgs}
with open(os.path.join(a.out, 'organ_map.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['image', 'group'])
    w.writerows((s, organ[s]) for s in imgs)
organs = sorted(set(organ.values()))
print(f'{len(imgs)} images, {len(organs)} organs, folds {a.folds}\n')

summ, per_organ = [], {}
hdr = f'{"run":<18}{"level":<13}' + ''.join(f'{m.upper():>9}' for m in a.metrics)
print(hdr + '\n' + '-' * len(hdr))
for run in a.runs:
    d = data[run]
    img_lvl = {m: np.mean([d[s][m] for s in imgs]) for m in a.metrics}
    po = {o: {m: np.mean([d[s][m] for s in imgs if organ[s] == o]) for m in a.metrics} for o in organs}
    per_organ[run] = po
    macro = {m: np.mean([po[o][m] for o in organs]) for m in a.metrics}
    fm = {m: [np.mean([d[s][m] for s in imgs if fold_of[s] == k]) for k in a.folds] for m in a.metrics}
    for lab, vals in (('image-level', img_lvl), ('organ-macro', macro)):
        print(f'{run:<18}{lab:<13}' + ''.join(f'{100 * vals[m]:>9.2f}' for m in a.metrics))
        summ.append([run, lab] + [f'{100 * vals[m]:.2f}' for m in a.metrics])
    print(f'{run:<18}{"fold mean±SD":<13}' + ''.join(
        f'{100 * np.mean(fm[m]):>5.1f}±{100 * np.std(fm[m], ddof=1):<3.1f}' for m in a.metrics))
    summ.append([run, 'fold mean±SD'] + [f'{100 * np.mean(fm[m]):.2f}±{100 * np.std(fm[m], ddof=1):.2f}'
                                         for m in a.metrics])
    with open(os.path.join(a.out, f'per_organ_{run}.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['organ', 'n_images'] + a.metrics)
        for o in organs:
            w.writerow([o, sum(1 for s in imgs if organ[s] == o)] + [f'{po[o][m]:.4f}' for m in a.metrics])
    print()

if len(a.runs) >= 2:
    A, B = a.runs[0], a.runs[1]
    print(f'Organ-level paired difference  {A} - {B}  (percentage points)')
    for m in a.metrics:
        diff = np.array([per_organ[A][o][m] - per_organ[B][o][m] for o in organs]) * 100
        wins = int((diff > 0).sum())
        print(f'  {m.upper():<5} mean {diff.mean():+.2f}   median {np.median(diff):+.2f}   '
              f'{A} better in {wins}/{len(organs)} organs, worse in {int((diff < 0).sum())}')
    print('\nFor confidence intervals and the corrected fold-level test run tools\\paired_stats.py with '
          f'--groups {os.path.join(a.out, "organ_map.csv")} --n_train 478 --n_test 133')
with open(os.path.join(a.out, 'organ_macro_summary.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['run', 'level'] + a.metrics)
    w.writerows(summ)
print(f'\nWritten to {a.out}\\')
