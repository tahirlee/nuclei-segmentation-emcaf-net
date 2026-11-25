r"""
check_organ_disjoint.py - are the folds organ-disjoint?

Organ = file name up to the first underscore (same rule as build_dataset.py), e.g.
"human bladder_human_bladder_03" -> "human bladder" (species + organ).

Organ-disjoint means: within every fold, no organ occurs in both test and train (or val), and
every organ is the test organ of exactly one fold. Pick ONE input:

  python data_prep\check_organ_disjoint.py --membership fold_membership.csv     (from check_original_folds.py)
  python data_prep\check_organ_disjoint.py --root .\dataset_original            (Fold_N\train|test\images)
  python data_prep\check_organ_disjoint.py --manifest .\dataset\split_manifest.csv   (from build_dataset.py)

Only file names are read. Nothing is changed. Exit code 0 = organ-disjoint, 1 = not.
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
def parse(stem):
    """-> (organ, number-part). Names look like '<organ>_<organ with _>_<n>', with the organ written
    either with a space ('human bladder_human_bladder_01') or with underscores
    ('human_bladder_human_bladder_01'). Falls back to the text before the first underscore."""
    base, _, num = stem.rpartition('_')
    for i, ch in enumerate(base):
        if ch == '_' and base[:i].replace(' ', '_') == base[i + 1:]:
            return base[:i].replace('_', ' '), num
    return stem.split('_')[0], num


organ_of = lambda stem: parse(stem)[0]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--membership', default='')
ap.add_argument('--root', default='')
ap.add_argument('--manifest', default='')
a = ap.parse_args()
if sum(bool(x) for x in (a.membership, a.root, a.manifest)) != 1:
    sys.exit('Give exactly one of --membership, --root, --manifest.')

cnt = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))   # fold -> split -> organ -> n
if a.membership:
    rows = list(csv.DictReader(open(a.membership, newline='', encoding='utf-8')))
    for r in rows:
        cnt[r['test_fold']]['test'][organ_of(r['name_in_test'])] += 1
    # in a k-fold design train(fold k) = test of all other folds
    folds = sorted(cnt, key=lambda s: int(re.sub(r'\D', '', s)))
    for f in folds:
        for g in folds:
            if g != f:
                for o, n in cnt[g]['test'].items():
                    cnt[f]['train'][o] += n
elif a.root:
    def add(fold, sp, sdir):
        idir = next((os.path.join(sdir, c) for c in os.listdir(sdir)
                     if c.lower() in ('images', 'image', 'img', 'imgs')), None)
        for f in (os.listdir(idir) if idir else []):
            stem, ext = os.path.splitext(f)
            if ext.lower() in EXTS:
                cnt[f'fold{fold}'][sp][organ_of(stem)] += 1
    fold_re = re.compile(r'fold[_\- ]?(\d+)', re.I)
    for d in os.listdir(a.root):
        p = os.path.join(a.root, d)
        if not os.path.isdir(p):
            continue
        if fold_re.fullmatch(d):                       # root\Fold_N\split\images
            for sp in os.listdir(p):
                if sp.lower() in ('train', 'val', 'test') and os.path.isdir(os.path.join(p, sp)):
                    add(fold_re.fullmatch(d).group(1), sp.lower(), os.path.join(p, sp))
        elif d.lower() in ('train', 'val', 'test'):    # root\split\foldN\images
            for fd in os.listdir(p):
                m = fold_re.fullmatch(fd)
                if m and os.path.isdir(os.path.join(p, fd)):
                    add(m.group(1), d.lower(), os.path.join(p, fd))
else:
    for r in csv.DictReader(open(a.manifest, newline='', encoding='utf-8')):
        cnt[r['fold']][r['split']][r['organ']] += 1

folds = sorted(cnt, key=lambda s: int(re.sub(r'\D', '', s)))
if not folds:
    sys.exit('No folds found. Check the path.')
leak_organs = 0
test_folds_of = defaultdict(set)
print(f'{"fold":<7}{"test imgs":>10}{"test organs":>13}{"also in train/val":>19}{"test imgs affected":>20}{"val organs in train":>21}')
for f in folds:
    c = cnt[f]
    shared = [o for o in c['test'] if c['train'].get(o) or c['val'].get(o)]
    for o in c['test']:
        test_folds_of[o].add(f)
    val_leak = [o for o in c['val'] if c['train'].get(o)]
    leak_organs += len(shared) + len(val_leak)
    print(f'{f:<7}{sum(c["test"].values()):>10}{len(c["test"]):>13}{len(shared):>19}'
          f'{sum(c["test"][o] for o in shared):>20}{len(val_leak):>21}')

split_organs = {o: sorted(v) for o, v in test_folds_of.items() if len(v) > 1}
print(f'\nOrgans: {len(test_folds_of)}.  Organs with test images in more than one fold: '
      f'{len(split_organs)}')
for o, v in list(sorted(split_organs.items()))[:5]:
    print(f'   {o}: test in {", ".join(v)}')
if len(split_organs) > 5:
    print(f'   ... and {len(split_organs) - 5} more')
ok = leak_organs == 0 and not split_organs
print('\nRESULT: ' + ('ORGAN-DISJOINT: no test organ occurs in train or val, and every organ is tested '
                      'in exactly one fold.' if ok else
                      'NOT organ-disjoint: the folds are split at patch level, test organs also occur '
                      'in training.'))
sys.exit(0 if ok else 1)
