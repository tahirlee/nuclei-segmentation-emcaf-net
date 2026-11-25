r"""
check_original_folds.py - verify the ORIGINAL cross-validation folders (before any augmentation)

Expected layout (folder names are matched case-insensitively):
    dataset_original\
        Fold_1\
            train\ images\  GT\
            test\  images\  GT\
        Fold_2\ ...  Fold_5\

Usage (from the project folder; check_dataset.py must be in the same folder):
    python data_prep\check_original_folds.py --root .\dataset_original

Checks, per fold and across folds
  - every image has a mask (same file stem) and vice versa, files open, same height/width
  - masks are binary, not empty, no label speckle, edges of the mask agree with the image
  - train and test of one fold share NO image (identical pixels, flipped copies, same file name)
  - the test sets of the folds do not overlap each other
  - every image is in exactly one test set (so every image is tested once)
  - train of fold k = all test sets of the other folds (nothing missing, nothing extra)
  - no duplicated images inside a split

Output in  dataset_original_check\ :
    report.txt            what is printed on screen
    issues.csv            every problem
    samples.csv           measurements per image
    fold_membership.csv   image name, content hash, the fold in which it is test   <- use this to rebuild folds
    sheets\*.png          overlay sheets (red = mask boundary)
"""
import argparse
import csv
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

try:
    from check_dataset import EXTS, POP, analyse, list_stems, ndi
except ImportError:
    sys.exit('check_dataset.py must be in the same folder as this script.')
from PIL import ImageDraw

IMG_NAMES = ('images', 'image', 'img', 'imgs')
GT_NAMES = ('gt', 'mask', 'masks', 'label', 'labels', 'groundtruth')
INST_NAMES = ('gt_instance', 'instance', 'gt_inst')


def find_child(parent, names):
    if not os.path.isdir(parent):
        return None
    for d in os.listdir(parent):
        if d.lower() in names and os.path.isdir(os.path.join(parent, d)):
            return os.path.join(parent, d)
    return None


def tile(rec, size=256):
    img = Image.open(rec['ip']).convert('RGB').resize((size, size), Image.BILINEAR)
    m = np.asarray(Image.open(rec['mp']))
    m = m[..., 0] if m.ndim == 3 else m
    m = np.asarray(Image.fromarray(((m > 0) * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)) > 0
    a = np.asarray(img).copy()
    a[m & ~ndi.binary_erosion(m)] = (255, 0, 0)
    t = Image.fromarray(a)
    d = ImageDraw.Draw(t)
    d.rectangle([0, 0, size, 22], fill=(0, 0, 0))
    d.text((3, 2), f'{rec["split"]}/{rec["fold"]}/{rec["name"]}'[-40:], fill=(255, 255, 255))
    g = rec['edge_gain']
    d.text((3, 11), f'gain={g:.2f}' if g == g else 'gain=n/a', fill=(255, 255, 0))
    return t


def sheet(recs, path, cols=4, size=256):
    if not recs:
        return
    rows = (len(recs) + cols - 1) // cols
    s = Image.new('RGB', (cols * size, rows * size), (30, 30, 30))
    for i, r in enumerate(recs):
        try:
            s.paste(tile(r, size), ((i % cols) * size, (i // cols) * size))
        except Exception:
            pass
    s.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./dataset_original')
    ap.add_argument('--out', default='dataset_original_check')
    ap.add_argument('--expected_total', type=int, default=0,
                    help='expected number of different images in the whole dataset (e.g. 665); 0 = do not check')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--gain_min', type=float, default=1.3)
    ap.add_argument('--hash_dist', type=int, default=3)
    ap.add_argument('--sheet_n', type=int, default=16)
    ap.add_argument('--no_sheets', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lines = []

    def say(s=''):
        print(s)
        lines.append(s)

    # ---- discover layout
    folds = {}
    for d in os.listdir(args.root) if os.path.isdir(args.root) else []:
        m = re.fullmatch(r'fold[_\- ]?(\d+)', d, re.I)
        if m and os.path.isdir(os.path.join(args.root, d)):
            folds[int(m.group(1))] = os.path.join(args.root, d)
    if not folds:
        say(f'No Fold_N folders found in {os.path.abspath(args.root)}. Check --root.')
        sys.exit(2)
    say(f'Root: {os.path.abspath(args.root)}')
    say('Folds found: ' + ', '.join(f'{k} ({os.path.basename(v)})' for k, v in sorted(folds.items())))

    jobs, issues, counts = [], [], {}
    for n, fdir in sorted(folds.items()):
        fo = f'fold{n}'
        subs = {d.lower(): os.path.join(fdir, d) for d in os.listdir(fdir) if os.path.isdir(os.path.join(fdir, d))}
        for sp in ('train', 'test'):
            if sp not in subs:
                issues.append(('ERROR', 'split_folder_missing', sp, fo, '', f'no "{sp}" folder in {fdir} '
                               f'(found: {sorted(subs)})'))
                continue
            idir, gdir = find_child(subs[sp], IMG_NAMES), find_child(subs[sp], GT_NAMES)
            xdir = find_child(subs[sp], INST_NAMES)
            if not idir or not gdir:
                issues.append(('ERROR', 'images_or_gt_missing', sp, fo, '',
                               f'images dir: {idir}, GT dir: {gdir} in {subs[sp]}'))
                continue
            im, gt, ins = list_stems(idir), list_stems(gdir), list_stems(xdir) if xdir else {}
            for s, fl in list(im.items()) + list(gt.items()):
                if len(fl) > 1:
                    issues.append(('ERROR', 'duplicate_stem', sp, fo, s, f'several files: {fl}'))
            for s in sorted(set(im) - set(gt)):
                issues.append(('ERROR', 'mask_missing', sp, fo, s, 'image has no mask'))
            for s in sorted(set(gt) - set(im)):
                issues.append(('ERROR', 'image_missing', sp, fo, s, 'mask has no image'))
            common = sorted(set(im) & set(gt))
            counts[(sp, fo)] = (len(im), len(gt), len(common))
            for s in common:
                jobs.append((sp, fo, s, os.path.join(idir, im[s][0]), os.path.join(gdir, gt[s][0]),
                             os.path.join(xdir, ins[s][0]) if s in ins else None, args))

    say('\nImages / masks / paired')
    fl = [f'fold{n}' for n in sorted(folds)]
    say(f'{"":<8}' + ''.join(f'{f:>16}' for f in fl))
    for sp in ('train', 'test'):
        say(f'{sp:<8}' + ''.join(
            f'{"missing" if (sp, f) not in counts else "/".join(map(str, counts[(sp, f)])):>16}' for f in fl))

    say(f'\nAnalysing {len(jobs)} image/mask pairs ...')
    with ThreadPoolExecutor(args.workers) as ex:
        recs = list(ex.map(analyse, jobs))
    paths = {(j[0], j[1], j[2]): (j[3], j[4]) for j in jobs}
    for r in recs:
        r['ip'], r['mp'] = paths[(r['split'], r['fold'], r['name'])]

    ok = [r['edge_gain'] for r in recs if r['edge_gain'] == r['edge_gain']]
    gain_lim = max(args.gain_min, 0.25 * float(np.median(ok))) if ok else args.gain_min
    for r in recs:
        if r['edge_gain'] == r['edge_gain'] and r['edge_gain'] < gain_lim:
            r['issues'].append(('WARN', 'weak_alignment', f'gain={r["edge_gain"]:.2f} (limit {gain_lim:.2f}); '
                                'check the overlay'))

    # ---- cross-fold logic
    by = defaultdict(list)
    for r in recs:
        by[(r['split'], r['fold'])].append(r)
    H = {k: {r['md5'] for r in v if r['md5']} for k, v in by.items()}
    nm = {k: {r['name'] for r in v} for k, v in by.items()}
    cross = []

    # 1. train/test of the same fold
    for f in fl:
        tr, te = by.get(('train', f), []), by.get(('test', f), [])
        if not tr or not te:
            continue
        inter = H[('train', f)] & H[('test', f)]
        if inter:
            names = sorted(r['name'] for r in te if r['md5'] in inter)
            cross.append(('ERROR', 'train_test_identical', 'test', f, '',
                          f'{len(inter)} test images are also in train, e.g. {names[:4]}'))
        same = nm[('train', f)] & nm[('test', f)]
        if same:
            cross.append(('ERROR', 'train_test_same_name', 'test', f, '',
                          f'{len(same)} file names occur in train and test, e.g. {sorted(same)[:4]}'))
        trh = np.stack([r['hashes'][0] for r in tr if r['hashes'] is not None])
        trn = [r['name'] for r in tr if r['hashes'] is not None]
        near = []
        for q in te:
            if q['hashes'] is None or q['md5'] in inter:
                continue
            d = POP[np.bitwise_xor(q['hashes'][:, None, :], trh[None, :, :])].sum(-1).min(axis=0)
            j = int(d.argmin())
            if d[j] <= args.hash_dist:
                near.append(f'{q["name"]}~{trn[j]}')
        if near:
            cross.append(('WARN', 'train_test_near_duplicate', 'test', f, '',
                          f'{len(near)} test images look like (flipped) train images, e.g. {near[:3]}'))

    # 2. test sets must not overlap
    for i, a in enumerate(fl):
        for b in fl[i + 1:]:
            if ('test', a) in H and ('test', b) in H:
                inter = H[('test', a)] & H[('test', b)]
                if inter:
                    cross.append(('ERROR', 'test_sets_overlap', 'test', f'{a}+{b}', '',
                                  f'{len(inter)} identical images in both test sets '
                                  f'({len(H[("test", a)])} and {len(H[("test", b)])} images)'))

    # 3. every image tested exactly once; train = other tests
    all_test = set().union(*[H[k] for k in H if k[0] == 'test']) if any(k[0] == 'test' for k in H) else set()
    all_train = set().union(*[H[k] for k in H if k[0] == 'train']) if any(k[0] == 'train' for k in H) else set()
    never = all_train - all_test
    if never:
        cross.append(('ERROR', 'image_never_tested', '', '', '',
                      f'{len(never)} different images appear in train somewhere but in no test set'))
    for f in fl:
        if ('train', f) not in H:
            continue
        exp = set().union(*[H[('test', g)] for g in fl if g != f and ('test', g) in H])
        miss, extra = exp - H[('train', f)], H[('train', f)] - exp
        if miss:
            cross.append(('ERROR', 'train_missing_images', 'train', f, '',
                          f'{len(miss)} images that belong to the other folds\' test sets are not in train'))
        if extra:
            cross.append(('ERROR', 'train_has_extra_images', 'train', f, '',
                          f'{len(extra)} train images are not in any other fold\'s test set '
                          '(for example because they are in this fold\'s test set)'))
    if args.expected_total and len(all_test | all_train) != args.expected_total:
        cross.append(('ERROR', 'total_image_count', '', '', '',
                      f'{len(all_test | all_train)} different images in the dataset, expected {args.expected_total}'))

    # 4. duplicates inside a split
    for k, v in by.items():
        c = Counter(r['md5'] for r in v if r['md5'])
        d = [m for m, n in c.items() if n > 1]
        if d:
            ex_names = [[r['name'] for r in v if r['md5'] == m] for m in d[:2]]
            cross.append(('WARN', 'duplicate_images_in_split', k[0], k[1], '',
                          f'{len(d)} images occur more than once, e.g. {ex_names}'))

    # 5. names that mean different images in different folds
    first = {}
    clash = 0
    for r in recs:
        if r['md5']:
            if r['name'] in first and first[r['name']] != r['md5']:
                clash += 1
            first.setdefault(r['name'], r['md5'])
    if clash:
        cross.append(('INFO', 'name_reused_for_other_image', '', '', '',
                      f'{clash} file names refer to different pictures in different folds (naming is per fold)'))

    # ---- summary
    all_issues = list(issues) + cross
    for r in recs:
        for lv, code, dt in r['issues']:
            all_issues.append((lv, code, r['split'], r['fold'], r['name'], dt))
    say('\n' + '=' * 78)
    say('PROBLEMS')
    say('=' * 78)
    groups = defaultdict(list)
    lvl = {}
    for lv, code, sp, fo, nme, dt in all_issues:
        groups[code].append((sp, fo, nme, dt))
        lvl[code] = lv
    if not groups:
        say('No problems found.')
    for code in sorted(groups, key=lambda c: (lvl[c] != 'ERROR', lvl[c] != 'WARN', c)):
        g = groups[code]
        say(f'[{lvl[code]}] {code}: {len(g)}')
        for sp, fo, nme, dt in g[:5]:
            say(f'      {sp}/{fo}/{nme}: {dt}' if nme else f'      {sp} {fo}: {dt}')

    say('\n' + '=' * 78)
    say('MEASUREMENTS')
    say('=' * 78)
    say(f'{"split/fold":<14}{"pairs":>7}{"fg% mean":>10}{"empty":>7}{"gain med":>10}{"weak %":>8}')
    for k in sorted(by, key=lambda x: (x[1], x[0])):
        v = by[k]
        fgv = np.array([r['fg_frac'] for r in v], float)
        g = np.array([r['edge_gain'] for r in v], float)
        g = g[~np.isnan(g)]
        say(f'{k[0] + "/" + k[1]:<14}{len(v):>7}{np.nanmean(fgv) * 100:>10.1f}{int((fgv == 0).sum()):>7}'
            f'{(np.median(g) if len(g) else float("nan")):>10.2f}{((g < gain_lim).mean() * 100 if len(g) else 0):>8.1f}')
    say(f'\nDifferent images in all test sets: {len(all_test)};  in all train sets: {len(all_train)};  '
        f'union: {len(all_test | all_train)}')
    say('Test sizes: ' + ', '.join(f'{f}: {len(H.get(("test", f), []))}' for f in fl))
    say('Image sizes: ' + str(dict(Counter((r['h'], r['w']) for r in recs if r['h']).most_common(5))))

    # ---- files
    with open(os.path.join(args.out, 'issues.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['level', 'code', 'split', 'fold', 'name', 'detail'])
        w.writerows(all_issues)
    with open(os.path.join(args.out, 'samples.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'fold', 'name', 'h', 'w', 'nuniq', 'fg_frac', 'n_comp', 'specks', 'holes',
                    'edge_ratio', 'edge_gain', 'contrast', 'md5'])
        for r in recs:
            w.writerow([r['split'], r['fold'], r['name'], r['h'], r['w'], r['nuniq'], r['fg_frac'], r['n_comp'],
                        r['specks'], r['holes'], r['edge_ratio'], r['edge_gain'], r['contrast'], r['md5']])
    with open(os.path.join(args.out, 'fold_membership.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['md5', 'name_in_test', 'test_fold', 'image_path', 'mask_path'])
        for r in recs:
            if r['split'] == 'test':
                w.writerow([r['md5'], r['name'], r['fold'], r['ip'], r['mp']])

    if not args.no_sheets:
        sd = os.path.join(args.out, 'sheets')
        os.makedirs(sd, exist_ok=True)
        rnd = random.Random(args.seed)
        for (sp, fo), v in sorted(by.items(), key=lambda x: (x[0][1], x[0][0])):
            sheet(rnd.sample(v, min(args.sheet_n, len(v))), os.path.join(sd, f'{fo}_{sp}_random.png'))
            worst = sorted([r for r in v if r['edge_gain'] == r['edge_gain']], key=lambda r: r['edge_gain'])
            sheet(worst[:args.sheet_n], os.path.join(sd, f'{fo}_{sp}_weakest_alignment.png'))
        say(f'\nOverlay sheets: {os.path.abspath(sd)}')

    ne = sum(1 for x in all_issues if x[0] == 'ERROR')
    nw = sum(1 for x in all_issues if x[0] == 'WARN')
    say('\n' + '=' * 78)
    say(f'RESULT: {ne} errors, {nw} warnings. ' + ('The folds are NOT usable as they are.' if ne else
                                                  'No blocking errors in the original folds.'))
    say('=' * 78)
    with open(os.path.join(args.out, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    sys.exit(1 if ne else 0)


if __name__ == '__main__':
    main()
