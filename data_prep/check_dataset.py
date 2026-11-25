r"""
check_dataset.py  -  sanity checks for the EMCAF-Net dataset folders

Expected layout (what your train.py / test_instance.py already use):
    dataset/
        train_aug/fold1/images, GT          (offline-augmented training data)
        val/fold1/images, GT
        test/fold1/images, GT, GT_instance  (GT_instance optional)
        ... fold2 ... fold5

Usage (from the project folder):
    python data_prep\check_dataset.py --dataset_path .\dataset
    python data_prep\check_dataset.py --dataset_path .\dataset --folds 1 2 --no_sheets

Output goes to  dataset_check\  :
    report.txt        everything printed on screen
    issues.csv        one row per problem (level, code, file, detail)
    samples.csv       one row per image/mask pair with all measured values
    sheets\*.png      overlay contact sheets (red = mask boundary on the image)

What is checked
  1. Pairing        every image has exactly one mask (same file stem) and vice versa
  2. Readability    files open, image/mask have the same height and width
  3. Mask values    binary only ({0,1} or {0,255}); intermediate grey values come from
                    bilinear/bicubic resizing or rotation of masks and are an ERROR
  4. Label noise    isolated 1-3 pixel specks / holes in the mask (salt-and-pepper)
  5. Empty / full   masks with no nuclei, or almost all foreground
  6. Alignment      mean image gradient on the mask boundary / mean gradient overall.
                    Aligned masks give clearly > 1; a flipped/shifted mask gives ~ 1. Also divided by the
                    same value for a displaced copy of the mask ("gain"), which calibrates itself.
                    The distribution of train_aug is compared with test (original data).
  7. Image checks   constant or nearly black/white images
  8. Leakage        exact duplicates, flipped near-duplicates and file-name overlap
                    between train_aug <-> val <-> test; test overlap across folds
  9. GT_instance    (if present) instance mask foreground equals binary GT foreground
"""
import argparse
import csv
import hashlib
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
GRAY_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# ───────────────────────────── helpers ─────────────────────────────
def list_stems(d):
    out = defaultdict(list)
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            s, e = os.path.splitext(f)
            if e.lower() in EXTS:
                out[s].append(f)
    return out


def dhash(gray_u8):
    im = Image.fromarray(gray_u8).resize((9, 8), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel())  # 8 bytes


def to_2d(arr):
    if arr.ndim == 3:
        return arr[..., 0], not (arr.shape[2] >= 3 and
                                 np.array_equal(arr[..., 0], arr[..., 1]) and
                                 np.array_equal(arr[..., 0], arr[..., 2]))
    return arr, False


# ───────────────────────────── per-pair analysis ─────────────────────────────
def analyse(job):
    split, fold, stem, ip, mp, inst_p, a = job
    rec = dict(split=split, fold=fold, name=stem, img_file=os.path.basename(ip),
               mask_file=os.path.basename(mp), issues=[], h=0, w=0, img_mode='', mask_mode='',
               nuniq=0, fg_frac=np.nan, n_comp=0, specks=0, holes=0, edge_ratio=np.nan, edge_gain=np.nan,
               contrast=np.nan, md5='', hashes=None, n_inst=0)

    def add(level, code, detail=''):
        rec['issues'].append((level, code, detail))

    try:
        im = Image.open(ip)
        rec['img_mode'] = im.mode
        im.load()
        rgb = np.asarray(im.convert('RGB'))
    except Exception as e:
        add('ERROR', 'image_unreadable', str(e))
        return rec
    try:
        m = Image.open(mp)
        rec['mask_mode'] = m.mode
        m.load()
        marr = np.asarray(m)
    except Exception as e:
        add('ERROR', 'mask_unreadable', str(e))
        return rec

    marr, colour = to_2d(marr)
    if colour:
        add('WARN', 'mask_colour', 'mask has different R/G/B channels (taking channel 0)')
    rec['h'], rec['w'] = rgb.shape[:2]
    if rgb.shape[:2] != marr.shape[:2]:
        add('ERROR', 'size_mismatch', f'image {rgb.shape[:2]} vs mask {marr.shape[:2]}')
        return rec

    # ---- mask values
    vals = np.unique(marr)
    rec['nuniq'] = len(vals)
    top = int(vals.max())
    fg = marr > 0
    if len(vals) > 2:
        mid = ((marr > 0) & (marr < top)).mean()
        add('ERROR', 'mask_not_binary',
            f'{len(vals)} unique values, {mid * 100:.2f}% intermediate pixels (max={top})')
    elif len(vals) == 2 and top not in (1, 255):
        add('WARN', 'mask_odd_value', f'foreground value is {top}, expected 1 or 255')
    frac = float(fg.mean())
    rec['fg_frac'] = frac
    if frac == 0:
        add('WARN', 'mask_empty', 'no foreground pixels')
    elif frac > 0.9:
        add('WARN', 'mask_almost_full', f'{frac * 100:.1f}% foreground')

    # ---- speckle (salt-and-pepper) and component statistics
    if frac > 0:
        lab, n = ndi.label(fg)
        sizes = np.bincount(lab.ravel())[1:]
        rec['n_comp'] = int(n)
        rec['specks'] = int((sizes <= 3).sum())
        lab_bg, nb = ndi.label(~fg)
        if nb:
            sb = np.bincount(lab_bg.ravel())[1:]
            rec['holes'] = int((sb <= 3).sum())
        if rec['specks'] > 20 or rec['holes'] > 20:
            add('WARN', 'mask_speckle',
                f'{rec["specks"]} foreground specks and {rec["holes"]} background holes of <= 3 px '
                '(salt-and-pepper noise on the label?)')

    # ---- image checks
    if rgb.std() < 2.0:
        add('WARN', 'image_constant', f'std={rgb.std():.2f}')
    elif rgb.mean() < 10 or rgb.mean() > 245:
        add('WARN', 'image_extreme', f'mean intensity {rgb.mean():.1f}')

    # ---- alignment (edge agreement) and contrast
    gray = rgb.astype(np.float32) @ GRAY_W
    if 0 < frac < 1:
        g = ndi.gaussian_filter(gray, 1.0)
        gy, gx = np.gradient(g)
        gm = np.hypot(gx, gy)
        boundary = fg & ~ndi.binary_erosion(fg)
        if boundary.sum() >= 20 and gm.mean() > 1e-6:
            rec['edge_ratio'] = float(gm[boundary].mean() / gm.mean())
            shifted = np.roll(boundary, (19, 27), axis=(0, 1))   # same boundary shape, wrong place
            rec['edge_gain'] = rec['edge_ratio'] / max(float(gm[shifted].mean() / gm.mean()), 1e-6)
        if fg.sum() >= 50 and (~fg).sum() >= 50:
            rec['contrast'] = float(gray[fg].mean() - gray[~fg].mean())

    # ---- hashes for leakage checks (original + flips)
    g8 = np.clip(gray, 0, 255).astype(np.uint8)
    rec['md5'] = hashlib.md5(rgb.tobytes()).hexdigest()
    rec['hashes'] = np.stack([dhash(g8), dhash(g8[:, ::-1]), dhash(g8[::-1]), dhash(g8[::-1, ::-1])])

    # ---- instance GT consistency
    if inst_p:
        try:
            inst, _ = to_2d(np.asarray(Image.open(inst_p)))
            if inst.shape != fg.shape:
                add('ERROR', 'instance_size_mismatch', f'{inst.shape} vs {fg.shape}')
            else:
                diff = float(((inst > 0) != fg).mean())
                rec['n_inst'] = int(len(np.unique(inst)) - (1 if (inst == 0).any() else 0))
                if diff > 0.02:
                    add('ERROR', 'instance_binary_mismatch', f'{diff * 100:.2f}% of pixels differ')
                elif diff > 0.001:
                    add('WARN', 'instance_binary_mismatch', f'{diff * 100:.3f}% of pixels differ')
        except Exception as e:
            add('ERROR', 'instance_unreadable', str(e))
    return rec


# ───────────────────────────── contact sheets ─────────────────────────────
def overlay_tile(rec, root, tile=256):
    ip = os.path.join(root, rec['split'], rec['fold'], 'images', rec['img_file'])
    mp = os.path.join(root, rec['split'], rec['fold'], 'GT', rec['mask_file'])
    img = Image.open(ip).convert('RGB').resize((tile, tile), Image.BILINEAR)
    m = Image.open(mp)
    m = np.asarray(m)
    m = m[..., 0] if m.ndim == 3 else m
    m = np.asarray(Image.fromarray((m > 0).astype(np.uint8) * 255).resize((tile, tile), Image.NEAREST)) > 0
    edge = m & ~ndi.binary_erosion(m)
    a = np.asarray(img).copy()
    a[edge] = (255, 0, 0)
    t = Image.fromarray(a)
    d = ImageDraw.Draw(t)
    label = rec['name'][-34:]
    er = rec['edge_gain']
    d.rectangle([0, 0, tile, 22], fill=(0, 0, 0))
    d.text((3, 2), label, fill=(255, 255, 255))
    d.text((3, 11), f'gain={er:.2f}' if er == er else 'gain=n/a', fill=(255, 255, 0))
    return t


def save_sheet(recs, root, path, cols=4, tile=256):
    if not recs:
        return
    rows = (len(recs) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tile, rows * tile), (30, 30, 30))
    for i, r in enumerate(recs):
        try:
            sheet.paste(overlay_tile(r, root, tile), ((i % cols) * tile, (i // cols) * tile))
        except Exception:
            pass
    sheet.save(path)


# ───────────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', default='./dataset')
    ap.add_argument('--splits', nargs='+', default=['train_aug', 'val', 'test'])
    ap.add_argument('--folds', nargs='*', type=int, default=None, help='default: all fold folders found')
    ap.add_argument('--out', default='dataset_check')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--edge_min', type=float, default=1.15,
                    help='edge_ratio below this is reported as weak alignment (WARN)')
    ap.add_argument('--gain_min', type=float, default=1.3,
                    help='edge_ratio divided by the same measure for a displaced mask; ~1 means no alignment')
    ap.add_argument('--hash_dist', type=int, default=3, help='max dHash bit distance for near-duplicates')
    ap.add_argument('--sheet_n', type=int, default=16)
    ap.add_argument('--no_sheets', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    root = args.dataset_path
    os.makedirs(args.out, exist_ok=True)
    lines = []

    def say(s=''):
        print(s)
        lines.append(s)

    # ---- discover folds
    found = set()
    for sp in args.splits:
        d = os.path.join(root, sp)
        if os.path.isdir(d):
            found |= {x for x in os.listdir(d) if re.fullmatch(r'fold\d+', x)}
    folds = [f'fold{n}' for n in args.folds] if args.folds else sorted(found, key=lambda s: int(s[4:]))
    if not folds:
        say(f'No fold folders found under {root}/{{{",".join(args.splits)}}}. Check --dataset_path.')
        sys.exit(2)
    say(f'Dataset: {os.path.abspath(root)}')
    say(f'Splits: {args.splits}   Folds: {folds}')

    # ---- pairing
    jobs, pair_issues, counts = [], [], {}
    for sp in args.splits:
        for fo in folds:
            idir = os.path.join(root, sp, fo, 'images')
            gdir = os.path.join(root, sp, fo, 'GT')
            xdir = os.path.join(root, sp, fo, 'GT_instance')
            if not os.path.isdir(idir) or not os.path.isdir(gdir):
                pair_issues.append(('ERROR', 'folder_missing', sp, fo, '', f'{idir} or {gdir} not found'))
                continue
            im, gt, ins = list_stems(idir), list_stems(gdir), list_stems(xdir)
            for s, fl in list(im.items()) + list(gt.items()):
                if len(fl) > 1:
                    pair_issues.append(('ERROR', 'duplicate_stem', sp, fo, s, f'several files: {fl}'))
            for s in sorted(set(im) - set(gt)):
                pair_issues.append(('ERROR', 'mask_missing', sp, fo, s, 'image has no mask'))
            for s in sorted(set(gt) - set(im)):
                pair_issues.append(('ERROR', 'image_missing', sp, fo, s, 'mask has no image'))
            common = sorted(set(im) & set(gt))
            counts[(sp, fo)] = (len(im), len(gt), len(common))
            for s in common:
                ipth = os.path.join(idir, im[s][0])
                mpth = os.path.join(gdir, gt[s][0])
                inst = os.path.join(xdir, ins[s][0]) if s in ins else None
                if ins and s not in ins:
                    pair_issues.append(('WARN', 'instance_missing', sp, fo, s, 'no GT_instance file'))
                jobs.append((sp, fo, s, ipth, mpth, inst, args))

    say('\nImages / masks / paired per split and fold')
    say(f'{"split":<10}' + ''.join(f'{f:>16}' for f in folds))
    for sp in args.splits:
        row = f'{sp:<10}'
        for fo in folds:
            c = counts.get((sp, fo))
            row += f'{"missing":>16}' if c is None else f'{f"{c[0]}/{c[1]}/{c[2]}":>16}'
        say(row)

    # ---- analyse everything
    say(f'\nAnalysing {len(jobs)} image/mask pairs ...')
    recs = []
    with ThreadPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(analyse, jobs), 1):
            recs.append(r)
            if i % 500 == 0:
                print(f'  {i}/{len(jobs)}', flush=True)

    # weak alignment flag (needs thresholds, done after measurement)
    ref = [r['edge_gain'] for r in recs if r['split'] in ('val', 'test') and r['edge_gain'] == r['edge_gain']]
    ref = ref or [r['edge_gain'] for r in recs if r['edge_gain'] == r['edge_gain']]
    gain_lim = max(args.gain_min, 0.25 * float(np.median(ref))) if ref else args.gain_min
    say(f'Alignment limit: gain < {gain_lim:.2f} (max of --gain_min and 0.25 x median of val/test)')
    for r in recs:
        er, eg = r['edge_ratio'], r['edge_gain']
        if er == er and (er < args.edge_min or eg < gain_lim):
            r['issues'].append(('WARN', 'weak_alignment',
                                f'edge_ratio={er:.2f}, gain over shifted mask={eg:.2f} '
                                f'(limits {args.edge_min} / {gain_lim:.2f}); check the overlay, '
                                'mask may be shifted/flipped or nuclei are very faint'))

    # ---- leakage
    leak = []
    by = defaultdict(list)
    for r in recs:
        by[(r['split'], r['fold'])].append(r)
    for fo in folds:
        tr = by.get(('train_aug', fo), [])
        if not tr:
            continue
        tr_md5 = defaultdict(list)
        for r in tr:
            tr_md5[r['md5']].append(r['name'])
        tr_h = np.stack([r['hashes'][0] for r in tr if r['hashes'] is not None]) if tr else None
        tr_names = [r['name'] for r in tr if r['hashes'] is not None]
        tr_stems = [r['name'] for r in tr]
        for other in ('val', 'test'):
            for q in by.get((other, fo), []):
                if q['hashes'] is None:
                    continue
                if q['md5'] in tr_md5:
                    leak.append(('ERROR', 'leak_exact', other, fo, q['name'],
                                 f'identical image in train_aug: {tr_md5[q["md5"]][:3]}'))
                    continue
                dist = POP[np.bitwise_xor(q['hashes'][:, None, :], tr_h[None, :, :])].sum(-1)  # (4,M)
                dmin = dist.min(axis=0)
                hit = np.where(dmin <= args.hash_dist)[0]
                if len(hit):
                    leak.append(('WARN', 'leak_near_duplicate', other, fo, q['name'],
                                 f'looks like (flip of) train_aug image {tr_names[hit[0]]} '
                                 f'(dHash distance {int(dmin[hit[0]])}); {len(hit)} candidate(s)'))
                pat = re.compile(r'(^|[^A-Za-z0-9])' + re.escape(q['name']) + r'([^A-Za-z0-9]|$)')
                nm = [t for t in tr_stems if t != q['name'] and pat.search(t)]
                if q['name'] in tr_stems:
                    leak.append(('ERROR', 'leak_same_name', other, fo, q['name'],
                                 'same file name exists in train_aug'))
                elif nm:
                    leak.append(('ERROR', 'leak_name_overlap', other, fo, q['name'],
                                 f'train_aug names contain this name, e.g. {nm[:3]} '
                                 '(augmented copies of a val/test image?)'))
        v = {r['md5']: r['name'] for r in by.get(('val', fo), [])}
        for q in by.get(('test', fo), []):
            if q['md5'] in v:
                leak.append(('ERROR', 'leak_val_test', 'test', fo, q['name'], f'identical to val image {v[q["md5"]]}'))
    # test images repeated in several folds (information only if the folds should be disjoint)
    tmap = defaultdict(set)
    for r in recs:
        if r['split'] == 'test':
            tmap[r['md5']].add(r['fold'])
    rep = sum(1 for v in tmap.values() if len(v) > 1)
    if rep:
        leak.append(('INFO', 'test_in_several_folds', 'test', '', '',
                     f'{rep} test images appear in more than one fold (expected if folds overlap, '
                     'a problem if they should be disjoint)'))

    # ---- collect issues
    all_issues = []
    for lv, code, sp, fo, nm, dt in pair_issues + leak:
        all_issues.append((lv, code, sp, fo, nm, dt))
    for r in recs:
        for lv, code, dt in r['issues']:
            all_issues.append((lv, code, r['split'], r['fold'], r['name'], dt))

    # ---- print summary
    say('\n' + '=' * 78)
    say('PROBLEMS BY TYPE (split / count)  -  full list in issues.csv')
    say('=' * 78)
    codes = defaultdict(lambda: defaultdict(list))
    level_of = {}
    for lv, code, sp, fo, nm, dt in all_issues:
        codes[code][sp].append((fo, nm, dt))
        level_of[code] = lv
    if not codes:
        say('No problems found.')
    for code in sorted(codes, key=lambda c: (level_of[c] != 'ERROR', c)):
        per = ', '.join(f'{sp}: {len(v)}' for sp, v in codes[code].items())
        say(f'[{level_of[code]}] {code}  ->  {per}')
        shown = 0
        for sp, v in codes[code].items():
            for fo, nm, dt in v[:2]:
                if shown < 4:
                    say(f'      {sp}/{fo}/{nm}: {dt}')
                    shown += 1

    say('\n' + '=' * 78)
    say('MEASUREMENTS PER SPLIT (all folds together)')
    say('=' * 78)
    say(f'{"split":<10}{"pairs":>7}{"fg%mean":>9}{"empty":>7}{"comp/img":>10}{"gain med":>10}{"gain p5":>9}'
        f'{"weak %":>12}{"contrast med":>14}')
    for sp in args.splits:
        rs = [r for r in recs if r['split'] == sp]
        if not rs:
            continue
        fgv = np.array([r['fg_frac'] for r in rs], dtype=float)
        ed = np.array([r['edge_gain'] for r in rs], dtype=float)
        ct = np.array([r['contrast'] for r in rs], dtype=float)
        nc = np.array([r['n_comp'] for r in rs], dtype=float)
        edv = ed[~np.isnan(ed)]
        say(f'{sp:<10}{len(rs):>7}{np.nanmean(fgv) * 100:>9.1f}{int((fgv == 0).sum()):>7}{nc.mean():>10.1f}'
            f'{np.median(edv) if len(edv) else float("nan"):>10.2f}'
            f'{np.percentile(edv, 5) if len(edv) else float("nan"):>9.2f}'
            f'{(edv < gain_lim).mean() * 100 if len(edv) else float("nan"):>12.1f}'
            f'{np.nanmedian(ct):>14.1f}')
    say('\nHow to read this: train_aug should look like test in foreground %, components per image,')
    say('edge agreement (gain) and contrast. A clearly lower gain median or a larger "weak %" in train_aug')
    say('means some augmented masks are not aligned with their images.')
    sizes = Counter((r['h'], r['w']) for r in recs if r['h'])
    say('\nImage sizes: ' + ', '.join(f'{k[0]}x{k[1]}: {v}' for k, v in sizes.most_common(6)))
    say('Image modes: ' + str(dict(Counter(r['img_mode'] for r in recs))) +
        '   Mask modes: ' + str(dict(Counter(r['mask_mode'] for r in recs))))
    for sp in args.splits:
        ex = [r['name'] for r in recs if r['split'] == sp][:3]
        if ex:
            say(f'Example names in {sp}: {ex}')

    # ---- files
    with open(os.path.join(args.out, 'issues.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['level', 'code', 'split', 'fold', 'name', 'detail'])
        w.writerows(all_issues)
    with open(os.path.join(args.out, 'samples.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'fold', 'name', 'h', 'w', 'img_mode', 'mask_mode', 'nuniq', 'fg_frac',
                    'n_comp', 'specks', 'holes', 'edge_ratio', 'edge_gain', 'contrast', 'n_inst'])
        for r in recs:
            w.writerow([r['split'], r['fold'], r['name'], r['h'], r['w'], r['img_mode'], r['mask_mode'],
                        r['nuniq'], r['fg_frac'], r['n_comp'], r['specks'], r['holes'], r['edge_ratio'],
                        r['edge_gain'], r['contrast'], r['n_inst']])

    if not args.no_sheets:
        sd = os.path.join(args.out, 'sheets')
        os.makedirs(sd, exist_ok=True)
        rnd = random.Random(args.seed)
        for sp in args.splits:
            for fo in folds:
                rs = by.get((sp, fo), [])
                if not rs:
                    continue
                save_sheet(rnd.sample(rs, min(args.sheet_n, len(rs))), root,
                           os.path.join(sd, f'{sp}_{fo}_random.png'))
                worst = sorted([r for r in rs if r['edge_ratio'] == r['edge_ratio']],
                               key=lambda r: r['edge_gain'])[:args.sheet_n]
                save_sheet(worst, root, os.path.join(sd, f'{sp}_{fo}_weakest_alignment.png'))
        say(f'\nOverlay sheets: {os.path.abspath(sd)}  (red line = mask boundary)')
        say('Open the *_weakest_alignment.png sheets first, then a few *_random.png of train_aug.')

    n_err = sum(1 for x in all_issues if x[0] == 'ERROR')
    n_warn = sum(1 for x in all_issues if x[0] == 'WARN')
    say('\n' + '=' * 78)
    say(f'RESULT: {n_err} errors, {n_warn} warnings.  ' +
        ('Fix the errors before training.' if n_err else 'No blocking errors.'))
    say('=' * 78)
    with open(os.path.join(args.out, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    sys.exit(1 if n_err else 0)


if __name__ == '__main__':
    main()
