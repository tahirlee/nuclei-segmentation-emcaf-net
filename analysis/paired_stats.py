"""
Paired comparison of two runs from the per-image CSVs written by test.py.

Reports, for each metric:
  1. fold level      per-fold means of both runs, per-fold differences, the Nadeau-Bengio
                     corrected resampled t-test (variance x (1/k + n_test/n_train)) and its
                     95% CI, Holm-adjusted p-values across the metrics.
  2. image level     mean difference and win rate over all test images, with a CLUSTER
                     bootstrap interval. Whole clusters (organs, or WSIs when --groups maps
                     images to slides) are resampled with all of their image pairs, so the
                     pairing is preserved. The trained models are FIXED: the interval covers
                     test-sample variation only, NOT training variation (seed, split); that
                     is what the fold-level analysis and repeated seeds are for.
  3. cluster-macro   the same difference after averaging the images of each cluster first,
                     so that clusters with many images do not dominate (WSI-macro result).

Both runs must have been evaluated on the same images.

Example (NuInsSeg, file names as written by test.py):
  python analysis/paired_stats.py --a nucleinet --b dualdecoder \\
      --folds 1 2 3 4 5 --n_train 478 --n_test 133 \\
      --groups organ_map.csv --out stats_nucleinet_vs_dual
  # organ_map.csv: columns image,group (or wsi / organ); image = file stem

Per-fold file pattern: per_image_scores/<run>_fold<k>_lr3__test<suffix>.csv
Use --pattern to change it (placeholders {run} {k} {suffix}); for leave-one-organ-out
datasets (CryoNuSeg) pass the fold numbers 1..10 and --n_train 24 --n_test 3.
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy import stats


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def load_run(run, folds, pattern, scores_dir, suffix, metrics):
    per_fold = {}
    for k in folds:
        path = os.path.join(scores_dir, pattern.format(run=run, k=k, suffix=suffix))
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        rows = read_csv(path)
        per_fold[k] = {r['image']: {m: float(r[m]) for m in metrics} for r in rows}
    return per_fold


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def cluster_bootstrap(diffs, groups, n_boot, seed, macro):
    """diffs: per-image paired differences; groups: cluster label per image."""
    rng = np.random.default_rng(seed)
    labels = sorted(set(groups))
    idx = {g: np.array([i for i, x in enumerate(groups) if x == g]) for g in labels}
    sums = np.array([diffs[idx[g]].sum() for g in labels])
    cnt = np.array([len(idx[g]) for g in labels])
    means = sums / cnt
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(labels), len(labels))
        out[b] = means[pick].mean() if macro else sums[pick].sum() / cnt[pick].sum()
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', required=True, help='run name of the model under test')
    ap.add_argument('--b', required=True, help='run name of the comparator')
    ap.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--metrics', default='dice,aji,pq')
    ap.add_argument('--n_train', type=int, required=True, help='training images per fold')
    ap.add_argument('--n_test', type=int, required=True, help='test images per fold')
    ap.add_argument('--scores_dir', default='per_image_scores')
    ap.add_argument('--pattern', default='{run}_fold{k}_lr3__test{suffix}.csv')
    ap.add_argument('--suffix', default='', help='e.g. __md9 for a sweep file, or a --tag')
    ap.add_argument('--groups', default='',
                    help='CSV image,group for the cluster bootstrap / macro average')
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='paired_stats')
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(',')]
    A = load_run(args.a, args.folds, args.pattern, args.scores_dir, args.suffix, metrics)
    B = load_run(args.b, args.folds, args.pattern, args.scores_dir, args.suffix, metrics)
    gmap = {}
    if args.groups:
        grows = read_csv(args.groups)
        col = next((c for c in ('group', 'wsi', 'organ') if c in grows[0]), None)
        if col is None:
            raise KeyError('--groups needs a column named group, wsi or organ')
        gmap = {r['image']: r[col] for r in grows}

    k = len(args.folds)
    correction = 1.0 / k + args.n_test / args.n_train
    result = {'a': args.a, 'b': args.b, 'folds': args.folds, 'metrics': {}}
    lines = [f'# {args.a} vs {args.b}', '',
             f'Folds: {args.folds}; Nadeau-Bengio factor 1/k + n_test/n_train = '
             f'{correction:.4f}', '']

    fold_p = []
    for m in metrics:
        fa = np.array([np.mean([v[m] for v in A[f].values()]) for f in args.folds]) * 100
        fb = np.array([np.mean([v[m] for v in B[f].values()]) for f in args.folds]) * 100
        d = fa - fb
        se = np.sqrt(correction * d.var(ddof=1)) if k > 1 else float('nan')
        t = d.mean() / se if se > 0 else float('inf')
        p = float(2 * stats.t.sf(abs(t), k - 1))
        half = stats.t.ppf(0.975, k - 1) * se
        fold_p.append(p)
        result['metrics'][m] = dict(
            fold_a=fa.tolist(), fold_b=fb.tolist(), fold_diff=d.tolist(),
            mean_a=fa.mean(), sd_a=fa.std(ddof=1), mean_b=fb.mean(), sd_b=fb.std(ddof=1),
            mean_diff=d.mean(), ci_low=d.mean() - half, ci_high=d.mean() + half,
            t=t, p=p, folds_a_better=int((d > 0).sum()))

    for m, ph in zip(metrics, holm(np.array(fold_p))):
        result['metrics'][m]['p_holm'] = float(ph)

    lines += ['## Fold level (percentage points)', '',
              '| Metric | ' + ' | '.join(f'Fold {f}' for f in args.folds) +
              ' | Mean diff [95% CI] | p (NB) | p (Holm) | folds A>B |',
              '|---|' + '---|' * (k + 4)]
    for m in metrics:
        r = result['metrics'][m]
        lines.append(f'| {m} diff (A-B) | ' + ' | '.join(f'{x:+.2f}' for x in r['fold_diff']) +
                     f' | {r["mean_diff"]:+.2f} [{r["ci_low"]:+.2f}, {r["ci_high"]:+.2f}] | '
                     f'{r["p"]:.2g} | {r["p_holm"]:.2g} | {r["folds_a_better"]}/{k} |')
    lines += ['', '| Metric | A mean ± SD | B mean ± SD |', '|---|---|---|']
    for m in metrics:
        r = result['metrics'][m]
        lines.append(f'| {m} | {r["mean_a"]:.2f} ± {r["sd_a"]:.2f} | '
                     f'{r["mean_b"]:.2f} ± {r["sd_b"]:.2f} |')

    # image level
    names, groups = [], []
    for f in args.folds:
        common = sorted(set(A[f]) & set(B[f]))
        if len(common) != len(A[f]) or len(common) != len(B[f]):
            raise ValueError(f'fold {f}: the two runs were not evaluated on the same images')
        names += [(f, n) for n in common]
    if gmap:
        missing = [n for _, n in names if n not in gmap]
        if missing:
            raise KeyError(f'{len(missing)} image(s) missing from --groups, e.g. {missing[:3]}')
        groups = [gmap[n] for _, n in names]

    lines += ['', f'## Image level ({len(names)} test images, models fixed)', '',
              '| Metric | Mean diff | Win rate (A>B) |' +
              (' Cluster-bootstrap 95% CI (image-weighted) | Cluster-macro diff [95% CI] |'
               if gmap else ''),
              '|---|---|---|' + ('---|---|' if gmap else '')]
    for m in metrics:
        diff = np.array([A[f][n][m] - B[f][n][m] for f, n in names]) * 100
        row = dict(mean_diff=float(diff.mean()), win_rate=float((diff > 0).mean()))
        txt = f'| {m} | {diff.mean():+.2f} | {100 * row["win_rate"]:.1f}% |'
        if gmap:
            lo, hi = cluster_bootstrap(diff, groups, args.n_boot, args.seed, macro=False)
            glabels = sorted(set(groups))
            gm = np.array([diff[[i for i, g in enumerate(groups) if g == lab]].mean()
                           for lab in glabels])
            mlo, mhi = cluster_bootstrap(diff, groups, args.n_boot, args.seed, macro=True)
            row.update(ci_low=lo, ci_high=hi, macro_diff=float(gm.mean()),
                       macro_ci_low=mlo, macro_ci_high=mhi, n_clusters=len(glabels))
            txt += f' [{lo:+.2f}, {hi:+.2f}] | {gm.mean():+.2f} [{mlo:+.2f}, {mhi:+.2f}] |'
        result['metrics'][m]['image_level'] = row
        lines.append(txt)
    if gmap:
        lines += ['', f'Clusters resampled: {len(set(groups))}. Whole clusters are drawn '
                  f'with replacement ({args.n_boot} draws, seed {args.seed}) together with all '
                  'of their paired image scores; the trained models are not refitted.']

    with open(args.out + '.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(args.out + '.json', 'w') as f:
        json.dump(result, f, indent=1)
    print('\n'.join(lines))
    print(f'\nWritten: {args.out}.md / {args.out}.json')


if __name__ == '__main__':
    main()
