# -*- coding: utf-8 -*-
"""Design-space sweeps on the rendering grid, standalone and resumable.

Checkpoints after every work point, so it can be interrupted with Ctrl+C and restarted;
completed points are skipped. Needs only numpy and iris_model.py.

  noise      u(sigma_eff, alpha) at r = 1.00 mm/line             90 points
  pitch      u(r, alpha) at sigma_eff = 0.10                     63 points
  budget     sigma_within, sigma_between at alpha = 0             7 points
  fine       fine-pitch sweep at sigma_eff = 0.10                 7 points
  finenoise  sigma_within against noise at three fine pitches    18 points

Usage:
    python run_converged.py --estimate            # time the machine, print a forecast
    python run_converged.py                       # run everything, resuming if needed
    python run_converged.py --only noise budget          # run a subset

Output: results/converged_os<N>.json, N = --oversample (default 5). The grid
convergence study (gci.py) needs all three levels: --oversample 1, 3 and 5.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np

import iris_model as M

VERS = np.array([0.0, 1.0, 2.0])
ANGLES = list(range(-60, 61, 15))
SN_NOISE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
DENS_PITCH = [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]          # r = 3.33 ... 0.50 mm/line
SN_BUDGET = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
DENS_FINE = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]         # r = 0.50 ... 0.10 mm/line
DN_PITCH = [(0.50, 2.0), (0.25, 4.0), (0.10, 10.0)]
DN_SN = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]

RES = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', '..', 'results'))


# --------------------------------------------------------------- guard
def check_model():
    """Stop unless the pupil-shape model reproduces Wyatt's noncircularity.

    The model gives NC ~ 0.0176; amplitudes applied as absolute metres give NC ~ 0.30.
    """
    rng = np.random.default_rng(0)
    nc = np.mean([M.noncircularity(M.population_distortion(rng)[0]) for _ in range(2000)])
    if not (0.010 < nc < 0.030):
        sys.exit('STOP: iris_model.py generates pupils with noncircularity %.4f.\n'
                 'Expected about 0.0176 (Wyatt 1995); check population_distortion().'
                 % nc)
    print('model check: NC = %.4f (Wyatt: 0.0176), ok' % nc)
    return float(nc)


# --------------------------------------------------------------- worker
def _trial(task):
    seed, dens, sn, angle, oversample = task
    rng = np.random.default_rng(int(seed))
    da, dp = M.population_distortion(rng)
    ycR, zcR, xcM, ycM, xcMr, ycMr = M.single_trial(
        VERS, angle, da, dp, dens, sn, rng, oversample=oversample)
    if np.isnan(xcMr):
        return (np.nan,) * 6
    return (ycR, zcR, xcM, ycM, xcMr, ycMr)


def point(pool, rep, dens, sn, angle, oversample, seed):
    """One operating point. Returns the summary the tables need."""
    ss = np.random.SeedSequence(seed).spawn(rep)
    tasks = [(s.generate_state(1)[0], dens, sn, angle, oversample) for s in ss]
    res = np.array(pool.map(_trial, tasks, chunksize=32))
    ok = ~np.isnan(res).any(axis=1)
    g = res[ok]
    if len(g) < 10:
        return dict(n=int(ok.sum()), discard=float(1 - ok.mean()), s=None)
    tx, ty, hx, hy, nx, ny = (g[:, i] for i in range(6))
    s = float(np.sqrt((nx.std(ddof=1) ** 2 + ny.std(ddof=1) ** 2) / 2) * 1e6)
    sb = float(np.sqrt((hx.std(ddof=1) ** 2 + hy.std(ddof=1) ** 2) / 2) * 1e6)
    bias = float(np.hypot(nx.mean() - tx.mean(), ny.mean() - ty.mean()) * 1e6)
    return dict(n=int(len(g)), discard=float(1 - ok.mean()), s=round(s, 2),
                sigma_between=round(sb, 2),
                sigma_within=round(float(np.sqrt(max(s * s - sb * sb, 0.0))), 2),
                bias=round(bias, 2), rmse=round(float(np.hypot(bias, s)), 2))


# --------------------------------------------------------------- job list
def jobs(rep, oversample):
    J = []
    for i, sn in enumerate(SN_NOISE):
        for j, a in enumerate(ANGLES):
            J.append(('noise', 'sn%.2f_a%+03d' % (sn, a), 1.0, sn, float(a), rep, 1000 + 17 * i + j))
    for i, d in enumerate(DENS_PITCH):
        for j, a in enumerate(ANGLES):
            J.append(('pitch', 'r%.3f_a%+03d' % (1.0 / d, a), d, 0.10, float(a), rep, 3000 + 23 * i + j))
    for i, sn in enumerate(SN_BUDGET):
        J.append(('budget', 'sn%.2f' % sn, 1.0, sn, 0.0, rep, 5000 + i))
    for i, d in enumerate(DENS_FINE):
        J.append(('fine', 'r%.3f' % (1.0 / d), d, 0.10, 0.0, rep, 6000 + i))
    for i, (r_mm, d) in enumerate(DN_PITCH):
        for j, sn in enumerate(DN_SN):
            J.append(('finenoise', 'r%.2f_sn%.3f' % (r_mm, sn), d, sn, 0.0, rep, 7000 + 13 * i + j))
    return J


def fmt(sec):
    h, rem = divmod(int(sec), 3600)
    return '%dh %02dm' % (h, rem // 60) if h else '%dm %02ds' % (rem // 60, rem % 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rep', type=int, default=1800,
                    help='trials per operating point (default 1800, type-A 1.7%%)')
    ap.add_argument('--oversample', type=int, default=5)
    ap.add_argument('--only', nargs='*', default=None, choices=['noise', 'pitch', 'budget', 'fine', 'finenoise'])
    ap.add_argument('--max-dens', type=float, default=4.0,
                    help='skip pitches finer than r = 1/max_dens mm/line (default 4.0, '
                         'i.e. r >= 0.25). Measured: refining the grid changes s by '
                         '+4.9%% at r = 0.50 but by -1.0%% at r = 0.333 and -1.8%% at '
                         'r = 0.25, both within the statistical noise, so finer pitches '
                         'gain nothing from the refinement and cost most of the time.')
    ap.add_argument('--out', default=None,
                    help='output JSON (default results/converged_os<oversample>.json)')
    ap.add_argument('--estimate', action='store_true',
                    help='time a few points and forecast the total, then exit')
    ap.add_argument('-j', '--nproc', type=int, default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(RES, 'converged_os%d.json' % args.oversample)
    nc = check_model()
    J = jobs(args.rep, args.oversample)
    if args.only:
        J = [j for j in J if j[0] in args.only]
    if args.max_dens:
        n0 = len(J)
        J = [j for j in J if j[2] <= args.max_dens + 1e-9]
        if len(J) < n0:
            print('skipping %d points finer than r = %.3f mm/line (see --help)'
                  % (n0 - len(J), 1.0 / args.max_dens))

    doc = json.load(open(args.out)) if os.path.exists(args.out) else {}
    doc.setdefault('meta', {}).update(
        rep=args.rep, oversample=args.oversample, noncircularity=nc,
        numpy=np.__version__, cores=args.nproc or cpu_count())
    done = {k for k in doc if k != 'meta'}
    todo = [j for j in J if '%s/%s' % (j[0], j[1]) not in done]
    print('%d work points, %d already done, %d to do'
          % (len(J), len(J) - len(todo), len(todo)))
    if not todo:
        print('nothing to do')
        return

    with Pool(processes=args.nproc or cpu_count()) as pool:
        if args.estimate:
            # Cost varies by two orders of magnitude across the studies, because the
            # rendering grid grows as the pitch is refined. Probe the cheapest and the
            # dearest point of each study and interpolate, rather than averaging over
            # everything, which would be dominated by whichever point happened to be
            # sampled.
            tot = 0.0
            print('%-4s %6s %14s %14s' % ('', 'points', 'per point', 'study total'))
            for st in ['noise', 'pitch', 'budget', 'fine', 'finenoise']:
                sub = [j for j in todo if j[0] == st]
                if not sub:
                    continue
                probes = {sub[0], sub[len(sub) // 2], sub[-1]}
                per = []
                for study, key, d, sn, a, rep, seed in probes:
                    t = time.time()
                    point(pool, max(rep // 30, 60), d, sn, a, args.oversample, seed)
                    per.append((time.time() - t) * 30)
                mean = sum(per) / len(per)
                sub_tot = mean * len(sub)
                tot += sub_tot
                print('%-4s %6d %14s %14s   (%s ... %s per point)'
                      % (st, len(sub), fmt(mean), fmt(sub_tot),
                         fmt(min(per)), fmt(max(per))))
            print('%-4s %6d %14s %14s' % ('all', len(todo), '', fmt(tot)))
            print('\nCheckpointing is per point, so this can be run in several sittings.')
            return

        t0 = time.time()
        for i, (study, key, d, sn, a, rep, seed) in enumerate(todo, 1):
            t = time.time()
            doc['%s/%s' % (study, key)] = dict(
                study=study, r_mm=round(1.0 / d, 4), sigma_n=sn, alpha=a,
                **point(pool, rep, d, sn, a, args.oversample, seed))
            json.dump(doc, open(args.out, 'w'), indent=1)      # checkpoint every point
            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            print('[%3d/%3d] %-3s %-16s s=%8s  %5.1fs   elapsed %s, eta %s'
                  % (i, len(todo), study, key,
                     doc['%s/%s' % (study, key)].get('s'), time.time() - t,
                     fmt(el), fmt(eta)), flush=True)

    print('\ndone in %s -> %s' % (fmt(time.time() - t0), os.path.abspath(args.out)))


if __name__ == '__main__':
    main()
