# -*- coding: utf-8 -*-
"""Adaptive Monte Carlo procedure of JCGM 101:2008, clause 7.9.

  - carry out sequences of M trials each;
  - after each sequence h, form the average over sequences of the estimate y, the
    standard uncertainty u(y) and the endpoints of the 95 % coverage interval, and
    the standard deviations of those averages;
  - stop when twice each standard deviation is below the numerical tolerance delta,
    delta = 0.5 * 10^(floor(log10(u)) - n_dig + 1) for n_dig significant digits.

Both the probabilistically symmetric and the shortest coverage interval are reported
(JCGM 101, 7.7).

Usage:
    python run_jcgm101.py --sn 0.10 --dens 1.0
    python run_jcgm101.py --all              # the five reported operating points
"""
import argparse
import json
import os
import time

import numpy as np
from multiprocessing import Pool, cpu_count

import iris_model as M

VERS = np.array([0.0, 1.0, 2.0])
ANG = 0.0
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))

# the operating points tabulated as the expanded-uncertainty budget
POINTS = [(0.05, 1.0), (0.10, 1.0), (0.20, 1.0), (0.10, 2.0), (0.10, 0.5)]


def _trial(task):
    """Return the two axial errors about the true centre, in metres.

    The measurand is the pupil-centre coordinate along one image axis, so the two
    axes of a trial are two realisations of the same output quantity and are pooled.
    Averaging them instead would report the dispersion of their mean, which is
    smaller by a factor of sqrt(2) and is not what the budget states.
    """
    seed, dens, sn, oversample = task
    rng = np.random.default_rng(int(seed))
    da, dp = M.population_distortion(rng)
    ycR, zcR, xcM, ycM, xcMr, ycMr = M.single_trial(VERS, ANG, da, dp, dens, sn, rng,
                                                    oversample=oversample)
    if np.isnan(xcMr):
        return (np.nan, np.nan)
    return (xcMr - ycR, ycMr - zcR)


def coverage(v, p=0.95):
    """Probabilistically symmetric and shortest 100p % coverage intervals (JCGM 101 7.7)."""
    s = np.sort(v)
    n = s.size
    q = int(np.floor(n * (1 - p) / 2 + 0.5))
    sym = (s[q], s[q + int(np.floor(p * n + 0.5)) - 1])
    r = int(np.floor(p * n + 0.5))
    j = int(np.argmin(s[r - 1:] - s[:n - r + 1]))
    return sym, (s[j], s[j + r - 1])


def digits_tolerance(u, n_dig=2):
    return 0.5 * 10.0 ** (np.floor(np.log10(abs(u))) - n_dig + 1)


def adaptive(dens, sn, M_seq=8000, h_max=60, n_dig=2, seed=20260816, nproc=None,
             oversample=1):
    pool = Pool(processes=nproc or cpu_count())
    ss = np.random.SeedSequence(seed)
    ys, us, los, his = [], [], [], []
    t0 = time.time()
    discarded = 0
    total = 0
    try:
        for h in range(1, h_max + 1):
            seeds = [s.generate_state(1)[0] for s in ss.spawn(M_seq)]
            v = np.array(pool.map(_trial, [(sd, dens, sn, oversample) for sd in seeds],
                                 chunksize=256)).ravel()
            total += v.size // 2
            discarded += int(np.isnan(v).sum()) // 2
            v = v[~np.isnan(v)]
            sym, _ = coverage(v)
            ys.append(v.mean()); us.append(v.std(ddof=1))
            los.append(sym[0]); his.append(sym[1])
            if h < 2:
                continue
            sd = [np.std(a, ddof=1) / np.sqrt(h) for a in (ys, us, los, his)]
            delta = digits_tolerance(np.mean(us), n_dig)
            achieved = max(2 * x for x in sd)
            if achieved < delta:
                break
    finally:
        pool.close(); pool.join()

    achieved = locals().get('achieved', float('nan'))
    y, u = float(np.mean(ys)), float(np.mean(us))
    lo, hi = float(np.mean(los)), float(np.mean(his))
    return dict(sigma_n=sn, r_mm=float(M.sensor_pitch_mm(dens)),
                trials=total, sequences=h, seq_len=M_seq,
                y_um=y * 1e6, u_um=u * 1e6,
                cov_lo_um=lo * 1e6, cov_hi_um=hi * 1e6,
                cov_halfwidth_um=(hi - lo) / 2 * 1e6,
                k_effective=float((hi - lo) / 2 / u),
                U_k2_um=2 * u * 1e6,
                tolerance_um=digits_tolerance(u, n_dig) * 1e6,
                achieved_um=float(achieved) * 1e6,
                converged=bool(h < h_max),
                discard_rate=discarded / max(total, 1),
                minutes=(time.time() - t0) / 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sn', type=float)
    ap.add_argument('--dens', type=float)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--seq', type=int, default=8000)
    ap.add_argument('--oversample', type=int, default=5)
    ap.add_argument('--out', default=os.path.join(RESULTS, 'jcgm101.json'))
    args = ap.parse_args()

    pts = POINTS if args.all else [(args.sn, args.dens)]
    doc = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for sn, dens in pts:
        r = adaptive(dens, sn, M_seq=args.seq, oversample=args.oversample)
        key = 'sn%.2f_r%.3f_os%d' % (sn, r['r_mm'], args.oversample)
        doc[key] = r
        json.dump(doc, open(args.out, 'w'), indent=1)
        print('sn=%.2f r=%.3f | M=%d in %d seq | u=%.1f um | 95%% cov [%.0f, %.0f] '
              'halfwidth %.1f (k_eff=%.2f) | k2 gives %.1f | conv=%s | %.1f min'
              % (sn, r['r_mm'], r['trials'], r['sequences'], r['u_um'],
                 r['cov_lo_um'], r['cov_hi_um'], r['cov_halfwidth_um'],
                 r['k_effective'], r['U_k2_um'], r['converged'], r['minutes']))
    print('\nsaved', args.out)


if __name__ == '__main__':
    main()
