# -*- coding: utf-8 -*-
"""Sensitivity to the detection threshold t = 0.30-0.70.

Population pupil shapes for the standard uncertainty; a circular pupil for the bias
direction, which is only defined when the shape is fixed. Rendering grid OS = 5.

Writes results/threshold.json.
"""
import argparse
import json
import os
import time

import numpy as np
from multiprocessing import Pool, cpu_count

import iris_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))
OUT = os.path.join(RES, 'threshold.json')
VERS = np.array([0.0, 1.0, 2.0])
OS = 5
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
NOISE = [0.10, 0.30, 0.50]


def trial(task):
    seed, dens, sn, angle, thr, shape = task
    rng = np.random.default_rng(int(seed))
    amp, ph = ((np.zeros(7), np.zeros(7)) if shape == 'circ'
               else M.population_distortion(rng))
    block = M.REDUCE * OS
    dens_hr = dens * M.RES_MULT * OS
    bnd, ycR, zcR = M.build_pupil_boundary(VERS, angle, amp, ph)
    Yb, Zb = bnd[:, 0], bnd[:, 1]
    mX = M.MARGP * (Yb.max() - Yb.min())
    x0 = Yb.min() - mX * (rng.random() + 1.0)
    x1 = Yb.max() + mX * (rng.random() + 1.0)
    mY = M.MARGP * (Zb.max() - Zb.min())
    y0 = Zb.min() - mY * (rng.random() + 1.0)
    y1 = Zb.max() + mY * (rng.random() + 1.0)
    nx = max(int(np.floor(1e3 * dens_hr * (x1 - x0))), block)
    ny = max(int(np.floor(1e3 * dens_hr * (y1 - y0))), block)
    PX, PY = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    ins = M.points_in_shape_radial(PX.ravel(), PY.ravel(), Yb, Zb,
                                   ycR, zcR).reshape(PX.shape).astype(float)
    XT, YT, V = M._block_average(PX, PY, ins, block)
    V = V + rng.standard_normal(V.shape) * sn
    B = V >= thr
    k = B.sum()
    if k == 0:
        return (np.nan, np.nan)
    return (XT[B].mean() - ycR, YT[B].mean() - zcR)


def point(pool, rep, thr, sn, shape, seed):
    ss = np.random.SeedSequence(seed).spawn(rep)
    a = np.array(pool.map(trial, [(s.generate_state(1)[0], 1.0, sn, 0.0, thr, shape)
                                  for s in ss], chunksize=16))
    a = a[~np.isnan(a).any(axis=1)]
    if len(a) < 10:
        return None
    ex, ey = a[:, 0] * 1e6, a[:, 1] * 1e6
    return dict(n=int(len(a)),
                u=round(float(np.sqrt((ex.std(ddof=1) ** 2
                                       + ey.std(ddof=1) ** 2) / 2)), 1),
                bias=round(float(np.hypot(ex.mean(), ey.mean())), 1),
                phi=round(float(np.degrees(np.arctan2(ey.mean(), ex.mean()))), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rep', type=int, default=2000)
    ap.add_argument('-j', '--nproc', type=int, default=None)
    args = ap.parse_args()
    doc = dict(meta=dict(rep=args.rep, oversample=OS, r_mm=1.0, alpha=0.0))
    t0 = time.time()
    with Pool(processes=args.nproc or cpu_count()) as pool:
        for shape in ('pop', 'circ'):
            doc[shape] = {}
            for sn in NOISE:
                row = {}
                for thr in THRESHOLDS:
                    row['%.2f' % thr] = point(pool, args.rep, thr, sn, shape,
                                              9000 + int(1000 * sn) + int(100 * thr))
                    print('%s sn=%.2f t=%.2f  %s' % (shape, sn, thr,
                                                     row['%.2f' % thr]), flush=True)
                doc[shape]['%.2f' % sn] = row
                json.dump(doc, open(OUT, 'w'), indent=1)
    print('\n%.1f min -> %s' % ((time.time() - t0) / 60, OUT))


if __name__ == '__main__':
    main()
