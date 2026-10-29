# -*- coding: utf-8 -*-
"""Threshold bias of a circular pupil at a fixed and at a randomised subpixel phase.

A pixel-locking offset would average to zero when the subpixel phase of the pupil
against the sensor grid is randomised from frame to frame; the threshold bias does
not. The mean bias vector of a circular pupil (alpha = 0, r = 1.00 mm/line, 2000 trials
per point, rendering grid OS = 5) is computed with the analysis-window margins frozen
(fixed phase) and drawn per trial (random phase), at four noise levels.

Run from code/python/:
    python phase_test.py            # minutes; writes ../../results/phase_test.json
"""
import json
import os

import numpy as np
from multiprocessing import Pool, cpu_count

import iris_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))
VERS = np.array([0.0, 1.0, 2.0])
OS = 5
REP = 2000
NOISE = (0.20, 0.30, 0.40, 0.50)


def _trial(task):
    seed, sn, phase = task
    rng = np.random.default_rng(int(seed))
    block = M.REDUCE * OS
    dens_hr = 1.0 * M.RES_MULT * OS
    bnd, ycR, zcR = M.build_pupil_boundary(VERS, 0.0, np.zeros(7), np.zeros(7))
    Yb, Zb = bnd[:, 0], bnd[:, 1]
    mX = M.MARGP * (Yb.max() - Yb.min())
    mY = M.MARGP * (Zb.max() - Zb.min())
    if phase == 'fixed':
        x0, x1 = Yb.min() - 1.5 * mX, Yb.max() + 1.5 * mX
        y0, y1 = Zb.min() - 1.5 * mY, Zb.max() + 1.5 * mY
    else:
        x0 = Yb.min() - mX * (rng.random() + 1.0)
        x1 = Yb.max() + mX * (rng.random() + 1.0)
        y0 = Zb.min() - mY * (rng.random() + 1.0)
        y1 = Zb.max() + mY * (rng.random() + 1.0)
    nx = max(int(np.floor(1e3 * dens_hr * (x1 - x0))), block)
    ny = max(int(np.floor(1e3 * dens_hr * (y1 - y0))), block)
    PX, PY = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    ins = M.points_in_shape_radial(PX.ravel(), PY.ravel(), Yb, Zb,
                                   ycR, zcR).reshape(PX.shape).astype(float)
    XT, YT, V = M._block_average(PX, PY, ins, block)
    V = V + rng.standard_normal(V.shape) * sn
    B = V >= M.THRESH
    if B.sum() == 0:
        return (np.nan, np.nan)
    return (XT[B].mean() - ycR, YT[B].mean() - zcR)


def main():
    out = {'rep': REP, 'r_mm_per_line': 1.0, 'alpha': 0.0, 'oversample': OS}
    with Pool(cpu_count()) as pool:
        for ph in ('fixed', 'random'):
            out[ph] = {}
            for sn in NOISE:
                ss = np.random.SeedSequence(int(5500 + 100 * sn)).spawn(REP)
                a = np.array(pool.map(_trial, [(s.generate_state(1)[0], sn, ph)
                                               for s in ss], chunksize=16))
                a = a[~np.isnan(a).any(axis=1)]
                bx, by = a[:, 0].mean() * 1e6, a[:, 1].mean() * 1e6
                out[ph]['%.2f' % sn] = dict(bx=float(bx), by=float(by),
                                            mag=float(np.hypot(bx, by)))
                print('phase %-6s sigma_eff = %.2f  |b| = %.1f um'
                      % (ph, sn, out[ph]['%.2f' % sn]['mag']), flush=True)
    with open(os.path.join(RES, 'phase_test.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('saved', os.path.join(RES, 'phase_test.json'))


if __name__ == '__main__':
    main()
