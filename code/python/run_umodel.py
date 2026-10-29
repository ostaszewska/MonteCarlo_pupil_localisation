# -*- coding: utf-8 -*-
"""Model-incompleteness components of the uncertainty budget.

One run of the pipeline per mechanism, each paired with the baseline (identical seed
streams, so pupil shape, window margins and noise realisations are the same):

  shot       signal-dependent (photon) noise at the same image-mean noise power
  contrast   pupil-iris contrast C < 1, effective relative noise sigma_n / C
  optics     Gaussian PSF of FWHM 0.5 and 1.0 sensor pixels applied before sampling
  occlusion  pupil truncated by a chord removing 10, 20, 30 % of its area
  diameter   pupil diameter drawn with s.d. 0.5 mm, and fixed at 3.09 mm (photopic)

Writes results/umodel.json; budget.py combines it.
"""
import argparse
import json
import os
import time

import numpy as np
from multiprocessing import Pool, cpu_count
from scipy import ndimage as ndi

import iris_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))
OUT = os.path.join(RES, 'umodel.json')

VERS = np.array([0.0, 1.0, 2.0])
SN_LADDER = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
SEED_BASE = dict((sn, 5000 + i) for i, sn in enumerate(SN_LADDER))   # baseline seeds
OS = 5                                                            # reporting grid


# --------------------------------------------------------------------- pipeline
def trial(task):
    """One Monte Carlo trial with an optional mechanism switched on.

    Returns (true_y, true_z, est_y, est_z) in metres, or NaNs when the thresholded
    image keeps no foreground pixel.
    """
    seed, dens, sn, angle, oversample, mech, par, iris_d = task
    if iris_d is not None:
        M.IRIS_D = iris_d                      # per worker: Windows spawns processes
    rng = np.random.default_rng(int(seed))
    if mech == 'diameter' and par is not None:      # par = (mean, sd) in metres
        M.IRIS_D = float(par[0] + par[1] * np.clip(rng.standard_normal(), -2, 2))
    amp, ph = M.population_distortion(rng)

    block = M.REDUCE * oversample
    dens_hr = dens * M.RES_MULT * oversample

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

    if mech == 'occlusion' and par:
        # upper-eyelid chord: drop the top fraction `par` of the pupil area
        rows = ins.sum(axis=1)
        tot = rows.sum()
        if tot > 0:
            above = np.cumsum(rows[::-1])[::-1]          # pupil area from row i up
            cut = int(np.argmax(above <= par * tot))
            ins[cut:, :] = 0.0

    if mech == 'optics' and par:
        # Gaussian PSF, FWHM given in SENSOR pixels, applied before sampling
        sig = par * block / 2.3548200450309493
        ins = ndi.gaussian_filter(ins, sig, mode='nearest')

    PXr, PYr, V = M._block_average(PX, PY, ins, block)

    if mech == 'shot':
        # photon noise: in dark-pupil imaging the collected signal is the iris, so
        # the photon count per pixel scales with (1 - coverage). The scale is fixed
        # by requiring the image-mean noise variance to equal that of the
        # homoscedastic baseline, so the two runs are compared at equal noise power.
        w = 1.0 - V
        mw = w.mean()
        sd = sn * np.sqrt(w / mw) if mw > 0 else np.full_like(V, sn)
        V = V + rng.standard_normal(V.shape) * sd
    else:
        V = V + rng.standard_normal(V.shape) * sn
    V = (V >= M.THRESH).astype(float)

    k = V.sum()
    if k <= 0:
        return (np.nan,) * 4
    return (ycR, zcR, float((PXr * V).sum() / k), float((PYr * V).sum() / k))


def point(pool, rep, dens, sn, angle, seed, mech=None, par=None, iris_d=None):
    ss = np.random.SeedSequence(seed).spawn(rep)
    tasks = [(s.generate_state(1)[0], dens, sn, angle, OS, mech, par, iris_d)
             for s in ss]
    a = np.array(pool.map(trial, tasks, chunksize=16))
    ok = ~np.isnan(a).any(axis=1)
    g = a[ok]
    if len(g) < 10:
        return dict(n=int(ok.sum()), discard=float(1 - ok.mean()), s=None)
    ex, ey = g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]
    s = float(np.sqrt((ex.std(ddof=1) ** 2 + ey.std(ddof=1) ** 2) / 2) * 1e6)
    bias = float(np.hypot(ex.mean(), ey.mean()) * 1e6)
    return dict(n=int(len(g)), discard=round(float(1 - ok.mean()), 4),
                s=round(s, 2), bias=round(bias, 2),
                rmse=round(float(np.hypot(bias, s)), 2))


# --------------------------------------------------------------------- studies
def study(pool, name, rep):
    out = {}
    if name == 'baseline':
        for sn in SN_LADDER:
            out['sn%.2f' % sn] = point(pool, rep, 1.0, sn, 0.0, SEED_BASE[sn])
            print('  baseline sn=%.2f  %s' % (sn, out['sn%.2f' % sn]), flush=True)

    elif name == 'shot':
        for sn in SN_LADDER:
            out['sn%.2f' % sn] = point(pool, rep, 1.0, sn, 0.0, SEED_BASE[sn], 'shot')
            print('  shot     sn=%.2f  %s' % (sn, out['sn%.2f' % sn]), flush=True)

    elif name == 'contrast':
        # a relative noise sigma_n referred to the full pixel range behaves as
        # sigma_n / C at a threshold that acts on the pupil-iris contrast C
        for C in (0.5, 0.3):
            for sn in (0.05, 0.10, 0.15):
                eff = sn / C
                k = 'C%.1f_sn%.2f' % (C, sn)
                out[k] = point(pool, rep, 1.0, eff, 0.0, SEED_BASE[sn])
                out[k]['sigma_eff'] = round(eff, 4)
                print('  contrast %s (eff %.3f)  %s' % (k, eff, out[k]), flush=True)

    elif name == 'optics':
        for fwhm in (0.5, 1.0):
            for sn in (0.05, 0.10, 0.20, 0.50):
                k = 'fwhm%.1f_sn%.2f' % (fwhm, sn)
                out[k] = point(pool, rep, 1.0, sn, 0.0, SEED_BASE[sn], 'optics', fwhm)
                print('  optics   %s  %s' % (k, out[k]), flush=True)

    elif name == 'occlusion':
        for q in (0.10, 0.20, 0.30):
            for sn in (0.05, 0.10, 0.20):
                k = 'q%.2f_sn%.2f' % (q, sn)
                out[k] = point(pool, rep, 1.0, sn, 0.0, SEED_BASE[sn], 'occlusion', q)
                print('  occl     %s  %s' % (k, out[k]), flush=True)

    elif name == 'diameter':
        # (a) sensitivity: fixed diameters at fixed sampling pitch
        for D in (3.09, 4.0, 4.5, 5.0, 5.5, 6.0):
            k = 'fixed_D%.2f' % D
            out[k] = point(pool, rep, 1.0, 0.10, 0.0, SEED_BASE[0.10], iris_d=D * 1e-3)
            print('  diam     %s  %s' % (k, out[k]), flush=True)
        # (b) the component itself: diameter redrawn per trial
        for sd_mm in (0.25, 0.50):
            k = 'drawn_sd%.2f' % sd_mm
            out[k] = point(pool, rep, 1.0, 0.10, 0.0, SEED_BASE[0.10],
                           'diameter', (5.0e-3, sd_mm * 1e-3))
            print('  diam     %s  %s' % (k, out[k]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*',
                    default=['baseline', 'shot', 'contrast', 'optics',
                             'occlusion', 'diameter'])
    ap.add_argument('--rep', type=int, default=1800)
    ap.add_argument('-j', '--nproc', type=int, default=None)
    args = ap.parse_args()

    doc = json.load(open(OUT)) if os.path.exists(OUT) else {}
    doc.setdefault('meta', {}).update(rep=args.rep, oversample=OS, r_mm=1.0, alpha=0.0)
    t0 = time.time()
    with Pool(processes=args.nproc or cpu_count()) as pool:
        for name in args.only:
            print('--- %s ---' % name, flush=True)
            doc[name] = study(pool, name, args.rep)
            json.dump(doc, open(OUT, 'w'), indent=1)
    print('\n%.1f min -> %s' % ((time.time() - t0) / 60, OUT))


if __name__ == '__main__':
    main()
