# -*- coding: utf-8 -*-
"""On-line estimation of the inputs of the coverage-fraction bias model from the frame.

    x0        raw binary centroid of the analysed region
    D         from the number of pixels above threshold, D = 2 sqrt(m/pi) [px]
    sigma_n   robust scale (1.4826 x MAD) of the grey values in the outer ring of the
              analysed region
    c_k       coverage map of a model disc of diameter D centred on the current
              estimate, or of an ellipse from the second moments of the blob
    b         the coverage-fraction bias model on that map; x <- x0 - b; iterate

Measured: iterations to convergence, the residual against an oracle given the true
coverage map, and the sensitivity to errors in D and sigma_n.

Writes results/estimator.json.
"""
import argparse
import json
import os
import time

import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.special import ndtr

import iris_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))
OUT = os.path.join(RES, 'estimator.json')

VERS = np.array([0.0, 1.0, 2.0])
OS = 5
SUB = 8                      # subpixel samples per axis for the model coverage map
MAXIT = 12
TOL = 1e-3                   # convergence tolerance, in low-resolution pixels


# ------------------------------------------------------------------ scene
def scene(seed, dens, angle, oversample=OS):
    """Noise-free sensor image of one pupil: coordinates, coverage map, true centre."""
    rng = np.random.default_rng(int(seed))
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
    XT, YT, V = M._block_average(PX, PY, ins, block)
    return XT, YT, V, ycR, zcR, rng


def disc_coverage(XT, YT, cx, cy, D, pitch, sub=SUB):
    """Coverage fraction of every pixel by a disc of diameter D centred at (cx, cy).

    Distances are in metres; pitch is the sensor pitch in metres. Evaluated by
    supersampling the pixel footprint, which is what an instrument would tabulate.
    """
    off = (np.arange(sub) + 0.5) / sub - 0.5
    ox, oy = np.meshgrid(off * pitch, off * pitch)
    ox = ox.ravel(); oy = oy.ravel()
    dx = XT[..., None] + ox - cx
    dy = YT[..., None] + oy - cy
    return (dx * dx + dy * dy <= (D / 2.0) ** 2).mean(axis=-1)


def ellipse_coverage(XT, YT, cx, cy, cov, area, pitch, sub=SUB):
    """As disc_coverage but for the ellipse of the given second-moment matrix,
    rescaled so that its area equals `area`."""
    w, Vv = np.linalg.eigh(cov)
    w = np.clip(w, 1e-30, None)
    ax = np.sqrt(w)
    ax = ax * np.sqrt(area / (np.pi * ax[0] * ax[1]))
    off = (np.arange(sub) + 0.5) / sub - 0.5
    ox, oy = np.meshgrid(off * pitch, off * pitch)
    ox = ox.ravel(); oy = oy.ravel()
    dx = XT[..., None] + ox - cx
    dy = YT[..., None] + oy - cy
    u = Vv[0, 0] * dx + Vv[1, 0] * dy
    v = Vv[0, 1] * dx + Vv[1, 1] * dy
    return ((u / ax[0]) ** 2 + (v / ax[1]) ** 2 <= 1.0).mean(axis=-1)


def bias_from_map(XT, YT, c, sn, t=M.THRESH):
    p = ndtr((c - t) / sn)
    sp = p.sum()
    if sp <= 0:
        return None
    return (XT * p).sum() / sp, (YT * p).sum() / sp


# ------------------------------------------------------------------ one trial
def run_trial(task):
    seed, dens, sn, angle, mode, dscale, sscale, maxit, fixedD, fixedSN = task
    XT, YT, V0, tx, ty, rng = scene(seed, dens, angle)
    pitch = float(XT[0, 1] - XT[0, 0])
    V = V0 + rng.standard_normal(V0.shape) * sn
    B = V >= M.THRESH
    m = int(B.sum())
    if m < 3:
        return None
    x0 = float(XT[B].mean()); y0 = float(YT[B].mean())

    # --- sigma_n from the outer ring, which no plausible pupil reaches
    D_px = 2.0 * np.sqrt(m / np.pi)
    rr = np.hypot(XT - x0, YT - y0)
    ring = rr > 1.1 * D_px * pitch
    if ring.sum() < 20:
        ring = rr > rr.max() * 0.7
    vals = V[ring]
    sn_hat = 1.4826 * float(np.median(np.abs(vals - np.median(vals))))
    sn_hat = max(sn_hat, 1e-4) * sscale

    D_hat = D_px * pitch * dscale
    # A tracker sees thousands of frames per second, and the pupil diameter and the
    # noise level change on a far slower timescale than the position does. Both may
    # therefore be averaged over a block of frames, which is what `fixedD`/`fixedSN`
    # stand for here; only the raw centroid stays per-frame.
    if fixedD is not None:
        D_hat = fixedD
    if fixedSN is not None:
        sn_hat = fixedSN
    cov = None
    if mode == 'ellipse':
        wx = XT[B] - x0; wy = YT[B] - y0
        cov = np.array([[float((wx * wx).mean()), float((wx * wy).mean())],
                        [float((wx * wy).mean()), float((wy * wy).mean())]])

    # --- iterate
    cx, cy = x0, y0
    it = 0
    for it in range(1, maxit + 1):
        if mode == 'ellipse' and cov is not None:
            c = ellipse_coverage(XT, YT, cx, cy, cov, m * pitch * pitch, pitch)
        else:
            c = disc_coverage(XT, YT, cx, cy, D_hat, pitch)
        e = bias_from_map(XT, YT, c, sn_hat)
        if e is None:
            break
        bx, by = e[0] - cx, e[1] - cy
        nx_, ny_ = x0 - bx, y0 - by
        if np.hypot(nx_ - cx, ny_ - cy) < TOL * pitch:
            cx, cy = nx_, ny_
            break
        cx, cy = nx_, ny_

    # --- oracle: the same correction computed from the TRUE coverage map
    e = bias_from_map(XT, YT, V0, sn)
    if e is None:
        return None
    obx, oby = e[0] - tx, e[1] - ty
    return (x0 - tx, y0 - ty,                 # raw error
            cx - tx, cy - ty,                 # error after the online correction
            x0 - obx - tx, y0 - oby - ty,     # error after the oracle correction
            it, D_hat / pitch, sn_hat)


def summarise(rows, pitch_um):
    a = np.array([r for r in rows if r is not None], float)
    if len(a) < 10:
        return None
    def stat(ix, iy):
        ex, ey = a[:, ix] * 1e6, a[:, iy] * 1e6
        return dict(bias=float(np.hypot(ex.mean(), ey.mean())),
                    s=float(np.sqrt((ex.std(ddof=1) ** 2 + ey.std(ddof=1) ** 2) / 2)),
                    rmse=float(np.sqrt((ex ** 2 + ey ** 2).mean() / 2)))
    return dict(n=int(len(a)),
                raw=stat(0, 1), online=stat(2, 3), oracle=stat(4, 5),
                iterations_mean=float(a[:, 6].mean()),
                iterations_max=int(a[:, 6].max()),
                D_hat_px_mean=float(a[:, 7].mean()),
                sigma_hat_mean=float(a[:, 8].mean()))


def sweep(pool, rep, dens, sn, angle, mode='disc', dscale=1.0, sscale=1.0,
          seed=42, maxit=MAXIT, fixedD=None, fixedSN=None):
    ss = np.random.SeedSequence(seed).spawn(rep)
    tasks = [(s.generate_state(1)[0], dens, sn, angle, mode, dscale, sscale, maxit,
              fixedD, fixedSN) for s in ss]
    return summarise(pool.map(run_trial, tasks, chunksize=8), 1.0 / dens * 1e3)


def sweep_pair(pool, rep, dens, sn, angle, **kw):
    """Per-frame inputs, then the same run with the two slow inputs block-averaged."""
    a = sweep(pool, rep, dens, sn, angle, **kw)
    if not a:
        return a, None
    pitch = 1.0 / (dens * 1e3)
    b = sweep(pool, rep, dens, sn, angle,
              fixedD=a['D_hat_px_mean'] * pitch, fixedSN=a['sigma_hat_mean'], **kw)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rep', type=int, default=2400)
    ap.add_argument('-j', '--nproc', type=int, default=None)
    args = ap.parse_args()
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {}
    doc.setdefault('meta', {}).update(rep=args.rep, oversample=OS, subsample=SUB,
                                      maxit=MAXIT, tol_px=TOL)
    t0 = time.time()
    with Pool(processes=args.nproc or cpu_count()) as pool:
        def report(k, a, b):
            doc[k] = a
            if b:
                doc[k + '_smoothed'] = b
            print('%-20s raw %6.1f/%6.1f | per-frame %6.1f/%6.1f | smoothed '
                  '%6.1f/%6.1f | oracle %6.1f/%6.1f   (bias/rmse um), it %.1f'
                  % (k, a['raw']['bias'], a['raw']['rmse'],
                     a['online']['bias'], a['online']['rmse'],
                     b['online']['bias'] if b else float('nan'),
                     b['online']['rmse'] if b else float('nan'),
                     a['oracle']['bias'], a['oracle']['rmse'],
                     a['iterations_mean']), flush=True)
            json.dump(doc, open(OUT, 'w'), indent=1)

        # (1) noise ladder at the reference pitch
        for sn in (0.10, 0.20, 0.30, 0.40, 0.50):
            report('ladder_sn%.2f' % sn,
                   *sweep_pair(pool, args.rep, 1.0, sn, 0.0))

        # (2) across the pitch range, at the noise level where the correction matters
        for r_mm, dens in ((0.500, 2.0), (0.667, 1.5), (1.000, 1.0),
                           (1.429, 0.7), (2.000, 0.5)):
            report('pitch_r%.3f' % r_mm,
                   *sweep_pair(pool, args.rep, dens, 0.40, 0.0))

        # (3) eccentric gaze, where a circular model pupil is wrong
        for a in (30.0, 45.0, 60.0):
            for mode in ('disc', 'ellipse'):
                report('gaze_a%02d_%s' % (a, mode),
                       *sweep_pair(pool, args.rep, 1.0, 0.40, a, mode=mode))

        # (4) sensitivity to the two estimated inputs
        for lab, ds, ss_ in (('D-20', 0.8, 1.0), ('D-10', 0.9, 1.0),
                             ('D+10', 1.1, 1.0), ('D+20', 1.2, 1.0),
                             ('sn-20', 1.0, 0.8), ('sn-10', 1.0, 0.9),
                             ('sn+10', 1.0, 1.1), ('sn+20', 1.0, 1.2)):
            k = 'sens_%s' % lab
            doc[k] = sweep(pool, args.rep, 1.0, 0.40, 0.0, 'disc', ds, ss_)
            print('%-18s online bias %6.1f um' % (k, doc[k]['online']['bias']),
                  flush=True)
            json.dump(doc, open(OUT, 'w'), indent=1)

        # (5) convergence trace: bias after each iteration count
        trace = {}
        for nit in (1, 2, 3, 4, 6, 8):
            r = sweep(pool, args.rep, 1.0, 0.40, 0.0, maxit=nit)
            trace['it%d' % nit] = r['online']['bias'] if r else None
            print('  iterations %d -> residual bias %.1f um'
                  % (nit, trace['it%d' % nit]), flush=True)
        doc['trace'] = trace
        json.dump(doc, open(OUT, 'w'), indent=1)
    print('\n%.1f min -> %s' % ((time.time() - t0) / 60, OUT))


if __name__ == '__main__':
    main()
