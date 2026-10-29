# -*- coding: utf-8 -*-
"""Bias correction and competing estimators.

Four estimators on the identical image in every trial, so every comparison is paired:

    centroid    centre of mass of the thresholded foreground
    weighted    intensity-weighted centre of gravity over the whole window
    weighted-t  intensity-weighted centre of gravity over the pixels above threshold
    ellipse     direct least-squares ellipse fit (Halir and Flusser) to the boundary
                of the largest foreground blob

For the binary centroid, the bias predicted by the coverage-fraction bias model on the
true coverage map and the residual after correction are recorded. Two further studies:
the noise level drawn per frame across the knee (variable-noise scenario), and the
processing time of the estimators in a 64 x 64 window.

Writes results/alternatives.json.
"""
import argparse
import json
import os
import time

import numpy as np
from multiprocessing import Pool, cpu_count
from scipy import ndimage
from scipy.special import ndtr

import iris_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))
OUT = os.path.join(RES, 'alternatives.json')
VERS = np.array([0.0, 1.0, 2.0])
OS = 5


# ------------------------------------------------------------------ ellipse
def fit_ellipse_center(x, y):
    """Direct least-squares ellipse fit (Halir and Flusser); (xc, yc) or None."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 6:
        return None
    mx, my = x.mean(), y.mean()
    xc, yc = x - mx, y - my
    D1 = np.column_stack([xc * xc, xc * yc, yc * yc])
    D2 = np.column_stack([xc, yc, np.ones_like(xc)])
    S1 = D1.T @ D1; S2 = D1.T @ D2; S3 = D2.T @ D2
    if np.linalg.cond(S3) > 1e12:
        return None
    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        return None
    M0 = S1 + S2 @ T
    Mr = np.vstack([M0[2, :] / 2.0, -M0[1, :], M0[0, :] / 2.0])
    try:
        w, v = np.linalg.eig(Mr)
    except np.linalg.LinAlgError:
        return None
    cond = 4 * v[0, :].real * v[2, :].real - v[1, :].real ** 2
    sel = np.where(cond > 0)[0]
    if sel.size == 0:
        return None
    a1 = v[:, sel[0]].real
    A, B, C = a1
    D, E, F = (T @ a1)
    den = B * B - 4 * A * C
    if abs(den) < 1e-30:
        return None
    return (2 * C * D - B * E) / den + mx, (2 * A * E - B * D) / den + my


# ------------------------------------------------------------------ one trial
def trial(task):
    """Returns errors of the four estimates about the true centre, in metres.

    (cx, cy, wx, wy, ex, ey, kx, ky, ok_ellipse, px, py, gx, gy) where c = binary centroid,
    w = intensity-weighted centroid, e = ellipse fit, k = the centroid corrected by
    the coverage-fraction bias model evaluated on the true coverage map, and (px, py)
    is the predicted bias vector.

    The seed streams are those of run_converged.py, so the centroid columns reproduce
    converged_os5.json trial for trial.
    """
    seed, dens, angle, sn = task
    rng = np.random.default_rng(int(seed))
    amp, ph = M.population_distortion(rng)
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
    XT, YT, V0 = M._block_average(PX, PY, ins, block)
    V = V0 + rng.standard_normal(V0.shape) * sn
    B = V >= M.THRESH
    out = [np.nan] * 13

    if B.sum() > 0:
        out[0] = XT[B].mean() - ycR
        out[1] = YT[B].mean() - zcR
        # oracle correction from the true coverage map
        p = ndtr((V0 - M.THRESH) / sn)
        sp = p.sum()
        if sp > 0:
            bx = (XT * p).sum() / sp - ycR
            by = (YT * p).sum() / sp - zcR
            out[6] = out[0] - bx
            out[7] = out[1] - by
            out[9] = bx
            out[10] = by

    # intensity-weighted centre of gravity: the dark-pupil signal is the coverage
    # itself, clipped at zero because a negative weight has no physical meaning
    w = np.clip(V, 0.0, None)
    sw = w.sum()
    if sw > 0:
        out[2] = (XT * w).sum() / sw - ycR
        out[3] = (YT * w).sum() / sw - zcR
    # the same estimator gated by the threshold, which is how it is used in practice
    wt = np.where(B, w, 0.0)
    swt = wt.sum()
    if swt > 0:
        out[11] = (XT * wt).sum() / swt - ycR
        out[12] = (YT * wt).sum() / swt - zcR

    if B.sum() > 0:
        lab, nl = ndimage.label(B)
        if nl >= 1:
            sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, nl + 1))
            big = lab == (int(np.argmax(sizes)) + 1)
            edge = big & ~ndimage.binary_erosion(big)
            f = fit_ellipse_center(XT[edge], YT[edge])
            if f is not None and np.isfinite(f).all():
                out[4] = f[0] - ycR
                out[5] = f[1] - zcR
                out[8] = 1.0
            else:
                out[8] = 0.0
    return tuple(out)


GATE_UM = 5000.0        # one pupil diameter: beyond it the estimate is a failure


def _time_once(f):
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


def stats(a, ix, iy):
    """Dispersion, bias and RMSE, with gross failures counted rather than averaged.

    A trial whose least-squares conic is degenerate, or whose estimate lies more than
    one pupil diameter from the true centre, is counted in `fail_rate` and excluded
    from the moments, in the same way as a trial whose thresholded image keeps no
    foreground pixel is counted in the availability.
    """
    e = a[:, [ix, iy]]
    n_all = len(e)
    e = e[~np.isnan(e).any(axis=1)]
    if len(e) < 10:
        return None
    r = np.hypot(e[:, 0], e[:, 1]) * 1e6
    keep = r <= GATE_UM
    n_fail = int(n_all - keep.sum())
    e = e[keep]
    if len(e) < 10:
        return None
    ex, ey = e[:, 0] * 1e6, e[:, 1] * 1e6
    return dict(n=int(len(e)), fail_rate=round(n_fail / max(n_all, 1), 4),
                s=round(float(np.sqrt((ex.std(ddof=1) ** 2
                                       + ey.std(ddof=1) ** 2) / 2)), 1),
                bias=round(float(np.hypot(ex.mean(), ey.mean())), 1),
                rmse=round(float(np.sqrt((ex ** 2 + ey ** 2).mean() / 2)), 1))


def point(pool, rep, dens, angle, sn, seed):
    ss = np.random.SeedSequence(seed).spawn(rep)
    a = np.array(pool.map(trial, [(s.generate_state(1)[0], dens, angle, sn)
                                  for s in ss], chunksize=16))
    pv = a[:, [9, 10]]
    pv = pv[~np.isnan(pv).any(axis=1)]
    pred = pred_phi = None
    if len(pv):
        px, py = pv[:, 0].mean(), pv[:, 1].mean()
        pred = round(float(np.hypot(px, py) * 1e6), 1)
        pred_phi = round(float(np.degrees(np.arctan2(py, px))), 1)
    mv = a[:, [0, 1]]
    mv = mv[~np.isnan(mv).any(axis=1)]
    meas_phi = (round(float(np.degrees(np.arctan2(mv[:, 1].mean(),
                                                  mv[:, 0].mean()))), 1)
                if len(mv) else None)
    return dict(centroid=stats(a, 0, 1), weighted=stats(a, 2, 3),
                measured_phi=meas_phi, predicted_phi=pred_phi,
                weighted_t=stats(a, 11, 12),
                ellipse=stats(a, 4, 5), corrected=stats(a, 6, 7),
                predicted_bias=pred,
                ellipse_fit_rate=round(float(np.nanmean(a[:, 8])), 3))


# ------------------------------------------------------------------ study V
def variable_noise(pool, rep, seed=31337):
    """The noise level drawn per frame, straddling the knee.

    A lognormal centred on sigma_n = 0.12, with a spread chosen so that about a tenth
    of the frames exceed 0.20 (varying illumination).
    """
    rng = np.random.default_rng(seed)
    mu, sg = np.log(0.12), 0.42
    sn = np.exp(mu + sg * rng.standard_normal(rep))
    sn = np.clip(sn, 0.02, 0.60)
    ss = np.random.SeedSequence(seed).spawn(rep)
    a = np.array(pool.map(trial, [(s.generate_state(1)[0], 1.0, 0.0, float(v))
                                  for s, v in zip(ss, sn)], chunksize=16))
    out = dict(rep=rep, median_sigma_n=float(np.median(sn)),
               frac_above_020=float((sn > 0.20).mean()),
               frac_above_015=float((sn > 0.15).mean()))
    for name, ix in (('centroid', 0), ('weighted', 2), ('weighted_t', 11),
                     ('ellipse', 4), ('corrected', 6)):
        out[name] = stats(a, ix, ix + 1)
    return out


# ------------------------------------------------------------------ study C
def cost_benchmark(n=400, w=64, seed=7):
    """Per-frame cost of the three estimators in a 64 x 64 window.

    NumPy timings on one desktop core, in the window size of a 2 kHz tracker; only
    the ratio between the three estimators is meaningful, not the absolute value.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:w, 0:w]
    imgs = []
    for _ in range(n):
        cx, cy = w / 2 + rng.uniform(-3, 3), w / 2 + rng.uniform(-3, 3)
        d = np.hypot(xx - cx, yy - cy)
        img = (d <= 12.0).astype(np.float32)
        img += rng.standard_normal(img.shape).astype(np.float32) * 0.10
        imgs.append(img)

    def t_centroid():
        for im in imgs:
            b = im >= 0.5
            k = b.sum()
            if k:
                (xx * b).sum() / k, (yy * b).sum() / k

    def t_weighted():
        for im in imgs:
            b = im >= 0.5
            wgt = np.where(b, np.clip(im, 0, None), 0.0)
            sw = wgt.sum()
            if sw:
                (xx * wgt).sum() / sw, (yy * wgt).sum() / sw

    def t_ellipse():
        for im in imgs:
            b = im >= 0.5
            lab, nl = ndimage.label(b)
            if nl < 1:
                continue
            sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, nl + 1))
            big = lab == (int(np.argmax(sizes)) + 1)
            edge = big & ~ndimage.binary_erosion(big)
            fit_ellipse_center(xx[edge], yy[edge])

    out = {}
    for name, f in (('centroid', t_centroid), ('weighted', t_weighted),
                    ('ellipse', t_ellipse)):
        f()                                   # warm up
        # the minimum over repetitions, which is the standard way to time code:
        # anything above it is contention with the rest of the machine
        best = min(_time_once(f) for _ in range(5))
        out[name + '_us_per_frame'] = best / n * 1e6
    out['window_px'] = w
    out['frames'] = n
    for name in ('weighted', 'ellipse'):
        out[name + '_relative'] = (out[name + '_us_per_frame']
                                   / out['centroid_us_per_frame'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rep', type=int, default=1800)
    ap.add_argument('-j', '--nproc', type=int, default=None)
    ap.add_argument('--only', nargs='*', default=['noise', 'pitch', 'gaze',
                                                  'variable', 'cost'])
    args = ap.parse_args()
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {}
    doc.setdefault('meta', {}).update(rep=args.rep, oversample=OS)
    t0 = time.time()
    with Pool(processes=args.nproc or cpu_count()) as pool:
        if 'noise' in args.only:
            print('--- noise sweep (r = 1.00, alpha = 0) ---')
            d = {}
            for i, sn in enumerate([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]):
                d['sn%.2f' % sn] = point(pool, args.rep, 1.0, 0.0, sn, 5000 + i)
                p = d['sn%.2f' % sn]
                print('  sn=%.2f  centroid s=%s b=%s | weighted-t s=%s b=%s | '
                      'ellipse s=%s b=%s | corrected b=%s'
                      % (sn, p['centroid']['s'], p['centroid']['bias'],
                         p['weighted_t']['s'], p['weighted_t']['bias'],
                         p['ellipse']['s'] if p['ellipse'] else None,
                         p['ellipse']['bias'] if p['ellipse'] else None,
                         p['corrected']['bias'] if p['corrected'] else None),
                      flush=True)
            doc['noise'] = d
            json.dump(doc, open(OUT, 'w'), indent=1)

        if 'pitch' in args.only:
            print('--- pitch sweep (sigma_n = 0.10, alpha = 0) ---')
            d = {}
            for r_mm, dens, si in [(0.500, 2.0, 6), (0.667, 1.5, 5),
                                   (1.000, 1.0, 4), (1.429, 0.7, 3),
                                   (2.000, 0.5, 2)]:
                d['r%.3f' % r_mm] = point(pool, args.rep, dens, 0.0, 0.10,
                                          3000 + 23 * si + 4)
                p = d['r%.3f' % r_mm]
                print('  r=%.3f  centroid s=%s | weighted-t s=%s | ellipse s=%s '
                      '(fits %.2f)'
                      % (r_mm, p['centroid']['s'], p['weighted_t']['s'],
                         p['ellipse']['s'] if p['ellipse'] else None,
                         p['ellipse_fit_rate']), flush=True)
            doc['pitch'] = d
            json.dump(doc, open(OUT, 'w'), indent=1)

        if 'gaze' in args.only:
            print('--- gaze sweep (sigma_n = 0.10, r = 1.00) ---')
            d = {}
            for a, j in [(0.0, 4), (30.0, 6), (60.0, 8)]:
                d['a%02d' % a] = point(pool, args.rep, 1.0, a, 0.10,
                                       1000 + 17 * 1 + j)
                p = d['a%02d' % a]
                print('  alpha=%2.0f  centroid s=%s | weighted-t s=%s | ellipse s=%s'
                      % (a, p['centroid']['s'], p['weighted_t']['s'],
                         p['ellipse']['s'] if p['ellipse'] else None), flush=True)
            doc['gaze'] = d
            json.dump(doc, open(OUT, 'w'), indent=1)

        if 'variable' in args.only:
            print('--- variable-noise scenario ---')
            doc['variable'] = variable_noise(pool, max(args.rep * 3, 6000))
            v = doc['variable']
            print('  median sigma_n %.3f, %.0f %% of frames above 0.20'
                  % (v['median_sigma_n'], 100 * v['frac_above_020']))
            for k in ('centroid', 'corrected', 'weighted_t', 'weighted', 'ellipse'):
                if v.get(k):
                    print('  %-10s RMSE %7.1f um  (s %6.1f, bias %6.1f)'
                          % (k, v[k]['rmse'], v[k]['s'], v[k]['bias']))
            json.dump(doc, open(OUT, 'w'), indent=1)

    if 'cost' in args.only:
        print('--- windowed processing cost ---')
        doc['cost'] = cost_benchmark()
        for k, v in doc['cost'].items():
            print('  %-26s %s' % (k, round(v, 3) if isinstance(v, float) else v))
        json.dump(doc, open(OUT, 'w'), indent=1)
    print('\n%.1f min -> %s' % ((time.time() - t0) / 60, OUT))


if __name__ == '__main__':
    main()
