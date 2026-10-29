# -*- coding: utf-8 -*-
"""Analytic validation of the Monte Carlo against Van Rheeden & Jones (1988).

Van Rheeden and Jones derived, in closed form, the bias and variance of a BINARY
centroid computed over a FIXED analysis region when additive noise makes the
segmentation misclassify pixels. Their model is the one iris_model.py simulates, so
it is an analytic reference the Monte Carlo can be checked against.

    D. R. Van Rheeden, R. A. Jones, "Noise effects on centroid tracker aim point
    estimation", IEEE Trans. Aerosp. Electron. Syst. 24(2) (1988) 177-185,
    doi:10.1109/7.1051

Their model, in their notation: the region holds N_t true target pixels (set T,
centroid x_t) and N_b true background pixels (set B, centroid x_b); a target pixel
is called background with probability P_M = Pr(d_b|T), a background pixel is called
target with probability P_F = Pr(d_t|B). Replacing the denominator by its
expectation, their (27), gives their (29)/(30) for the aim point error x_t - E{x}:

    E{bias} = N_b P_F (x_t - x_b) / [ N_t (1 - P_M) + N_b P_F ]                (VR30)

and, with the origin at the target centroid, their (32) for the variance:

    Var{x} = [ P_M(1-P_M) SUM_T x_k^2 + P_F(1-P_F) SUM_B x_k^2 ] / D^2         (VR32)
    D = N_t (1 - P_M) + N_b P_F

(VR30) and (VR32) are reconstructed here from the model they state, because the
scanned equations are not machine-readable. They have the properties stated by Van
Rheeden and Jones: the bias vanishes when there are no background pixels in the
region; it is proportional to the probability of
misclassifying a background pixel; it is proportional to the displacement between
the target centroid and the centroid of the background pixels in the region; the
variance splits into a target term and a background term, the latter weighted by
the larger lever arms.

Two refinements are needed for the present model; each is reported separately.

  1. Van Rheeden's pixels are pure target or pure background. A sampled pupil also
     has boundary pixels of fractional coverage c_k. The natural generalisation
     assigns each pixel its own probability of being classified as foreground,

         p_k = P(c_k + noise >= t) = Phi((c_k - t) / sigma_n),

     which reduces to P_F for c_k = 0 and to 1 - P_M for c_k = 1. Then

         E{x} = SUM_k x_k p_k / SUM_k p_k                                    (GEN-bias)
         Var{x} = SUM_k (x_k - E{x})^2 p_k (1 - p_k) / (SUM_k p_k)^2          (GEN-var)

  2. Both forms neglect the randomness of the denominator, exactly as (27) does.

Usage (the four reported runs):
    python vrcheck.py --oversample 5 --json ../../results/vrcheck.json --key ref
    python vrcheck.py --oversample 5 --dens 2.0 --json ../../results/vrcheck.json --key r0500
    python vrcheck.py --oversample 5 --dens 0.5 --json ../../results/vrcheck.json --key r2000
    python vrcheck.py --oversample 5 --angle 60 --json ../../results/vrcheck.json --key a60
"""
import argparse
import json
import os

import numpy as np
from scipy.special import ndtr           # standard normal CDF
from scipy.io import savemat             # noqa: F401  (kept for parity of imports)

import iris_model as M

VERS = np.array([0.0, 1.0, 2.0])
SN = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def coverage_map(dens_ccd, angle, seed, oversample=1):
    """Sensor-pixel coverage map for one frozen pupil and one frozen window.

    Reproduces iris_model.single_trial up to, but not including, the noise, so the
    geometry it returns is exactly the geometry the Monte Carlo sees.
    Returns (x, y, c, x_true, y_true) with coordinates in metres and c in [0, 1].
    """
    rng = np.random.default_rng(seed)
    amp, phase = M.population_distortion(rng)
    bndYZ, ycR, zcR = M.build_pupil_boundary(VERS, angle, amp, phase)
    Yb, Zb = bndYZ[:, 0], bndYZ[:, 1]

    block = M.REDUCE * oversample
    dens_hr = dens_ccd * M.RES_MULT * oversample

    margX = M.MARGP * (Yb.max() - Yb.min())
    startX = Yb.min() - margX * (rng.random() + 1.0)
    stopX = Yb.max() + margX * (rng.random() + 1.0)
    margY = M.MARGP * (Zb.max() - Zb.min())
    startY = Zb.min() - margY * (rng.random() + 1.0)
    stopY = Zb.max() + margY * (rng.random() + 1.0)

    nx = max(int(np.floor(1e3 * dens_hr * (stopX - startX))), block)
    ny = max(int(np.floor(1e3 * dens_hr * (stopY - startY))), block)
    PX, PY = np.meshgrid(np.linspace(startX, stopX, nx),
                         np.linspace(startY, stopY, ny))
    inside = M.points_in_shape_radial(PX.ravel(), PY.ravel(),
                                      Yb, Zb, ycR, zcR).reshape(PX.shape).astype(float)
    PXr, PYr, C = M._block_average(PX, PY, inside, block)
    return PXr.ravel(), PYr.ravel(), C.ravel(), ycR, zcR


def analytic(x, y, c, sn, t=M.THRESH, two_class=False):
    """Predicted mean and SD of the binary centroid. two_class = pure (VR30)/(VR32)."""
    cc = np.round(c) if two_class else c
    p = ndtr((cc - t) / sn)
    D = p.sum()
    if D <= 0:
        return np.nan, np.nan, np.nan, np.nan
    mx, my = (x * p).sum() / D, (y * p).sum() / D
    v = p * (1.0 - p)
    sx = np.sqrt(((x - mx) ** 2 * v).sum()) / D
    sy = np.sqrt(((y - my) ** 2 * v).sum()) / D
    return mx, my, sx, sy


def monte_carlo(x, y, c, sn, n, rng, t=M.THRESH):
    """Same estimator, by simulation: add noise, threshold, take the centroid."""
    mx = np.empty(n); my = np.empty(n); keep = np.zeros(n, bool)
    for k in range(n):
        m = (c + rng.standard_normal(c.size) * sn) >= t
        s = m.sum()
        if s:
            mx[k] = x[m].mean(); my[k] = y[m].mean(); keep[k] = True
    mx, my = mx[keep], my[keep]
    return mx.mean(), my.mean(), mx.std(ddof=1), my.std(ddof=1), keep.mean()


def aggregate(args):
    """Bias vector averaged over frozen geometries, as the Monte Carlo bias of alternatives.py.

    That bias is the magnitude of the mean centroid offset over trials that
    draw a new shape and a new window phase each time. The per-geometry bias has a
    large component that is random in sign across geometries and averages out, so a
    single frozen geometry is not comparable with alternatives.py; this mode is.
    """
    print('bias averaged over %d frozen geometries, N_MC = %d each'
          % (args.shapes, args.nmc))
    print('%-6s | %8s %8s %8s | %8s' % ('sig_n', 'MC', 'gen', 'VR30', 'dev'))
    print('-' * 46)
    geoms = [coverage_map(args.dens, args.angle, args.seed + 97 * k,
                          oversample=args.oversample)
             for k in range(args.shapes)]
    rng = np.random.default_rng(args.seed + 1)
    for sn in SN:
        bm = np.zeros(2); bg = np.zeros(2); bv = np.zeros(2)
        for (x, y, c, xt, yt) in geoms:
            amx, amy, _, _ = analytic(x, y, c, sn)
            vmx, vmy, _, _ = analytic(x, y, c, sn, two_class=True)
            mmx, mmy, _, _, _ = monte_carlo(x, y, c, sn, args.nmc, rng)
            bm += [mmx - xt, mmy - yt]
            bg += [amx - xt, amy - yt]
            bv += [vmx - xt, vmy - yt]
        n = len(geoms)
        Bm, Bg, Bv = (np.hypot(*(v / n)) * 1e6 for v in (bm, bg, bv))
        print('%-6.2f | %8.1f %8.1f %8.1f | %7.1f%%'
              % (sn, Bm, Bg, Bv, 100 * abs(Bg - Bm) / max(Bm, 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dens', type=float, default=1.0)
    ap.add_argument('--angle', type=float, default=0.0)
    ap.add_argument('--nmc', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--oversample', type=int, default=1)
    ap.add_argument('--shapes', type=int, default=0,
                    help='>0: average the bias vector over this many geometries')
    ap.add_argument('--json', default=None, help='append the result to this JSON file')
    ap.add_argument('--key', default=None, help='key under which to store it')
    args = ap.parse_args()

    if args.shapes:
        return aggregate(args)

    x, y, c, xt, yt = coverage_map(args.dens, args.angle, args.seed,
                                   oversample=args.oversample)
    rng = np.random.default_rng(args.seed + 1)

    n_t = int((c > 0.999).sum())
    n_b = int((c < 0.001).sum())
    n_edge = c.size - n_t - n_b
    xb = x[c < 0.001].mean(); yb = y[c < 0.001].mean()
    d = np.hypot(xb - xt, yb - yt) * 1e6

    print('reference point: r = %.3f mm/line, alpha = %.0f deg, N_MC = %d'
          % (M.sensor_pitch_mm(args.dens), args.angle, args.nmc))
    print('region: %d sensor pixels = %d target + %d background + %d boundary'
          % (c.size, n_t, n_b, n_edge))
    print('|d| (pupil centre to background centroid) = %.0f um' % d)
    print()
    print('%-6s | %-21s | %-21s | %-13s' % ('', 'BIAS [um]', 'SD [um]', 'detect'))
    print('%-6s | %6s %6s %7s | %6s %6s %7s | %6s'
          % ('sig_n', 'MC', 'gen', 'VR30', 'MC', 'gen', 'VR32', 'rate'))
    print('-' * 74)

    rows = []
    for sn in SN:
        amx, amy, asx, asy = analytic(x, y, c, sn)
        vmx, vmy, vsx, vsy = analytic(x, y, c, sn, two_class=True)
        mmx, mmy, msx, msy, rate = monte_carlo(x, y, c, sn, args.nmc, rng)

        b_mc = np.hypot(mmx - xt, mmy - yt) * 1e6
        b_gen = np.hypot(amx - xt, amy - yt) * 1e6
        b_vr = np.hypot(vmx - xt, vmy - yt) * 1e6
        s_mc = np.sqrt((msx ** 2 + msy ** 2) / 2) * 1e6
        s_gen = np.sqrt((asx ** 2 + asy ** 2) / 2) * 1e6
        s_vr = np.sqrt((vsx ** 2 + vsy ** 2) / 2) * 1e6

        print('%-6.2f | %6.1f %6.1f %7.1f | %6.1f %6.1f %7.1f | %5.1f%%'
              % (sn, b_mc, b_gen, b_vr, s_mc, s_gen, s_vr, 100 * rate))
        rows.append((sn, b_mc, b_gen, b_vr, s_mc, s_gen, s_vr))

    a = np.array(rows)
    ok = a[:, 4] > 0
    print()
    print('agreement, generalised model vs Monte Carlo:')
    print('  bias: mean |rel. dev.| = %5.1f %%' %
          (100 * np.mean(np.abs(a[ok, 2] - a[ok, 1]) / np.maximum(a[ok, 1], 1e-9))))
    print('  SD  : mean |rel. dev.| = %5.1f %%' %
          (100 * np.mean(np.abs(a[ok, 5] - a[ok, 4]) / a[ok, 4])))
    print('agreement, Van Rheeden two-class form vs Monte Carlo:')
    print('  bias: mean |rel. dev.| = %5.1f %%' %
          (100 * np.mean(np.abs(a[ok, 3] - a[ok, 1]) / np.maximum(a[ok, 1], 1e-9))))
    print('  SD  : mean |rel. dev.| = %5.1f %%' %
          (100 * np.mean(np.abs(a[ok, 6] - a[ok, 4]) / a[ok, 4])))

    if args.json:
        doc = json.load(open(args.json)) if os.path.exists(args.json) else {}
        key = args.key or ('r%.3f_a%02d_os%d' % (M.sensor_pitch_mm(args.dens),
                                                 args.angle, args.oversample))
        doc[key] = dict(
            r_mm=M.sensor_pitch_mm(args.dens), alpha=args.angle,
            oversample=args.oversample, nmc=args.nmc,
            n_region=int(c.size), n_interior=n_t, n_background=n_b,
            n_boundary=int(n_edge), d_um=float(d),
            rows=[[float(v) for v in r] for r in rows],
            dev_bias_gen_pct=float(100 * np.mean(np.abs(a[ok, 2] - a[ok, 1])
                                                 / np.maximum(a[ok, 1], 1e-9))),
            dev_sd_gen_pct=float(100 * np.mean(np.abs(a[ok, 5] - a[ok, 4])
                                               / a[ok, 4])),
            dev_bias_vr_pct=float(100 * np.mean(np.abs(a[ok, 3] - a[ok, 1])
                                                / np.maximum(a[ok, 1], 1e-9))),
            dev_sd_vr_pct=float(100 * np.mean(np.abs(a[ok, 6] - a[ok, 4])
                                              / a[ok, 4])))
        json.dump(doc, open(args.json, 'w'), indent=1)
        print('saved', key, '->', args.json)


if __name__ == '__main__':
    main()
