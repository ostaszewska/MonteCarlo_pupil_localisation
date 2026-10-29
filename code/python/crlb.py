# -*- coding: utf-8 -*-
"""Cramer-Rao lower bound for pupil-centre localisation.

The measurement is the greyscale sensor image before thresholding,
    y_k = c_k(x) + e_k,    e_k ~ N(0, sigma_n^2)  independent,
where c_k(x) is the coverage of pixel k by a pupil of known shape at position x. The
Fisher information for one coordinate is

    I(x) = (1 / sigma_n^2) * sum_k (d c_k / d x)^2 ,

and any unbiased estimator satisfies var(x) >= 1 / I(x). The derivative is taken
numerically by translating the rendered pupil. The bound assumes known shape and size
and sees the greyscale image; the binary centroid sees the thresholded image, so the
ratio between the two is the price of binarisation.

Writes results/crlb.json.
"""
import argparse
import json
import os

import numpy as np
from multiprocessing import Pool, cpu_count

import iris_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))
OUT = os.path.join(RES, 'crlb.json')
VERS = np.array([0.0, 1.0, 2.0])
OS = 5
DELTA_PX = 0.02              # translation used for the numerical derivative


def coverage(dens, angle, amp, ph, margins, shift=(0.0, 0.0), oversample=OS):
    """Block-averaged coverage map of one pupil translated by `shift` (metres)."""
    block = M.REDUCE * oversample
    dens_hr = dens * M.RES_MULT * oversample
    bnd, ycR, zcR = M.build_pupil_boundary(VERS, angle, amp, ph)
    Yb, Zb = bnd[:, 0] + shift[0], bnd[:, 1] + shift[1]
    cy_, cz_ = ycR + shift[0], zcR + shift[1]
    x0, x1, y0, y1 = margins
    nx = max(int(np.floor(1e3 * dens_hr * (x1 - x0))), block)
    ny = max(int(np.floor(1e3 * dens_hr * (y1 - y0))), block)
    PX, PY = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    ins = M.points_in_shape_radial(PX.ravel(), PY.ravel(), Yb, Zb,
                                   cy_, cz_).reshape(PX.shape).astype(float)
    return M._block_average(PX, PY, ins, block)


def one_shape(task):
    """Fisher information per unit sigma_n^2 for one pupil realisation."""
    seed, dens, angle, oversample = task
    rng = np.random.default_rng(int(seed))
    amp, ph = M.population_distortion(rng)
    bnd, ycR, zcR = M.build_pupil_boundary(VERS, angle, amp, ph)
    Yb, Zb = bnd[:, 0], bnd[:, 1]
    mX = M.MARGP * (Yb.max() - Yb.min())
    mY = M.MARGP * (Zb.max() - Zb.min())
    margins = (Yb.min() - mX * (rng.random() + 1.0),
               Yb.max() + mX * (rng.random() + 1.0),
               Zb.min() - mY * (rng.random() + 1.0),
               Zb.max() + mY * (rng.random() + 1.0))
    pitch = 1.0 / (dens * 1e3)                       # metres per sensor pixel
    d = DELTA_PX * pitch
    _, _, cxp = coverage(dens, angle, amp, ph, margins, (+d, 0.0), oversample)
    _, _, cxm = coverage(dens, angle, amp, ph, margins, (-d, 0.0), oversample)
    _, _, cyp = coverage(dens, angle, amp, ph, margins, (0.0, +d), oversample)
    _, _, cym = coverage(dens, angle, amp, ph, margins, (0.0, -d), oversample)
    if cxp.shape != cxm.shape or cyp.shape != cym.shape:
        return None
    gx = (cxp - cxm) / (2 * d)
    gy = (cyp - cym) / (2 * d)
    return float((gx ** 2).sum()), float((gy ** 2).sum())


def bound(pool, dens, angle, sn, nshape, seed, oversample=OS):
    ss = np.random.SeedSequence(seed).spawn(nshape)
    r = [x for x in pool.map(one_shape,
                             [(s.generate_state(1)[0], dens, angle, oversample)
                              for s in ss]) if x]
    if not r:
        return None
    a = np.array(r)
    sx = sn / np.sqrt(a[:, 0])
    sy = sn / np.sqrt(a[:, 1])
    per = np.sqrt((sx ** 2 + sy ** 2) / 2) * 1e6
    return dict(nshape=len(r), crlb_um=float(per.mean()),
                crlb_sd_um=float(per.std(ddof=1)), oversample=oversample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nshape', type=int, default=48)
    ap.add_argument('-j', '--nproc', type=int, default=None)
    args = ap.parse_args()
    conv = json.load(open(os.path.join(RES, 'converged_os5.json')))
    fine = json.load(open(os.path.join(RES, 'fine_pitch_base.json')))
    global FINE_BASE
    FINE_BASE = {round(row['r_mm'], 3): row['s_um'] for row in fine}
    doc = dict(meta=dict(nshape=args.nshape, oversample=OS, delta_px=DELTA_PX))

    with Pool(processes=args.nproc or cpu_count()) as pool:
        # (a) against the sampling pitch at the design noise level
        rows = []
        # The bound depends only on the noise-free coverage map, which is converged
        # on the coarse rendering grid at fine pitch (gci.py); the coarse grid is
        # therefore used below r = 0.25 mm/line.
        PITCH = ((2.000, 0.5, 'pitch/r2.000_a+00', OS), (1.429, 0.7, 'pitch/r1.429_a+00', OS),
                 (1.000, 1.0, 'pitch/r1.000_a+00', OS), (0.667, 1.5, 'pitch/r0.667_a+00', OS),
                 (0.500, 2.0, 'pitch/r0.500_a+00', OS), (0.333, 3.0, 'fine/r0.333', OS),
                 (0.250, 4.0, 'fine/r0.250', OS), (0.250, 4.0, 'fine/r0.250', 1),
                 (0.200, 5.0, None, 1), (0.167, 6.0, None, 1),
                 (0.125, 8.0, None, 1), (0.100, 10.0, None, 1))
        for r_mm, dens, key, os_ in PITCH:
            b = bound(pool, dens, 0.0, 0.10, args.nshape, 900 + int(1000 * r_mm),
                      oversample=os_)
            sim = conv.get(key, {}).get('s') if key else FINE_BASE.get(r_mm)
            b.update(r_mm=r_mm, sigma_n=0.10, sim_um=sim,
                     ratio=(sim / b['crlb_um']) if sim else None)
            rows.append(b)
            print('r = %.3f mm/line (OS %d): CRLB %8.2f um, simulated %8.2f um, '
                  'ratio %5.2f' % (r_mm, os_, b['crlb_um'], sim or float('nan'),
                                   b['ratio'] or float('nan')), flush=True)
        doc['pitch'] = rows

        # (b) against the noise level at the reference pitch
        rows = []
        for sn in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
            b = bound(pool, 1.0, 0.0, sn, args.nshape, 1900)
            sim = conv['noise/sn%.2f_a+00' % sn]['s']
            b.update(r_mm=1.0, sigma_n=sn, sim_um=sim, ratio=sim / b['crlb_um'])
            rows.append(b)
            print('sigma_n = %.2f: CRLB %8.2f um, simulated %8.2f um, ratio %5.2f'
                  % (sn, b['crlb_um'], sim, b['ratio']), flush=True)
        doc['noise'] = rows

    json.dump(doc, open(OUT, 'w'), indent=1)
    print('\nsaved', OUT)


if __name__ == '__main__':
    main()
