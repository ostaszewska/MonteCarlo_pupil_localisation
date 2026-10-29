# -*- coding: utf-8 -*-
"""Threshold bias against the offset of the analysed window.

The two-class bias model (vrcheck.py, VR30) makes the bias proportional to d, the
displacement from the pupil centroid to the centroid of the background pixels inside
the analysed window, and d follows the placement of the window. The rendering grid
does not contain a whole number of sensor pixels, so a few rendering cells are left
over; which end they are dropped from decides where the analysed window sits. Dropping a fraction q of the leftover
at the low-index end and the rest at the high-index end moves the window from one
extreme (q = 0, the tiling of every other result) through centred (q = 0.5) to the
other (q = 1), and d reverses on the way.

For each q, noise level and gaze angle the mean bias vector b and the mean offset d
are recorded (3000 trials per cell, circular and population pupil shapes). The
summary, at sigma_eff = 0.50 for the population shapes, gives the mean |d| of the
q = 0 tiling, the range of |b|/|d| and of |d| over the off-centre tilings, the
median angle between b and d, and |b| for the centred and for the extreme tilings.

Run from code/python/:
    python window_offset.py              # ~1 h; writes ../../results/window_offset.json
    python window_offset.py --summary    # recompute the summary from the stored cells
"""
import json
import os

import numpy as np
from multiprocessing import Pool, cpu_count

import iris_model as M

AX = np.array([0., 1., 2.])
UM = 1e6
DENS = 1.0
RESULTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', '..', 'results'))
QS = (0.0, 0.25, 0.5, 0.75, 1.0)          # fraction of the leftover dropped at the low end
NOISE = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
ANGLES = (0.0, 30.0, 60.0)
N = 3000


def trial(task):
    """One trial at a given leftover split. Returns the centroid error and the offset."""
    seed, noise, q, shape, angle = task
    rng = np.random.default_rng(seed)
    amp, ph = (np.zeros(7), np.zeros(7)) if shape == 'circ' else M.population_distortion(rng)
    block = M.REDUCE
    bnd, ycR, zcR = M.build_pupil_boundary(AX, angle, amp, ph)
    Yb, Zb = bnd[:, 0], bnd[:, 1]
    mX = M.MARGP * (Yb.max() - Yb.min()); mY = M.MARGP * (Zb.max() - Zb.min())
    sx0 = Yb.min() - mX * (rng.random() + 1); sx1 = Yb.max() + mX * (rng.random() + 1)
    sy0 = Zb.min() - mY * (rng.random() + 1); sy1 = Zb.max() + mY * (rng.random() + 1)
    nx = max(int(np.floor(1e3 * DENS * M.RES_MULT * (sx1 - sx0))), block)
    ny = max(int(np.floor(1e3 * DENS * M.RES_MULT * (sy1 - sy0))), block)
    PX, PY = np.meshgrid(np.linspace(sx0, sx1, nx), np.linspace(sy0, sy1, ny))
    ins = M.points_in_shape_radial(PX.ravel(), PY.ravel(), Yb, Zb,
                                   ycR, zcR).reshape(PX.shape).astype(float)
    nr, nc = ny // block, nx // block
    oy = int(round(q * (ny - nr * block)))       # q of the leftover goes to the low end
    ox = int(round(q * (nx - nc * block)))
    sl = lambda A: A[oy:oy + nr * block, ox:ox + nc * block] \
                    .reshape(nr, block, nc, block).mean(axis=(1, 3))
    XTr, YTr, Vr = sl(PX), sl(PY), sl(ins)
    Vb = (Vr + rng.standard_normal(Vr.shape) * noise >= M.THRESH).astype(float)
    k = Vb.sum()
    if k == 0:
        return (np.nan,) * 4
    return ((XTr * Vb).sum() / k - ycR, (YTr * Vb).sum() / k - zcR,
            XTr.mean() - ycR, YTr.mean() - zcR)


def summary(cells, sn=0.50):
    """Summary statistics for the population shapes at one noise level."""
    pop = [c for c in cells if c['shape'] == 'pop']
    at = [c for c in pop if abs(c['sigma_n'] - sn) < 1e-9]
    off = [c for c in at if c['q'] != 0.5]
    ext = [c for c in at if c['q'] in (0.0, 1.0)]
    ctr = [c for c in at if c['q'] == 0.5]

    def angle(c):
        b = np.array([c['bx_um'], c['by_um']]); d = np.array([c['dx_um'], c['dy_um']])
        return float(np.degrees(np.arccos(min(1.0, abs(b @ d) / np.linalg.norm(b)
                                              / np.linalg.norm(d)))))
    same_sense = all(c['bx_um'] * c['dx_um'] + c['by_um'] * c['dy_um'] > 0 for c in off)
    ratio = [c['b_um'] / c['d_um'] for c in off]
    dd = [c['d_um'] for c in off]
    return {
        'sigma_n': sn,
        'mean_abs_d_q0_um': float(np.mean([c['d_um'] for c in pop
                                           if c['q'] == 0.0 and c['alpha'] == 0.0])),
        'ratio_b_over_d': [float(min(ratio)), float(max(ratio))],
        'abs_d_um': [float(min(dd)), float(max(dd))],
        'abs_d_factor': float(max(dd) / min(dd)),
        'median_angle_b_d_deg': float(np.median([angle(c) for c in off])),
        'b_reverses_with_d': bool(same_sense),
        'abs_b_centred_um': [float(min(c['b_um'] for c in ctr)),
                             float(max(c['b_um'] for c in ctr))],
        'abs_b_extremes_um': [float(min(c['b_um'] for c in ext)),
                              float(max(c['b_um'] for c in ext))],
    }


def write_summary(out):
    out['summary'] = summary(out['cells'])
    out['d_published_um'] = out['summary']['mean_abs_d_q0_um']
    for k, v in out['summary'].items():
        print('  %-24s %s' % (k, v))
    with open(os.path.join(RESULTS, 'window_offset.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('saved', os.path.join(RESULTS, 'window_offset.json'))


def main():
    ss = np.random.SeedSequence(20260805)
    out = {'N': N, 'r_mm_per_line': M.sensor_pitch_mm(DENS), 'q': list(QS),
           'sigma_n': list(NOISE), 'cells': []}
    print('threshold bias against analysis-window offset, r = %.2f mm/line, alpha = 0'
          % M.sensor_pitch_mm(DENS))
    print('q is the fraction of the leftover rendering cells dropped at the low end;')
    print('q = 0 is the tiling of every other result, q = 0.5 is centred.\n')

    with Pool(cpu_count()) as pool:
        for shape in ('circ', 'pop'):
            print('--- pupil shape: %s ---' % shape)
            print('  %-6s %8s %9s %9s' % ('q', '|d| [um]', 'range of', 'f = |b|/|d|'))
            print('  %-6s %8s %9s %9s' % ('', '', '|b| [um]', 'at sn=0.50'))
            for q in QS:
                mags, ds = [], []
                for sn in NOISE:
                    for al in ANGLES:
                        seeds = ss.spawn(N)
                        res = np.array(pool.map(trial, [(s, sn, q, shape, al) for s in seeds],
                                                chunksize=64))
                        g = res[~np.isnan(res).any(axis=1)]
                        bx, by = g[:, 0].mean() * UM, g[:, 1].mean() * UM
                        dx, dy = g[:, 2].mean() * UM, g[:, 3].mean() * UM
                        out['cells'].append(dict(shape=shape, q=q, sigma_n=sn, alpha=al,
                                                 b_um=float(np.hypot(bx, by)),
                                                 d_um=float(np.hypot(dx, dy)),
                                                 bx_um=float(bx), by_um=float(by),
                                                 dx_um=float(dx), dy_um=float(dy)))
                        if al == 0.0:
                            mags.append(float(np.hypot(bx, by)))
                            ds.append(float(np.hypot(dx, dy)))
                print('  %-6.2f %8.0f %9s %9.2f'
                      % (q, np.mean(ds), '%.0f-%.0f' % (min(mags), max(mags)),
                         mags[-1] / ds[-1] if ds[-1] > 0 else float('nan')))

            print()

    write_summary(out)


if __name__ == '__main__':
    import sys
    if '--summary' in sys.argv:
        with open(os.path.join(RESULTS, 'window_offset.json'), encoding='utf-8') as f:
            doc = json.load(f)
        for k in ('fit', 'B_max_equivalent_um'):
            doc.pop(k, None)
        write_summary(doc)
    else:
        main()
