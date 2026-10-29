# -*- coding: utf-8 -*-
"""Binary against intensity-weighted centroid on degraded images.

Two departures from the idealised scene, both estimators on the identical image:

  illum   a linear illumination gradient across the analysis window (+-15 %, +-30 %)
  glint   a corneal catch-light inside the pupil, a disc of zero pupil signal, which
          the binary pipeline removes by hole filling and the greyscale one cannot

Writes results/robustness.json.
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
OUT = os.path.join(RES, 'robustness.json')
VERS = np.array([0.0, 1.0, 2.0])
OS = 5


def trial(task):
    """Returns (cx, cy, wx, wy) errors in metres: binary centroid with hole filling,
    and threshold-gated intensity-weighted centroid, on the identical image."""
    seed, dens, sn, angle, mech, par = task
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

    if mech == 'glint':
        # a catch-light of radius `par` sensor pixels, placed one pixel from the
        # pupil centre in a random direction, removing the pupil signal under it
        pitch_hr = float(PX[0, 1] - PX[0, 0])
        rad = par * block * pitch_hr
        th = rng.random() * 2 * np.pi
        gx = ycR + np.cos(th) * block * pitch_hr
        gy = zcR + np.sin(th) * block * pitch_hr
        ins[np.hypot(PX - gx, PY - gy) <= rad] = 0.0

    XT, YT, V = M._block_average(PX, PY, ins, block)

    if mech == 'illum':
        # linear gradient of the pupil-iris contrast, +-par across the window
        w = XT.max() - XT.min()
        V = V * (1.0 + par * (2.0 * (XT - XT.min()) / max(w, 1e-12) - 1.0))

    V = V + rng.standard_normal(V.shape) * sn
    B = V >= M.THRESH
    out = [np.nan] * 4
    if B.sum() > 0:
        # the binary pipeline fills holes, as the real-image pipeline does, which
        # removes a catch-light entirely
        Bf = ndi.binary_fill_holes(B)
        k = Bf.sum()
        if k:
            out[0] = XT[Bf].mean() - ycR
            out[1] = YT[Bf].mean() - zcR
        wt = np.where(B, np.clip(V, 0.0, None), 0.0)
        sw = wt.sum()
        if sw > 0:
            out[2] = (XT * wt).sum() / sw - ycR
            out[3] = (YT * wt).sum() / sw - zcR
    return tuple(out)


def stats(a, ix):
    e = a[:, [ix, ix + 1]]
    e = e[~np.isnan(e).any(axis=1)]
    if len(e) < 10:
        return None
    ex, ey = e[:, 0] * 1e6, e[:, 1] * 1e6
    return dict(n=int(len(e)),
                u=round(float(np.sqrt((ex.std(ddof=1) ** 2
                                       + ey.std(ddof=1) ** 2) / 2)), 1),
                bias=round(float(np.hypot(ex.mean(), ey.mean())), 1),
                rmse=round(float(np.sqrt((ex ** 2 + ey ** 2).mean() / 2)), 1))


def point(pool, rep, sn, mech, par, seed):
    ss = np.random.SeedSequence(seed).spawn(rep)
    a = np.array(pool.map(trial, [(s.generate_state(1)[0], 1.0, sn, 0.0, mech, par)
                                  for s in ss], chunksize=16))
    return dict(centroid=stats(a, 0), weighted=stats(a, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rep', type=int, default=1800)
    ap.add_argument('-j', '--nproc', type=int, default=None)
    args = ap.parse_args()
    doc = dict(meta=dict(rep=args.rep, oversample=OS, r_mm=1.0, alpha=0.0))
    cases = [('none', None, 'ideal two-level scene'),
             ('illum', 0.15, 'illumination gradient, +-15 %'),
             ('illum', 0.30, 'illumination gradient, +-30 %'),
             ('glint', 0.30, 'catch-light of 0.3 px radius'),
             ('glint', 0.50, 'catch-light of 0.5 px radius')]
    t0 = time.time()
    with Pool(processes=args.nproc or cpu_count()) as pool:
        for i, (mech, par, label) in enumerate(cases):
            row = {}
            for sn in (0.05, 0.10, 0.15):
                row['sn%.2f' % sn] = point(pool, args.rep, sn, mech, par,
                                           6100 + 7 * i + int(100 * sn))
                p = row['sn%.2f' % sn]
                print('%-32s sn=%.2f  centroid u=%6.1f b=%5.1f | weighted u=%6.1f '
                      'b=%5.1f' % (label, sn, p['centroid']['u'], p['centroid']['bias'],
                                   p['weighted']['u'], p['weighted']['bias']),
                      flush=True)
            doc['%s_%s' % (mech, par)] = dict(label=label, **row)
            json.dump(doc, open(OUT, 'w'), indent=1)
    print('\n%.1f min -> %s' % ((time.time() - t0) / 60, OUT))


if __name__ == '__main__':
    main()
