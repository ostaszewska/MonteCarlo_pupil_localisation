# -*- coding: utf-8 -*-
"""Boundary discretisation and point-in-shape test.

The pupil boundary is a polygon of 100 vertices. A sensor-grid point is classified
as inside it either by radial interpolation in angle between the two bracketing
vertices (iris_model.points_in_shape_radial, used for every result) or by exact
ray casting (iris_model.points_in_polygon). This script checks that

  (i)  the rendering grid does not matter at this level: radial test, 100 vertices,
       oversampling 1, 3 and 5, at sigma_eff = 0.05;
  (ii) neither the number of vertices nor the choice of test changes u: both tests
       at 100, 200 and 400 vertices, oversampling 1, at sigma_eff = 0.05 and 0.10.

All runs at r = 1.00 mm/line, alpha = 0, 1500 trials each.

Run from code/python/:
    python boundary_vertices.py         # -> ../../results/boundary_vertices.json
"""
import numpy as np, json, time, os
from multiprocessing import Pool, cpu_count
import iris_model as M

AX = np.array([0.0, 1.0, 2.0]); UM = 1e6


def trial(amp, phase, noise, rng, OS, test, nb):
    M.DENS = nb
    block = M.REDUCE * OS; dens_hr = 1.0 * M.RES_MULT * OS
    bndYZ, ycR, zcR = M.build_pupil_boundary(AX, 0.0, amp, phase)
    Yb, Zb = bndYZ[:, 0], bndYZ[:, 1]
    mX = M.MARGP * (Yb.max() - Yb.min()); mY = M.MARGP * (Zb.max() - Zb.min())
    sx0 = Yb.min() - mX * (rng.random() + 1); sx1 = Yb.max() + mX * (rng.random() + 1)
    sy0 = Zb.min() - mY * (rng.random() + 1); sy1 = Zb.max() + mY * (rng.random() + 1)
    nx = max(int(np.floor(1e3 * dens_hr * (sx1 - sx0))), block)
    ny = max(int(np.floor(1e3 * dens_hr * (sy1 - sy0))), block)
    x = np.linspace(sx0, sx1, nx); y = np.linspace(sy0, sy1, ny)
    PX, PY = np.meshgrid(x, y)
    if test == 'radial':
        ins = M.points_in_shape_radial(PX.ravel(), PY.ravel(), Yb, Zb, ycR, zcR)
    else:
        ins = M.points_in_polygon(PX.ravel(), PY.ravel(), Yb, Zb)
    inside = ins.reshape(PX.shape).astype(float)
    PXr, PYr, Vr = M._block_average(PX, PY, inside, block)
    Vr = (Vr + rng.standard_normal(Vr.shape) * noise >= M.THRESH).astype(float)
    sr = Vr.sum()
    if sr > 0:
        return ycR, zcR, (PXr * Vr).sum() / sr, (PYr * Vr).sum() / sr
    return ycR, zcR, np.nan, np.nan


def _w(task):
    seed, noise, OS, test, nb = task
    rng = np.random.default_rng(seed)
    M.DENS = nb
    da, dp = M.population_distortion(rng)
    return trial(da, dp, noise, rng, OS, test, nb)


def run(pool, noise, OS, test, nb, rep, ss):
    seeds = ss.spawn(rep)
    res = np.array(pool.map(_w, [(seeds[i], noise, OS, test, nb) for i in range(rep)], chunksize=64))
    g = res[~np.isnan(res).any(axis=1)]
    return 0.5 * (g[:, 2].std(ddof=1) + g[:, 3].std(ddof=1)) * UM


def summary(out):
    """Largest relative differences between tests and between vertex counts [%]."""
    sc = out['scheme']
    tests = [abs(sc['radial_n%d_sn%.2f' % (n, z)] / sc['polygon_n%d_sn%.2f' % (n, z)] - 1)
             for n in (100, 200, 400) for z in (0.05, 0.10)]
    verts = [abs(sc['%s_n400_sn%.2f' % (t, z)] / sc['%s_n100_sn%.2f' % (t, z)] - 1)
             for t in ('radial', 'polygon') for z in (0.05, 0.10)]
    return {'radial_vs_raycasting_max_pct': 100 * max(tests),
            'n100_vs_n400_max_pct': 100 * max(verts)}


if __name__ == '__main__':
    ss = np.random.SeedSequence(909090); rep = 1500
    out = {'os_check': {}, 'scheme': {}}
    t0 = time.time()
    with Pool(cpu_count()) as pool:
        print('\n(i) OS grid-independence (radial, n=100, sigma_n=0.05):')
        for OS in (1, 3, 5):
            s = run(pool, 0.05, OS, 'radial', 100, rep, ss)
            out['os_check']['OS%d' % OS] = s
            print('   OS=%d : s=%.1f um' % (OS, s))
        print('\n(ii) scheme x boundary density (OS=1):')
        for nb in (100, 200, 400):
            for test in ('radial', 'polygon'):
                for nz in (0.05, 0.10):
                    s = run(pool, nz, 1, test, nb, rep, ss)
                    out['scheme']['%s_n%d_sn%.2f' % (test, nb, nz)] = s
                    print('   %-7s n=%3d sigma_n=%.2f : s=%.1f um' % (test, nb, nz, s))
    dest = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '..', '..', 'results', 'boundary_vertices.json'))
    out['summary'] = summary(out)
    print(out['summary'])
    json.dump(out, open(dest, 'w'), indent=1)
    print('\ntotal %.1f min -> saved %s' % ((time.time() - t0) / 60, dest))
