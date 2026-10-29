# -*- coding: utf-8 -*-
"""Fine-pitch sweep below r = 0.25 mm/line.

Extends the pitch sweep to r = 0.10 mm/line at alpha = 0, sigma_eff = 0.10, population
pupil shapes, on the base rendering grid. The pitch is the sensor pitch,
r = 1/DensCCD (iris_model.sensor_pitch_mm). Used by tremor.py.

Run from code/python/:
    python fine_pitch_sweep.py            # -> ../../results/fine_pitch_base.json
"""
import os
import numpy as np, time, json
from multiprocessing import Pool, cpu_count
import iris_model as M

RESULTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', '..', 'results'))

AX = np.array([0.0, 1.0, 2.0])
UM = 1e6


def _trial(task):
    seed, dens, noise = task
    rng = np.random.default_rng(seed)
    da, dp = M.population_distortion(rng)
    return M.single_trial(AX, 0.0, da, dp, dens, noise, rng)


if __name__ == '__main__':
    # dens -> r[mm/line] = 1/dens: 2->0.500 3->0.333 4->0.250 5->0.200
    #                            6->0.167 8->0.125 10->0.100
    dens_list = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    noise = 0.10
    rep = 3600
    ss = np.random.SeedSequence(20240606)
    rows = []
    t0 = time.time()
    with Pool(cpu_count()) as pool:
        for dens in dens_list:
            seeds = ss.spawn(rep)
            tasks = [(seeds[t], dens, noise) for t in range(rep)]
            tic = time.time()
            res = np.array(pool.map(_trial, tasks, chunksize=max(1, rep // (cpu_count() * 4))))
            good = res[~np.isnan(res).any(axis=1)]
            sx = good[:, 4].std(ddof=1) * UM
            sy = good[:, 5].std(ddof=1) * UM
            bx = (good[:, 4] - good[:, 0]).mean() * UM
            by = (good[:, 5] - good[:, 1]).mean() * UM
            r = M.sensor_pitch_mm(dens)
            s = 0.5 * (sx + sy)
            rows.append(dict(dens=dens, r_mm=r, pupil_px=M.IRIS_D * 1e3 / r,
                             s_um=s, sx=sx, sy=sy,
                             bias_um=float(np.hypot(bx, by)), s_over_r=s / (r * 1000.0),
                             n=int(len(good))))
            print('dens=%4.1f  r=%.4f mm/line  s=%6.1f um  s/r=%.3f px  bias=%5.1f um  (%d/%d) %.1fs'
                  % (dens, r, s, s / (r * 1000.0), np.hypot(bx, by), len(good), rep, time.time() - tic))
    print('total %.1f min' % ((time.time() - t0) / 60))
    dest = os.path.join(RESULTS, 'fine_pitch_base.json')
    json.dump(rows, open(dest, 'w'), indent=1)
    print('saved', dest)
