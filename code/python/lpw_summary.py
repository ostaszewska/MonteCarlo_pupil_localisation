# -*- coding: utf-8 -*-
"""Robust summary of the LPW verification of the coverage-fraction bias model.

The per-frame relative deviation is unbounded where the measured bias passes through
zero, which it does on a few frames near the dip below the noise knee. The summary
statistic is therefore the agreement of the frame-averaged curves; the per-frame spread
is given by its median and interquartile range.
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))


def summarise(px):
    d = json.load(open(os.path.join(RES, 'lpw_coverage_bias_%dpx.json' % px)))
    F = d['frames']
    SN = np.array(d['summary']['sigma_n'])
    P = np.array([f['b_pred_um'] for f in F])
    M = np.array([f['b_meas_um'] for f in F])
    ok = np.isfinite(P) & np.isfinite(M)

    mp = np.array([P[ok[:, j], j].mean() for j in range(len(SN))])
    mm = np.array([M[ok[:, j], j].mean() for j in range(len(SN))])
    curve_rel = (mp - mm) / mm

    rel = np.where(ok & (M > 0), (P - M) / np.where(M > 0, M, np.nan), np.nan)
    r = np.abs(rel[np.isfinite(rel)])

    out = dict(
        pupil_px=px, r_mm=d['summary']['r_mm_per_line'],
        n_frames=d['summary']['n_frames'],
        curve_pred_um=mp.tolist(), curve_meas_um=mm.tolist(),
        curve_rel_pct=(100 * curve_rel).tolist(),
        curve_mean_abs_rel_pct=float(100 * np.mean(np.abs(curve_rel))),
        curve_max_abs_rel_pct=float(100 * np.max(np.abs(curve_rel))),
        frame_median_abs_rel_pct=float(100 * np.median(r)),
        frame_q75_abs_rel_pct=float(100 * np.percentile(r, 75)),
        frame_q90_abs_rel_pct=float(100 * np.percentile(r, 90)))
    print('--- pupil %d px, r = %.2f mm/line, %d frames ---' % (px, out['r_mm'],
                                                               out['n_frames']))
    print('%6s %10s %10s %8s' % ('sn', 'pred[um]', 'meas[um]', 'dev[%]'))
    for j, sn in enumerate(SN):
        print('%6.2f %10.1f %10.1f %+8.1f' % (sn, mp[j], mm[j], 100 * curve_rel[j]))
    print('frame-averaged curves agree to %.1f %% on average, %.1f %% at worst'
          % (out['curve_mean_abs_rel_pct'], out['curve_max_abs_rel_pct']))
    print('per-frame |deviation|: median %.1f %%, 75th pct %.1f %%, 90th pct %.1f %%'
          % (out['frame_median_abs_rel_pct'], out['frame_q75_abs_rel_pct'],
             out['frame_q90_abs_rel_pct']))
    print()
    return out


if __name__ == '__main__':
    doc = {}
    for px in (5, 25):
        if os.path.exists(os.path.join(RES, 'lpw_coverage_bias_%dpx.json' % px)):
            doc['px%d' % px] = summarise(px)
    json.dump(doc, open(os.path.join(RES, 'lpw_coverage_bias_summary.json'), 'w'), indent=1)
    print('saved results/lpw_coverage_bias_summary.json')
