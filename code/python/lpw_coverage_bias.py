# -*- coding: utf-8 -*-
"""Verification of the coverage-fraction bias model on real pupil geometry.

vrcheck.py compares the simulation with the bias model on simulated coverage maps.
Here the coverage map comes from a real, irregular, partly occluded infrared pupil (one
segmented LPW frame); the coverage-fraction bias model is evaluated on it and compared
with the bias measured on the same frame over noise realisations. The noise is Gaussian
and synthetic on both sides, so the noise model itself is not tested.

For each frame and noise level it records
    b_pred   the coverage-fraction bias model evaluated on the measured coverage map,
    b_meas   the bias measured over MREAL noise realisations of the same map,
    |d|      the offset from the pupil centroid to the centroid of the background
             pixels of the analysed region, the quantity the law is proportional to,
and the segmentation-against-annotation statistics.

Usage:  python lpw_coverage_bias.py <LPW_ROOT> [frames_per_video] [pupil_px]
Writes results/lpw_coverage_bias_<pupil_px>px.json.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.special import ndtr            # standard normal CDF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iris_model as IM
from lpw_common import (segment_pupil_local, block_downsample, read_annotation,
                        sample_frame)

LPW_ROOT = sys.argv[1] if len(sys.argv) > 1 else 'LPW'
FRAMES_PER_VIDEO = int(sys.argv[2]) if len(sys.argv) > 2 else 3
PUPIL_LR = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

SN = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
MREAL = 400                     # noise realisations per frame and noise level
T = IM.THRESH                   # 0.5
UM_PER_PX = IM.IRIS_D * 1e6 / PUPIL_LR       # low-resolution pixel -> micrometres
R_MM = UM_PER_PX / 1e3
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))
OUT = os.path.join(RES, 'lpw_coverage_bias_%dpx.json' % int(PUPIL_LR))


def frame_stats(mask, rng):
    """One frame. Returns a dict, or None when the frame yields nothing usable."""
    area_hr = mask.sum()
    if area_hr < 50:
        return None
    D_hr = 2.0 * np.sqrt(area_hr / np.pi)
    factor = max(int(round(D_hr / PUPIL_LR)), 1)
    lr = block_downsample(mask, factor)                 # coverage map c_k in [0, 1]
    if lr.size < 25:
        return None

    ys, xs = np.nonzero(mask)
    # true pupil centre: centroid of the high-resolution mask, in low-resolution
    # pixel coordinates. The coverage-fraction bias model predicts the departure
    # from this point.
    tx = (xs.mean() + 0.5) / factor - 0.5
    ty = (ys.mean() + 0.5) / factor - 0.5

    fg = lr >= T
    if fg.sum() < 3:
        return None
    Y, X = np.mgrid[0:lr.shape[0], 0:lr.shape[1]].astype(float)
    bx, by = X[fg].mean(), Y[fg].mean()
    half = 2.0 * np.sqrt(fg.sum() / np.pi)              # one pupil diameter, as the
    roi = (np.abs(X - bx) <= half) & (np.abs(Y - by) <= half)   # tracker's window
    if roi.sum() < 9:
        return None

    c = lr[roi]
    xr, yr = X[roi], Y[roi]
    obj = c >= T
    if obj.sum() < 3 or (~obj).sum() < 3:
        return None
    # d: from the object centroid to the centroid of the background pixels of the
    # analysed region, the definition of Van Rheeden and Jones
    dx = xr[~obj].mean() - xr[obj].mean()
    dy = yr[~obj].mean() - yr[obj].mean()

    out = dict(factor=int(factor), D_hr_px=float(D_hr), n_roi=int(roi.sum()),
               n_interior=int((c >= 0.999).sum()), n_boundary=int(((c > 0.001) &
                                                                   (c < 0.999)).sum()),
               n_background=int((c <= 0.001).sum()),
               d_um=float(np.hypot(dx, dy) * UM_PER_PX),
               b_pred_um=[], b_meas_um=[], b_pred_vec=[], b_meas_vec=[])

    for sn in SN:
        # ---- coverage-fraction bias model on the measured coverage map
        p = ndtr((c - T) / sn)
        sp = p.sum()
        if sp <= 0:
            out['b_pred_um'].append(float('nan'))
            out['b_meas_um'].append(float('nan'))
            out['b_pred_vec'].append([float('nan')] * 2)
            out['b_meas_vec'].append([float('nan')] * 2)
            continue
        ex, ey = (xr * p).sum() / sp, (yr * p).sum() / sp
        pvx, pvy = (ex - tx) * UM_PER_PX, (ey - ty) * UM_PER_PX

        # ---- measured on the same map, MREAL noise realisations, fully vectorised
        noisy = c[None, :] + rng.standard_normal((MREAL, c.size)) * sn
        b = noisy >= T
        k = b.sum(axis=1)
        good = k > 0
        if good.sum() < 10:
            out['b_pred_um'].append(float(np.hypot(pvx, pvy)))
            out['b_meas_um'].append(float('nan'))
            out['b_pred_vec'].append([pvx, pvy])
            out['b_meas_vec'].append([float('nan')] * 2)
            continue
        mx = (b[good] * xr[None, :]).sum(axis=1) / k[good]
        my = (b[good] * yr[None, :]).sum(axis=1) / k[good]
        mvx, mvy = (mx.mean() - tx) * UM_PER_PX, (my.mean() - ty) * UM_PER_PX

        out['b_pred_um'].append(float(np.hypot(pvx, pvy)))
        out['b_meas_um'].append(float(np.hypot(mvx, mvy)))
        out['b_pred_vec'].append([float(pvx), float(pvy)])
        out['b_meas_vec'].append([float(mvx), float(mvy)])
    return out


def main():
    videos = sorted(glob.glob(os.path.join(LPW_ROOT, '*', '*.avi')))
    if not videos:
        sys.exit('No .avi under %r.' % LPW_ROOT)
    rng = np.random.default_rng(20260819)
    frames, seg_err_px, seg_err_frac = [], [], []
    for n, avi in enumerate(videos, 1):
        txt = avi[:-4] + '.txt'
        if not os.path.exists(txt):
            continue
        ann = read_annotation(txt)
        valid = [k for k, (x, y) in enumerate(ann) if x > 1 and y > 1]
        if not valid:
            continue
        pick = np.unique(np.linspace(0, len(valid) - 1,
                                     FRAMES_PER_VIDEO).round().astype(int))
        for k in pick:
            fi = valid[k]
            g = sample_frame(avi, fi)
            if g is None:
                continue
            gx, gy = ann[fi]
            mask = segment_pupil_local(g, (gx, gy))
            if mask is None:
                continue
            ys, xs = np.nonzero(mask)
            e = float(np.hypot(xs.mean() - gx, ys.mean() - gy))
            D_hr = 2.0 * np.sqrt(mask.sum() / np.pi)
            f = frame_stats(mask, rng)
            if f is None:
                continue
            seg_err_px.append(e)
            seg_err_frac.append(e / D_hr)
            f['video'] = os.path.relpath(avi, LPW_ROOT).replace('\\', '/')
            f['frame'] = int(fi)
            f['seg_err_px'] = e
            f['seg_err_frac_of_D'] = e / D_hr
            frames.append(f)
        print('  [%2d/%2d] %s  frames kept: %d'
              % (n, len(videos), os.path.relpath(avi, LPW_ROOT), len(frames)),
              flush=True)

    if not frames:
        sys.exit('No usable frame.')

    P = np.array([f['b_pred_um'] for f in frames])
    Mn = np.array([f['b_meas_um'] for f in frames])
    d = np.array([f['d_um'] for f in frames])

    ok = np.isfinite(P) & np.isfinite(Mn) & (Mn > 0)
    rel = np.where(ok, (P - Mn) / np.where(Mn == 0, np.nan, Mn), np.nan)

    per_sn = []
    for j, sn in enumerate(SN):
        col = rel[:, j]
        col = col[np.isfinite(col)]
        per_sn.append(dict(sigma_n=float(sn),
                           n=int(col.size),
                           median_rel_pct=float(np.median(col) * 100),
                           mean_abs_rel_pct=float(np.mean(np.abs(col)) * 100),
                           p95_abs_rel_pct=float(np.percentile(np.abs(col), 95) * 100),
                           mean_pred_um=float(np.nanmean(P[:, j])),
                           mean_meas_um=float(np.nanmean(Mn[:, j]))))
        print('sn=%.2f  n=%3d  pred %7.1f  meas %7.1f  median rel %+6.1f %%  '
              'mean|rel| %5.1f %%'
              % (sn, col.size, np.nanmean(P[:, j]), np.nanmean(Mn[:, j]),
                 np.median(col) * 100, np.mean(np.abs(col)) * 100))

    allrel = rel[np.isfinite(rel)]
    # |b|/|d| across frames at the high-noise point, the constancy the law requires
    j50 = int(np.argmin(np.abs(SN - 0.50)))
    ratio = Mn[:, j50] / d
    ratio = ratio[np.isfinite(ratio)]

    summary = dict(
        pupil_px=PUPIL_LR, r_mm_per_line=R_MM, um_per_px=UM_PER_PX,
        n_frames=len(frames), n_videos=len(set(f['video'] for f in frames)),
        mreal=MREAL, sigma_n=SN.tolist(),
        overall_mean_abs_rel_pct=float(np.mean(np.abs(allrel)) * 100),
        overall_median_abs_rel_pct=float(np.median(np.abs(allrel)) * 100),
        overall_p95_abs_rel_pct=float(np.percentile(np.abs(allrel), 95) * 100),
        per_sigma=per_sn,
        b_over_d_at_050=dict(median=float(np.median(ratio)),
                             iqr=[float(np.percentile(ratio, 25)),
                                  float(np.percentile(ratio, 75))],
                             cv_pct=float(100 * np.std(ratio) / np.mean(ratio))),
        d_um=dict(median=float(np.median(d)),
                  p05=float(np.percentile(d, 5)),
                  p95=float(np.percentile(d, 95))),
        roi_composition=dict(
            interior=float(np.mean([f['n_interior'] for f in frames])),
            boundary=float(np.mean([f['n_boundary'] for f in frames])),
            background=float(np.mean([f['n_background'] for f in frames]))),
        annotation=dict(
            median_px=float(np.median(seg_err_px)),
            p95_px=float(np.percentile(seg_err_px, 95)),
            mean_px=float(np.mean(seg_err_px)),
            max_px=float(np.max(seg_err_px)),
            median_frac_of_D=float(np.median(seg_err_frac)),
            p95_frac_of_D=float(np.percentile(seg_err_frac, 95))))

    json.dump(dict(summary=summary, frames=frames), open(OUT, 'w'), indent=1)
    print()
    print('frames %d from %d videos' % (summary['n_frames'], summary['n_videos']))
    print('coverage-fraction model against the measured bias, all frames and noise levels:')
    print('   mean |relative deviation| = %.1f %%, median %.1f %%, 95th pct %.1f %%'
          % (summary['overall_mean_abs_rel_pct'],
             summary['overall_median_abs_rel_pct'],
             summary['overall_p95_abs_rel_pct']))
    print('|b|/|d| at sigma_n = 0.50: median %.2f (IQR %.2f-%.2f), CV %.0f %%'
          % (summary['b_over_d_at_050']['median'],
             summary['b_over_d_at_050']['iqr'][0],
             summary['b_over_d_at_050']['iqr'][1],
             summary['b_over_d_at_050']['cv_pct']))
    print('|d| over the frames: median %.0f um (5-95 pct %.0f-%.0f)'
          % (summary['d_um']['median'], summary['d_um']['p05'],
             summary['d_um']['p95']))
    print('ROI composition (mean per frame): %.0f interior, %.0f boundary, '
          '%.0f background' % (summary['roi_composition']['interior'],
                               summary['roi_composition']['boundary'],
                               summary['roi_composition']['background']))
    print('segmentation vs annotation: median %.1f px, 95th pct %.1f px, '
          'max %.1f px; %.3f and %.3f of the pupil diameter'
          % (summary['annotation']['median_px'], summary['annotation']['p95_px'],
             summary['annotation']['max_px'],
             summary['annotation']['median_frac_of_D'],
             summary['annotation']['p95_frac_of_D']))
    print('\nsaved', OUT)


if __name__ == '__main__':
    main()
