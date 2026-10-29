# -*- coding: utf-8 -*-
"""Is the per-frame pupil-localisation error temporally uncorrelated?

The narrowband tremor requirements (tremor.py) assume white centroid errors: a
per-sample standard deviation s at frame rate f_s gives a one-sided power spectral
density s^2/(f_s/2) only if successive errors are independent. Correlated errors raise
the noise floor in a narrow band.

Simulated trials are independent by construction, so the test needs real sequential
video. The localisation pipeline is run over consecutive LPW frames and the residual
against the hand annotation is autocorrelated:

    e(t) = centroid(t) - ground truth(t)

Subtracting the annotation removes the eye movement and leaves the localisation error.
Two conditions are run: the real frames, and the same frames with synthetic sensor
noise added (control).

Needs the LPW dataset and opencv; see LPW_REPRODUCE.md.

Run from code/python/:
    python lpw_autocorr.py <LPW_ROOT> [frames_per_video] [pupil_px]
Writes ../../results/lpw_autocorr.json
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import signal

try:
    import cv2
except ImportError:
    sys.exit('opencv-python is required: pip install opencv-python')

import iris_model as IM
from lpw_common import (segment_pupil_local, block_downsample, centroid_roi,
                         read_annotation)

LPW_ROOT = sys.argv[1] if len(sys.argv) > 1 else 'LPW'
RUN_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 200      # consecutive frames per video
PUPIL_LR = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
UM_PER_PX = IM.IRIS_D * 1e6 / PUPIL_LR
MAXLAG = 10
SN_CTRL = 0.10
RESULTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', '..', 'results'))


FS_LPW = 95.0        # LPW frame rate [Hz]; Nyquist 47.5 Hz
F_TREMOR = 84.0      # dominant ocular-tremor frequency


def psd_flatness(x, fs=FS_LPW):
    """How far the residual departs from a white spectrum in the upper half-band.

    A slow drift raises the variance and the lag-1 autocorrelation without changing
    the high-frequency power density, which is what the tremor requirement depends
    on. Returns the ratio of the mean PSD above fs/4 to the white-noise level
    var/(fs/2): 1.0 means white in that band.
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    nper = min(128, len(x))
    f, P = signal.welch(x, fs=fs, nperseg=nper, detrend='constant')
    white = x.var(ddof=1) / (fs / 2.0)
    hi = f >= fs / 4.0
    return float(P[hi].mean() / white), float(P[f <= fs / 20.0].mean() / white)


def acf(x, maxlag):
    """Autocorrelation of a mean-removed series, lags 1..maxlag."""
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return np.full(maxlag, np.nan)
    return np.array([np.dot(x[:-k], x[k:]) / denom for k in range(1, maxlag + 1)])


def run_video(avi, rng):
    txt = avi[:-4] + '.txt'
    if not os.path.exists(txt):
        return None
    ann = read_annotation(txt)
    valid = [k for k, (x, y) in enumerate(ann) if x > 1 and y > 1]
    if len(valid) < RUN_LEN:
        return None
    start = valid[len(valid) // 4]                 # a quarter in, away from the edges
    cap = cv2.VideoCapture(avi)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ex, ey, exn, eyn = [], [], [], []
    for t in range(RUN_LEN):
        ok, frame = cap.read()
        if not ok:
            break
        fi = start + t
        gx, gy = ann[fi]
        if not (gx > 1 and gy > 1):
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        mask = segment_pupil_local(g, (gx, gy))
        if mask is None:
            continue
        D = 2 * np.sqrt(mask.sum() / np.pi)
        factor = max(int(round(D / PUPIL_LR)), 1)
        lr = block_downsample(mask, factor)
        base = centroid_roi(lr > 0.5, (lr.shape[1] / 2, lr.shape[0] / 2), 1e9)
        if base is None:
            continue
        Dlr = 2 * np.sqrt((lr > 0.5).sum() / np.pi)
        gt = np.array([gx / factor, gy / factor])
        ex.append(base[0] - gt[0]); ey.append(base[1] - gt[1])
        noisy = centroid_roi((lr + rng.normal(0, SN_CTRL, lr.shape)) > 0.5, base, 1.0 * Dlr)
        if noisy is None:
            noisy = base
        exn.append(noisy[0] - gt[0]); eyn.append(noisy[1] - gt[1])
    cap.release()
    if len(ex) < 3 * MAXLAG:
        return None
    return (np.array(ex), np.array(ey), np.array(exn), np.array(eyn))


def main():
    videos = sorted(glob.glob(os.path.join(LPW_ROOT, '*', '*.avi')))
    if not videos:
        sys.exit('No .avi found under %r.' % LPW_ROOT)
    rng = np.random.default_rng(0)
    A = {'real': [], 'noisy': []}
    sds, lens = [], []
    global FLAT, LOWB
    FLAT, LOWB = [], []
    for avi in videos:
        r = run_video(avi, rng)
        if r is None:
            continue
        ex, ey, exn, eyn = r
        A['real'].append(0.5 * (acf(ex, MAXLAG) + acf(ey, MAXLAG)))
        A['noisy'].append(0.5 * (acf(exn, MAXLAG) + acf(eyn, MAXLAG)))
        hx, lx = psd_flatness(ex); hy, ly = psd_flatness(ey)
        FLAT.append(0.5 * (hx + hy)); LOWB.append(0.5 * (lx + ly))
        sds.append(np.sqrt((ex.std(ddof=1) ** 2 + ey.std(ddof=1) ** 2) / 2))
        lens.append(len(ex))
    if not A['real']:
        sys.exit('No usable run of consecutive frames.')
    nv = len(A['real'])
    out = {'n_videos': nv, 'pupil_px': PUPIL_LR, 'um_per_px': UM_PER_PX,
           'run_len': RUN_LEN, 'mean_usable_frames': float(np.mean(lens)),
           'sigma_n_control': SN_CTRL, 'maxlag': MAXLAG}
    print('temporal autocorrelation of the localisation error, %d LPW videos,' % nv)
    print('%.0f consecutive frames each on average, pupil %.0f px (%.0f um per px)'
          % (np.mean(lens), PUPIL_LR, UM_PER_PX))
    print('  residual SD against the annotation: %.2f px = %.0f um'
          % (np.mean(sds), np.mean(sds) * UM_PER_PX))
    print()
    print('  %-5s %14s %14s' % ('lag', 'real frames', 'with sensor noise'))
    for k in range(MAXLAG):
        r = np.array([a[k] for a in A['real']]); n = np.array([a[k] for a in A['noisy']])
        print('  %-5d %8.3f +-%.3f %8.3f +-%.3f'
              % (k + 1, np.nanmean(r), np.nanstd(r) / np.sqrt(nv),
                 np.nanmean(n), np.nanstd(n) / np.sqrt(nv)))
    for key in ('real', 'noisy'):
        M_ = np.array(A[key])
        out['acf_' + key] = np.nanmean(M_, axis=0).tolist()
        out['acf_%s_sem' % key] = (np.nanstd(M_, axis=0) / np.sqrt(nv)).tolist()
    # variance inflation of a block mean if the series is correlated
    r1 = np.nanmean([a[0] for a in A['real']])
    out['acf1_real'] = float(r1)
    out['variance_inflation'] = float((1 + r1) / (1 - r1)) if abs(r1) < 1 else float('nan')
    out['psd_upper_band_ratio'] = float(np.mean(FLAT))
    out['psd_low_band_ratio'] = float(np.mean(LOWB))
    out['fs_lpw'] = FS_LPW
    out['f_tremor'] = F_TREMOR
    print()
    print('  spectral distribution of the same residual (Welch, fs = %.0f Hz):' % FS_LPW)
    print('    power density below %.1f Hz, relative to white : %.2fx'
          % (FS_LPW / 20, np.mean(LOWB)))
    print('    power density above %.1f Hz, relative to white : %.2fx'
          % (FS_LPW / 4, np.mean(FLAT)))
    print('    (1.0 = white. The tremor sits at %.0f Hz, above the %.1f Hz Nyquist of'
          % (F_TREMOR, FS_LPW / 2))
    print('     this dataset, so the upper band is the closest observable proxy.)')
    print()
    print('  lag-1 autocorrelation on real frames : %.3f' % r1)
    print('  implied variance inflation of a long average : %.2fx'
          % out['variance_inflation'])
    print('  equivalently the noise floor in a narrow band is %.0f%% higher than white'
          % (100 * (np.sqrt(max(out['variance_inflation'], 0)) - 1)))
    with open(os.path.join(RESULTS, 'lpw_autocorr.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('saved', os.path.join(RESULTS, 'lpw_autocorr.json'))


if __name__ == '__main__':
    main()
