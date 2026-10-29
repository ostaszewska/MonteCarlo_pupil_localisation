# -*- coding: utf-8 -*-
"""Real-image validation of the centroid model on the LPW dataset.

Labelled Pupils in the Wild (LPW), Tonsen, Zhang, Sugano & Bulling, ETRA 2016.
  Dataset DOI: 10.18419/darus-3237  (https://darus.uni-stuttgart.de)
  *** You must accept the LPW data agreement and download it yourself. ***

Expected layout (as distributed):  <LPW_ROOT>/<subject>/<video>.avi
                                   <LPW_ROOT>/<subject>/<video>.txt
Each .txt has one line per frame: "x y" = ground-truth pupil centre (px).

Run from code/python/:
    python lpw_precision.py <LPW_ROOT> [frames_per_video] [pupil_px]

The pipeline is the one used for the synthetic sweeps (block down-sample ->
Gaussian noise -> hard threshold t = 0.5 -> ROI centroid, 400 realisations per frame
and noise level), anchored on the LPW ground-truth centre, with an absolute-accuracy
check against that ground truth.

PUPIL_PX sets the low-resolution pupil diameter and therefore the pitch, because the
pupil is 5 mm: 25 px is r = 0.20 mm/line, and 5 px is r = 1.00 mm/line, the reference
point of the synthetic sweeps. The deposited results use both.

The model curve follows the protocol of the real-image measurement: one fixed pupil
shape at one fixed subpixel phase, re-noised M times, the standard deviation taken over
those realisations, and the result averaged over shapes. Two model variants differ
only in the region analysed:

  roi   the centroid is taken within 1.0 pupil diameter of the noise-free centre, the
        same rule the real-image pipeline applies;
  full  the centroid is taken over the whole analysis window, as in run_converged.py.

Writes ../../results/lpw_precision_<pupil_px>px.json with units recorded.

Requires: numpy, scipy, matplotlib, opencv-python (cv2).
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import iris_model as IM
from lpw_common import (WIN, segment_pupil_local, block_downsample, centroid_roi,
                         read_annotation, sample_frame)

LPW_ROOT = sys.argv[1] if len(sys.argv) > 1 else 'LPW'
FRAMES_PER_VIDEO = int(sys.argv[2]) if len(sys.argv) > 2 else 3
PUPIL_LR = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0

rng = np.random.default_rng(0)
SN = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
M = 400                       # noise realisations per (frame, sn)
PUPIL_MM = IM.IRIS_D * 1e3    # 5 mm
UM_PER_PX = PUPIL_MM * 1e3 / PUPIL_LR      # low-res px -> um
R_MM = UM_PER_PX / 1e3                     # sensor pitch [mm/line]
DENS = 1.0 / R_MM                          # DensCCD for the model at this pitch
K_SHAPES = 24                 # population shapes for the model curve
RESULTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', '..', 'results'))
# Per-video checkpoints in .lpw_cache/ (not part of the repository), so an
# interrupted run resumes.
CACHE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', '..', '.lpw_cache'))


def ck_path():
    return os.path.join(CACHE, 'realval_%dpx_f%d.json' % (int(PUPIL_LR), FRAMES_PER_VIDEO))


def ck_load():
    try:
        with open(ck_path(), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {'videos': {}, 'model': None}


def ck_save(state):
    os.makedirs(CACHE, exist_ok=True)
    tmp = ck_path() + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f)
    os.replace(tmp, ck_path())          # atomic, so a reset cannot leave a half file


def model_curves():
    """sigma_within and bias of the model, with the LPW protocol: one fixed shape at
    one fixed subpixel phase, re-noised M times, averaged over K_SHAPES shapes.

    Returned in micrometres. 'roi' restricts the centroid to 1.0 pupil diameter around
    the noise-free centre, as the real-image pipeline does; 'full' uses the whole
    analysis window, as run_converged.py does."""
    srng = np.random.default_rng(20260805)
    acc = {k: np.zeros((K_SHAPES, len(SN))) for k in ('prec_roi', 'bias_roi',
                                                      'prec_full', 'bias_full')}
    for si in range(K_SHAPES):
        amp, ph = IM.population_distortion(srng)
        bnd, ycR, zcR = IM.build_pupil_boundary(np.array([0., 1., 2.]), 0.0, amp, ph)
        Yb, Zb = bnd[:, 0], bnd[:, 1]
        mX = IM.MARGP * (Yb.max() - Yb.min()); mY = IM.MARGP * (Zb.max() - Zb.min())
        # fixed subpixel phase, as a single real frame has
        sx0, sx1 = Yb.min() - 1.5 * mX, Yb.max() + 1.5 * mX
        sy0, sy1 = Zb.min() - 1.5 * mY, Zb.max() + 1.5 * mY
        block = IM.REDUCE
        nx = max(int(np.floor(1e3 * DENS * IM.RES_MULT * (sx1 - sx0))), block)
        ny = max(int(np.floor(1e3 * DENS * IM.RES_MULT * (sy1 - sy0))), block)
        PX, PY = np.meshgrid(np.linspace(sx0, sx1, nx), np.linspace(sy0, sy1, ny))
        ins = IM.points_in_shape_radial(PX.ravel(), PY.ravel(), Yb, Zb,
                                        ycR, zcR).reshape(PX.shape).astype(float)
        XT, YT, V0 = IM._block_average(PX, PY, ins, block)
        clean = (V0 >= IM.THRESH)
        if clean.sum() == 0:
            continue
        c0x, c0y = XT[clean].mean(), YT[clean].mean()
        half = IM.IRIS_D                          # 1.0 pupil diameter, as in centroid_roi
        near = (np.abs(XT - c0x) <= half) & (np.abs(YT - c0y) <= half)
        nrng = np.random.default_rng(70000 + si)
        for j, sn in enumerate(SN):
            cx = {'roi': [], 'full': []}
            cy = {'roi': [], 'full': []}
            for _ in range(M):
                Vb = (V0 + nrng.standard_normal(V0.shape) * sn) >= IM.THRESH
                for key, msk in (('roi', Vb & near), ('full', Vb)):
                    k = msk.sum()
                    if k:
                        cx[key].append(XT[msk].mean()); cy[key].append(YT[msk].mean())
            for key in ('roi', 'full'):
                if not cx[key]:
                    continue
                ax_, ay_ = np.array(cx[key]), np.array(cy[key])
                acc['prec_' + key][si, j] = np.sqrt((ax_.std(ddof=1) ** 2 +
                                                    ay_.std(ddof=1) ** 2) / 2) * 1e6
                acc['bias_' + key][si, j] = np.hypot(ax_.mean() - c0x,
                                                     ay_.mean() - c0y) * 1e6
    return {k: v.mean(axis=0) for k, v in acc.items()}


print('LPW comparison at a %.0f px pupil: r = %.3f mm/line, %.0f um per low-res px'
      % (PUPIL_LR, R_MM, UM_PER_PX))
mc = model_curves()
model_within_px = mc['prec_roi'] / UM_PER_PX
model_bias_px = mc['bias_roi'] / UM_PER_PX


# ----------------------------------------------------------------- main sweep
def process_video(avi, rng_):
    """All frames of one video. Returns a JSON-serialisable dict, the checkpoint unit."""
    txt = avi[:-4] + '.txt'
    if not os.path.exists(txt):
        return None
    ann = read_annotation(txt)
    valid = [k for k, (x, y) in enumerate(ann) if x > 1 and y > 1]
    if not valid:
        return None
    out = {'prec': [], 'bias': [], 'abs_err': [], 'gt_seg_err': [], 'montage_at': []}
    pick = np.linspace(0, len(valid) - 1, FRAMES_PER_VIDEO).round().astype(int)
    for k in np.unique(pick):
        fi = valid[k]
        g = sample_frame(avi, fi)
        if g is None:
            continue
        gx, gy = ann[fi]
        mask = segment_pupil_local(g, (gx, gy))
        if mask is None:
            continue
        ys, xs = np.nonzero(mask)
        out['gt_seg_err'].append(float(np.hypot(xs.mean() - gx, ys.mean() - gy)))
        D = 2 * np.sqrt(mask.sum() / np.pi)
        factor = max(int(round(D / PUPIL_LR)), 1)
        lr = block_downsample(mask, factor)
        base = centroid_roi(lr > 0.5, (lr.shape[1] / 2, lr.shape[0] / 2), 1e9)
        if base is None:
            continue
        half = 1.0 * 2 * np.sqrt((lr > 0.5).sum() / np.pi)
        gt_lr = np.array([gx / factor, gy / factor])
        prec, bias = [], []
        for sn in SN:
            cs = []
            for _ in range(M):
                c = centroid_roi((lr + rng_.normal(0, sn, lr.shape)) > 0.5, base, half)
                cs.append(c if c is not None else base)
            cs = np.array(cs)
            prec.append(float(np.sqrt((cs[:, 0].std() ** 2 + cs[:, 1].std() ** 2) / 2)))
            bias.append(float(np.hypot(cs[:, 0].mean() - base[0],
                                       cs[:, 1].mean() - base[1])))
        out['prec'].append(prec)
        out['bias'].append(bias)
        out['abs_err'].append(float(np.hypot(base[0] - gt_lr[0], base[1] - gt_lr[1])))
        out['montage_at'].append(int(fi))
    return out if out['prec'] else None


videos = sorted(glob.glob(os.path.join(LPW_ROOT, '*', '*.avi')))
if not videos:
    sys.exit("No .avi found under %r. Download LPW and pass its root path." % LPW_ROOT)

state = ck_load()
if state['videos']:
    print('resuming: %d of %d videos already done' % (len(state['videos']), len(videos)))
for n, avi in enumerate(videos, 1):
    if avi in state['videos']:
        continue
    r = process_video(avi, rng)
    state['videos'][avi] = r          # None is recorded too, so it is not retried
    ck_save(state)
    print('  [%2d/%2d] %s%s' % (n, len(videos), os.path.relpath(avi, LPW_ROOT),
                                '' if r else '  (no usable frame)'))

if state.get('model') is None:
    state['model'] = {k: v.tolist() for k, v in model_curves().items()}
    ck_save(state)
mc = {k: np.array(v) for k, v in state['model'].items()}
model_within_px = mc['prec_roi'] / UM_PER_PX
model_bias_px = mc['bias_roi'] / UM_PER_PX

results, abs_err, gt_seg_err, subjects = [], [], [], set()
for avi, r in state['videos'].items():
    if not r:
        continue
    for pr, bi in zip(r['prec'], r['bias']):
        results.append((np.array(pr), np.array(bi)))
    abs_err += r['abs_err']
    gt_seg_err += r['gt_seg_err']
    subjects.add(os.path.basename(os.path.dirname(avi)))

if not results:
    sys.exit("No frame yielded a valid pupil segmentation. Check the LPW path/format.")

# frame montage: re-read six frames, cheap next to the sweep and not worth caching
montage = []
for avi, r in sorted(state['videos'].items()):
    if not r or len(montage) >= 6:
        continue
    fi = r['montage_at'][0]
    g = sample_frame(avi, fi)
    ann = read_annotation(avi[:-4] + '.txt')
    if g is None:
        continue
    gx, gy = ann[fi]
    mask = segment_pupil_local(g, (gx, gy))
    if mask is None:
        continue
    x0, y0 = max(int(gx - WIN), 0), max(int(gy - WIN), 0)
    montage.append((g[y0:y0 + 2 * WIN, x0:x0 + 2 * WIN].copy(),
                    mask[y0:y0 + 2 * WIN, x0:x0 + 2 * WIN].copy()))

P = np.array([r[0] for r in results])
Bz = np.array([r[1] for r in results])
pmean, pstd = P.mean(0), P.std(0)
bmean, bstd = Bz.mean(0), Bz.std(0)
N_FRAMES = len(results)
K_SUBJ = len(subjects)


def knee(curve):
    base = curve[:2].mean()
    for i, sn in enumerate(SN):
        if curve[i] > 3 * max(base, 1e-3):
            return sn
    return None


r_prec = np.corrcoef(pmean, model_within_px)[0, 1]
r_bias = np.corrcoef(bmean, model_bias_px)[0, 1]
kn = knee(pmean)
abs_low = float(np.median(abs_err))          # low-noise absolute GT error [px]
seg_err = float(np.median(gt_seg_err))

print('\n================ LPW VALIDATION SUMMARY ================')
print(' sn   real_prec[px]   model s_within[px]   real_bias[px]   model_bias[px]')
for i, sn in enumerate(SN):
    print('%.2f   %5.2f+-%4.2f     %5.2f               %5.2f          %5.2f' %
          (sn, pmean[i], pstd[i], model_within_px[i], bmean[i], model_bias_px[i]))
print('\n----- LPW validation summary -----')
print('[LPW: N frames] = %d' % N_FRAMES)
print('[LPW: K subjects] = %d' % K_SUBJ)
print('[LPW: GT px]   = %.1f   (hi-res seg-centroid vs annotation)' % seg_err)
print('[LPW: r_prec]  = %.3f' % r_prec)
print('[LPW: r_bias]  = %.3f' % r_bias)
print('[LPW: knee]    = %s' % ('%.2f' % kn if kn else '0.15-0.20'))
print('[LPW: abs px]  = %.1f  (low-res, low-noise abs error vs GT)' % abs_low)
print('---------------------------------------------------------------')

rel = (mc['prec_roi'] - pmean * UM_PER_PX) / (pmean * UM_PER_PX)
rms_rel = float(np.sqrt(np.mean(rel ** 2)))
print('[LPW: RMS rel dev] = %.1f%%  (model roi against real, absolute um)' % (100 * rms_rel))
print('[LPW: model full-window / roi at sn=0.50] = %.1f'
      % (mc['prec_full'][-1] / mc['prec_roi'][-1]))

out = dict(pupil_px=PUPIL_LR, r_mm_per_line=R_MM, um_per_px=UM_PER_PX,
           units='all *_um arrays are micrometres; *_px arrays are low-resolution pixels',
           sigma_n=SN.tolist(), n_frames=int(N_FRAMES), n_subjects=int(K_SUBJ),
           real_prec_um=(pmean * UM_PER_PX).tolist(),
           real_bias_um=(bmean * UM_PER_PX).tolist(),
           real_prec_sd_um=(pstd * UM_PER_PX).tolist(),
           model_prec_roi_um=mc['prec_roi'].tolist(),
           model_bias_roi_um=mc['bias_roi'].tolist(),
           model_prec_full_um=mc['prec_full'].tolist(),
           model_bias_full_um=mc['bias_full'].tolist(),
           pearson_prec=float(r_prec), pearson_bias=float(r_bias),
           rms_rel_dev=rms_rel, gt_seg_px=seg_err, abs_low_px=abs_low)
dest = os.path.join(RESULTS, 'lpw_precision_%dpx.json' % int(PUPIL_LR))
with open(dest, 'w', encoding='utf-8') as fh:
    json.dump(out, fh, indent=1)
print('saved', dest)

# ----- diagnostic curves -----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.0))
ax1.fill_between(SN, (pmean - pstd) / pmean[-1], (pmean + pstd) / pmean[-1],
                 color='tab:blue', alpha=0.18, label='LPW frames (mean ± SD, n=%d)' % N_FRAMES)
ax1.plot(SN, pmean / pmean[-1], 'o-', color='tab:blue', lw=2, ms=5, label='real-image precision')
ax1.plot(SN, model_within_px / model_within_px[-1], 's--', color='k', lw=2, ms=5,
         label='model σ_within (r=0.20 mm/line)')
ax1.axvspan(0.15, 0.20, color='gray', alpha=0.15, label='transition knee')
ax1.set_xlabel('relative noise σn'); ax1.set_ylabel('normalised centroid error (=1 at σn=0.5)')
ax1.set_title('(a) Noise-response shape (Pearson r = %.3f)' % r_prec)
ax1.legend(fontsize=7.2); ax1.grid(alpha=0.3)

ax2.fill_between(SN, bmean - bstd, bmean + bstd, color='tab:purple', alpha=0.15)
ax2.plot(SN, bmean, '^-', color='tab:purple', lw=2, ms=5, label='LPW threshold bias')
ax2.plot(SN, model_bias_px, 'v--', color='tab:red', lw=2, ms=5, label='model threshold bias')
ax2.axvspan(0.15, 0.20, color='gray', alpha=0.15, label='transition knee')
ax2.set_xlabel('relative noise σn'); ax2.set_ylabel('systematic bias [sensor px]')
ax2.set_title('(b) Threshold bias on real pupils (r = %.3f)' % r_bias)
ax2.legend(fontsize=7.5); ax2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(CACHE, 'lpw_curves_%dpx.png' % int(PUPIL_LR)), dpi=300)
print('saved lpw_curves_%dpx.png' % int(PUPIL_LR))

# ----- frame montage (example frames + masks) -----
# Written to the local cache, never to results/: it contains real LPW frames, and this
# repository does not redistribute the dataset or any imagery derived from it. Any
# published use must carry the CC BY-NC-SA 4.0 credit line of the dataset licence.
if montage:
    ncol = len(montage)
    fig2, axes = plt.subplots(2, ncol, figsize=(1.8 * ncol, 3.6))
    if ncol == 1:
        axes = axes.reshape(2, 1)
    for j, (crop, msk) in enumerate(montage):
        axes[0, j].imshow(crop, cmap='gray'); axes[0, j].axis('off')
        ov = np.dstack([crop] * 3)
        ov[msk] = [1, 0, 0]
        axes[1, j].imshow(ov); axes[1, j].axis('off')
    fig2.suptitle('LPW eye frames (top) and extracted pupil masks (bottom)', fontsize=9)
    fig2.tight_layout(); fig2.savefig(os.path.join(CACHE, 'lpw_frames_%dpx.png' % int(PUPIL_LR)), dpi=300)
    print('saved lpw_frames_%dpx.png' % int(PUPIL_LR))
