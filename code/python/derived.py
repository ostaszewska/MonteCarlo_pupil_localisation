# -*- coding: utf-8 -*-
"""Quantities derived from more than one stored result (ratios, ranges, directions).

Most results are read directly from one field of one file in results/. The ones
computed here combine several fields or files: relative changes, spans and ranges,
circular means of the bias direction, the threshold optimum, coverage-factor and
Cramer-Rao ranges. Values are written unrounded.

Run after the analysis scripts:
    python derived.py               # seconds; writes results/derived.json
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))


def load(name):
    with open(os.path.join(RES, name), encoding='utf-8') as f:
        return json.load(f)


conv = load('converged_os5.json')
gci = load('gci.json')
umod = load('umodel.json')
alt = load('alternatives.json')
crlb = load('crlb.json')
sp = load('stats_physics.json')
thr = load('threshold.json')
jcg = load('jcgm101.json')
figd = load('bias_vectors.json')

ANG = list(range(-60, 61, 15))
RR = [0.500, 0.667, 1.000, 1.429, 2.000, 2.500, 3.333]
REP = conv['meta']['rep']
KNEE = 0.20


def noise(sn, a=0):
    return conv['noise/sn%.2f_a%+03d' % (sn, a)]


def pitch(r, a=0):
    return conv['pitch/r%.3f_a%+03d' % (r, a)]


def budget(sn):
    return conv['budget/sn%.2f' % sn]


def circmean(deg):
    a = np.radians(deg)
    return math.degrees(math.atan2(float(np.mean(np.sin(a))), float(np.mean(np.cos(a)))))


def circdev(deg, m):
    return np.degrees(np.angle(np.exp(1j * (np.radians(deg) - math.radians(m)))))


def pct(a, b):
    return 100.0 * (a - b) / b


out = {}

# ------------------------------------------------ design space
ecc = lambda f, *k: max(f(*k, -60)['s'], f(*k, 60)['s'])
out['design_space'] = {
    'u_span_noise_0.50_over_0.05': noise(0.50)['s'] / noise(0.05)['s'],
    'u_span_pitch_2.5_over_0.5': pitch(2.500)['s'] / pitch(0.500)['s'],
    'u_range_um': [pitch(0.500)['s'], noise(0.50)['s']],
    'u_range_pct_of_pupil': [100 * pitch(0.500)['s'] / 5000, 100 * noise(0.50)['s'] / 5000],
    'gaze_rise_pct_at_sigma_0.05': pct(ecc(noise, 0.05), noise(0.05)['s']),
    'gaze_rise_pct_at_sigma_0.10': pct(ecc(noise, 0.10), noise(0.10)['s']),
    'gaze_rise_pct_over_pitch': [min(pct(ecc(pitch, r), pitch(r)['s']) for r in RR[:6]),
                                 max(pct(ecc(pitch, r), pitch(r)['s']) for r in RR[:6])],
    'discard_pct_r3.333_axis': 100 * pitch(3.333)['discard'],
    'availability_r3.333_alpha60': sorted(1 - pitch(3.333, a)['discard'] for a in (-60, 60)),
    'discard_pct_r2.5_max': 100 * max(pitch(2.500, a)['discard'] for a in ANG),
    'shared_point_u_um': [noise(0.10)['s'], pitch(1.000)['s'], budget(0.10)['s']],
}
sh = out['design_space']['shared_point_u_um']
out['design_space']['shared_point_spread_pct'] = 100 * (max(sh) - min(sh)) / min(sh)
out['design_space']['sigma_between_pct_of_within_ref'] = \
    100 * budget(0.10)['sigma_between'] / budget(0.10)['sigma_within']
if 'fine/r0.250' in conv:
    t9 = conv['fine/r0.250']
    out['design_space']['sigma_between_pct_of_within_r0.25'] = \
        100 * t9['sigma_between'] / t9['sigma_within']

# ------------------------------------------------ grid convergence
out['grid'] = {'base_grid_max_departure_pct':
               max(abs(100 * (p['f1'] - p['f3']) / p['f1']) for p in gci['points'].values())}

# ------------------------------------------------ error distribution
ry = sp['rayleigh']
out['error_distribution'] = {'u_y_over_u_x': ry['sy_um'] / ry['sx_um']}
dg = sp['lever_arm']['deg_per_s']
out['error_distribution']['deg_sigma_0.05_averaged_60_frames'] = dg['sn005_deg'] / math.sqrt(60)

# ------------------------------------------------ precision on real (LPW) geometry
lpw = {}
for px in (5, 25):
    d = load('lpw_precision_%dpx.json' % px)
    real, model = np.array(d['real_prec_um']), np.array(d['model_prec_roi_um'])
    rel = model / real - 1
    low = np.array(d['sigma_n']) <= 0.15 + 1e-9
    lpw['%dpx' % px] = {'rms_rel_dev_pct_all': 100 * float(np.sqrt(np.mean(rel ** 2))),
                        'rms_rel_dev_pct_sigma_le_0.15': 100 * float(np.sqrt(np.mean(rel[low] ** 2)))}
out['lpw_precision'] = lpw

# ------------------------------------------------ model incompleteness
base = umod['baseline']
rel_shot = [pct(umod['shot']['sn%.2f' % s]['s'], base['sn%.2f' % s]['s'])
            for s in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)]
opt = {f: [pct(umod['optics']['fwhm%.1f_sn%.2f' % (f, s)]['s'], base['sn%.2f' % s]['s'])
           for s in (0.05, 0.10, 0.20, 0.50)] for f in (0.5, 1.0)}
d0 = base['sn0.10']['s']
out['model_incompleteness'] = {
    'shot_change_pct_at_sigma_0.10': rel_shot[1],
    'shot_change_pct_max_abs': max(abs(x) for x in rel_shot),
    'optics_change_pct_fwhm0.5_max_abs': max(abs(x) for x in opt[0.5]),
    'optics_change_pct_fwhm1.0': [min(opt[1.0]), max(opt[1.0])],
    'contrast_0.3_factor': umod['contrast']['C0.3_sn0.10']['s'] / d0,
    'diameter_drawn_change_pct': pct(umod['diameter']['drawn_sd0.50']['s'], d0),
    'photopic_3.09mm_factor': umod['diameter']['fixed_D3.09']['s'] / d0,
}

# ------------------------------------------------ bias direction
bv = figd['biasvec']
big = [b for b in bv if b['mag'] > 5.0 * noise(b['sigma_n'], int(b['alpha']))['s'] / math.sqrt(1800)]
above = [b['phi'] for b in big if b['sigma_n'] >= KNEE]
below = [b['phi'] for b in big if b['sigma_n'] < KNEE]
m_hi = circmean(above)
out['bias_direction'] = {
    'knee_sigma': KNEE,
    'above_knee_deg': m_hi,
    'above_knee_rms_deg': float(np.sqrt(np.mean(circdev(above, m_hi) ** 2))),
    'above_knee_points': len(above),
    'below_knee_deg': circmean(below) if below else None,
    'below_knee_points': len(below),
}
pm = [(float(k[2:]), v['measured_phi'], v['predicted_phi'])
      for k, v in sorted(alt['noise'].items())
      if v.get('measured_phi') is not None and v.get('predicted_phi') is not None]
out['bias_direction'].update({
    'ladder_measured_below_deg': circmean([m for s, m, _ in pm if s < KNEE]),
    'ladder_measured_above_deg': circmean([m for s, m, _ in pm if s >= KNEE]),
    'ladder_predicted_below_deg': circmean([p for s, _, p in pm if s < KNEE]),
    'ladder_predicted_above_deg': circmean([p for s, _, p in pm if s >= KNEE]),
    'ladder_max_departure_deg': max(abs(float(circdev([m], p)[0])) for _, m, p in pm),
})

# ------------------------------------------------ detection threshold
row = thr['pop']['0.10']
best = min(row, key=lambda t: row[t]['u'])
nthr = thr['meta']['rep']


def thr_phi(levels):
    a = [thr['circ'][s][t]['phi'] for s in levels for t in thr['circ'][s]
         if thr['circ'][s][t]['bias'] > 5.0 * thr['circ'][s][t]['u'] / math.sqrt(nthr)]
    m = circmean(a)
    return m, float(np.max(np.abs(circdev(a, m))))


out['threshold'] = {
    'best_t_at_sigma_0.10': float(best),
    'u_at_best_um': row[best]['u'],
    'u_at_best_below_t0.50_pct': 100 * (1 - row[best]['u'] / row['0.50']['u']),
    'worst_cost_pct': pct(max(row[t]['u'] for t in row), row[best]['u']),
    'direction_sigma_0.30_0.50_deg': thr_phi(['0.30', '0.50'])[0],
    'direction_sigma_0.30_0.50_max_dev_deg': thr_phi(['0.30', '0.50'])[1],
    'direction_sigma_0.10_deg': thr_phi(['0.10'])[0],
}

# ------------------------------------------------ coverage intervals
keff = [v['k_effective'] for v in jcg.values() if isinstance(v, dict) and 'k_effective' in v]
out['coverage'] = {'k_effective_range': [min(keff), max(keff)]}

# ------------------------------------------------ estimators and Cramer-Rao bound
v = alt['variable']
c = alt['cost']
n10 = alt['noise']['sn0.10']
out['estimators'] = {
    'weighted_gated_gain_pct_at_sigma_0.10':
        100 * (1 - n10['weighted_t']['s'] / n10['centroid']['s']),
    'variable_noise_correction_gain_pct':
        100 * (1 - v['corrected']['rmse'] / v['centroid']['rmse']),
    'variable_noise_ellipse_gain_over_corrected_pct':
        100 * (1 - v['ellipse']['rmse'] / v['corrected']['rmse']),
    'variable_noise_frac_above_0.20_pct': 100 * v['frac_above_020'],
    'cost_centroid_pct_of_500us': 100 * c['centroid_us_per_frame'] / 500.0,
    'cost_ellipse_pct_of_500us': 100 * c['ellipse_us_per_frame'] / 500.0,
}
OSF = crlb['meta'].get('oversample', 5)
rp = [r for r in crlb['pitch'] if r.get('ratio') and r.get('oversample', OSF) == OSF]
rn = [r['ratio'] for r in crlb['noise'] if r.get('ratio')]
out['crlb'] = {
    'pitch_range_mm': [min(r['r_mm'] for r in rp), max(r['r_mm'] for r in rp)],
    'ratio_over_pitch': [min(r['ratio'] for r in rp), max(r['ratio'] for r in rp)],
    'ratio_over_noise': [min(rn), max(rn)],
}

with open(os.path.join(RES, 'derived.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
for sec, d in out.items():
    print(sec)
    for k, x in d.items():
        print('   %-48s %s' % (k, x))
print('saved', os.path.join(RES, 'derived.json'))
