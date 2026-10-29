# -*- coding: utf-8 -*-
"""Uncertainty budget of a single pupil-centre coordinate.

Reads the outputs of run_converged.py (sigma_within, sigma_between), gci.py (u_num),
alternatives.py (the residual of the bias correction, u_b) and run_umodel.py (the
model-incompleteness runs) and combines them in quadrature,

    u_c = sqrt(sigma_within^2 + u_b^2 + u_num^2 + u_model^2).

Each model-incompleteness component is the variance difference between the run with
the mechanism switched on and its paired baseline, (|u'^2 - u^2|)^(1/2), so that adding
it in quadrature to sigma_within reproduces the dispersion with the mechanism included.

  shot noise   measured at every noise level of the budget;
  optics       measured at sigma_eff = 0.05, 0.10, 0.20 and 0.50; the larger of the two
               PSF widths is taken, and between the evaluated levels the relative
               variance excess |u'^2 - u^2| / u^2 is interpolated linearly in sigma_eff;
  diameter     the relative effect found at sigma_eff = 0.10 (drawn with s.d. 0.5 mm),
               applied at every level;
  occlusion    reported in its own column: a chord removing a fraction q of the pupil
               shifts the centroid by slope*q; an instrument rejecting q > 0.05 is left
               with a rectangular distribution on [-slope*0.05, +slope*0.05] per
               coordinate, u_occl = slope*0.05/sqrt(3), and slope*0.30/sqrt(3) without
               occlusion detection.

    python budget.py             # seconds; writes results/budget.json
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))

SN = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
OPT_LEVELS = [0.05, 0.10, 0.20, 0.50]


def load(name):
    with open(os.path.join(RES, name), encoding='utf-8') as f:
        return json.load(f)


def vdiff(a, b):
    return math.sqrt(abs(a * a - b * b))


def main():
    conv = load('converged_os5.json')
    gci = load('gci.json')
    alt = load('alternatives.json')
    um = load('umodel.json')
    base = um['baseline']
    key = lambda s: 'sn%.2f' % s

    # residual of the bias correction over the noise, pitch and gaze sweeps
    resid = [v['corrected']['bias'] for grp in ('noise', 'pitch', 'gaze')
             for v in alt.get(grp, {}).values() if v.get('corrected')]
    u_b = math.sqrt(sum(x * x for x in resid) / len(resid))

    # optics: relative variance excess of the wider-effect PSF at the evaluated levels
    opt_rel = []
    for s in OPT_LEVELS:
        b = base[key(s)]['s']
        opt_rel.append(max(abs(um['optics']['fwhm%.1f_%s' % (f, key(s))]['s'] ** 2 - b * b)
                           for f in (0.5, 1.0)) / b ** 2)
    d0 = base['sn0.10']['s']
    diam_rel = um['diameter']['drawn_sd0.50']['s'] ** 2 / d0 ** 2 - 1.0

    qs = np.array([0.10, 0.20, 0.30])
    shift = np.array([um['occlusion']['q%.2f_sn0.10' % q]['bias'] for q in qs])
    slope = float(np.polyfit(qs, shift, 1)[0])
    u_occl = slope * 0.05 / math.sqrt(3)
    u_occl_none = slope * 0.30 / math.sqrt(3)

    rows = []
    for s in SN:
        b = base[key(s)]['s']
        t4 = conv['budget/%s' % key(s)]
        shot = vdiff(um['shot'][key(s)]['s'], b)
        optics = b * math.sqrt(float(np.interp(s, OPT_LEVELS, opt_rel)))
        diam = b * math.sqrt(diam_rel)
        u_model = math.sqrt(shot ** 2 + optics ** 2 + diam ** 2)
        u_num = gci['points']['budget/%s' % key(s)]['g_u_num_um']
        u_c = math.sqrt(t4['sigma_within'] ** 2 + u_b ** 2 + u_num ** 2 + u_model ** 2)
        rows.append(dict(sigma_eff=s, sigma_within=t4['sigma_within'], u_b=u_b,
                         u_num=u_num, u_shot=shot, u_optics=optics, u_diam=diam,
                         u_model=u_model, u_c=u_c, U95=2 * u_c,
                         u_c_occl=math.hypot(u_c, u_occl),
                         sigma_between=t4['sigma_between']))

    ref = rows[SN.index(0.10)]
    out = {
        'convention': '(|u_on^2 - u_off^2|)^(1/2), paired seed streams',
        'u_b': u_b, 'u_b_max': max(resid),
        'u_occl_detected': u_occl, 'u_occl_undetected': u_occl_none,
        'occlusion_shift_um': dict(zip(['0.10', '0.20', '0.30'], shift.round(2).tolist())),
        'occlusion_shift_at_5pct_um': slope * 0.05,
        'model_components_at_sigma_0.10': {'shot': ref['u_shot'], 'optics': ref['u_optics'],
                                 'diameter': ref['u_diam'], 'u_model': ref['u_model']},
        'budget': rows,
    }
    with open(os.path.join(RES, 'budget.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)

    print('u_b = %.1f um (largest residual %.1f um); u_occl = %.0f um (%.0f without '
          'detection)' % (u_b, max(resid), u_occl, u_occl_none))
    print('model components at sigma_eff = 0.10: shot %.0f, optics %.0f, diameter %.0f, '
          'u_model %.0f um' % (ref['u_shot'], ref['u_optics'], ref['u_diam'],
                               ref['u_model']))
    print('sigma_eff  s_within  u_b  u_num  u_model   u_c   U95  u_c+occl  s_between')
    for r in rows:
        print('  %.2f     %5.0f   %4.1f  %4.1f   %5.0f   %5.0f %5.0f  %6.0f     %.1f'
              % (r['sigma_eff'], r['sigma_within'], r['u_b'], r['u_num'], r['u_model'],
                 r['u_c'], r['U95'], r['u_c_occl'], r['sigma_between']))


if __name__ == '__main__':
    main()
