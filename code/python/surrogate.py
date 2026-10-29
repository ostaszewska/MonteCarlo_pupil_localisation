# -*- coding: utf-8 -*-
"""Closed-form surrogate of the on-axis uncertainty.

    u(sigma_eff, r) ~ u0 F_sigma(sigma_eff) F_r(r),   u0 = 102 um

    F_sigma(x) = [1 + a / (1 + exp(-(x - x0) / w))] / norm,   F_sigma(0.10) = 1
    F_r(r)     = (r / 1 mm/line) ** beta,                     F_r(1.00)     = 1

Both factors are fitted to the grid-converged sweeps in converged_os5.json at alpha = 0:
F_sigma to the noise ladder (r = 1.00 mm/line, ten noise levels) by least
squares in micrometres, F_r to the pitch ladder (sigma_eff = 0.10, r = 0.50 to 2.50
mm/line) by least squares in ln u through the reference point. norm is not a free
parameter: it is the value of the bracket at 0.10.

Also reported: the coefficient of determination of each factor, the RMS and largest
local deviation over both cross-sections, the finite-difference sensitivity
coefficients over the low-noise box (sigma_eff 0.05-0.15, r 0.50-2.00 mm/line), and the
analytic derivative of the surrogate at the reference point.

    python surrogate.py          # seconds; writes results/surrogate.json
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))

U0 = 102.0              # um, at the reference point sigma_eff = 0.10, r = 1.00 mm/line
SN = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
RR = np.array([0.500, 0.667, 1.000, 1.429, 2.000, 2.500])


def bracket(x, a, x0, w):
    return 1.0 + a / (1.0 + np.exp(-(x - x0) / w))


def r2(y, f):
    return float(1.0 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    conv = json.load(open(os.path.join(RES, 'converged_os5.json')))
    us = np.array([conv['noise/sn%.2f_a+00' % x]['s'] for x in SN])
    ur = np.array([conv['pitch/r%.3f_a+00' % x]['s'] for x in RR])

    p, _ = curve_fit(lambda x, a, x0, w: U0 * bracket(x, a, x0, w) / bracket(0.10, a, x0, w),
                     SN, us, p0=[6.0, 0.25, 0.05])
    a, x0, w = map(float, p)
    norm = float(bracket(0.10, a, x0, w))
    f_sig = lambda x: bracket(x, a, x0, w) / norm

    lr = np.log(RR)
    beta = float(np.sum(lr * np.log(ur / U0)) / np.sum(lr ** 2))

    pred_s = U0 * f_sig(SN)
    pred_r = U0 * RR ** beta
    dev = 100.0 * np.r_[(pred_s - us) / us, (pred_r - ur) / ur]

    lo = [conv['noise/sn%.2f_a+00' % x]['s'] for x in (0.05, 0.10, 0.15)]
    rl = [conv['pitch/r%.3f_a+00' % x]['s'] for x in (0.500, 2.000)]
    h = 1e-6
    out = {
        's0_um': U0,
        'G': {'a': a, 'x0': x0, 'w': w, 'norm': norm, 'r2': r2(us, pred_s)},
        'R': {'exponent': beta, 'r2': r2(ur, pred_r)},
        'rms_pct': float(np.sqrt(np.mean(dev ** 2))),
        'max_pct': float(np.max(np.abs(dev))),
        'max_at_sigma': float(SN[np.argmax(np.abs(dev[:len(SN)]))]),
        'c_sigma': float((lo[2] - lo[0]) / 0.10),
        'c_r': float((rl[1] - rl[0]) / 1.5),
        'd_sigma_ref': float(U0 * (f_sig(0.10 + h) - f_sig(0.10 - h)) / (2 * h)),
        'd_r_ref': float(U0 * beta),
        'deviation_pct': {'noise': dev[:len(SN)].round(2).tolist(),
                          'pitch': dev[len(SN):].round(2).tolist()},
    }
    with open(os.path.join(RES, 'surrogate.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('F_sigma = [1 + %.2f / (1 + exp(-(x - %.3f) / %.4f))] / %.3f   R2 = %.3f'
          % (a, x0, w, norm, out['G']['r2']))
    print('F_r = r^%.2f   R2 = %.3f' % (beta, out['R']['r2']))
    print('RMS deviation %.1f%%, largest %.0f%% at sigma_eff = %.2f'
          % (out['rms_pct'], out['max_pct'], out['max_at_sigma']))
    print('c_sigma = %.0f um, c_r = %.0f um per mm/line, dF/dsigma at 0.10 = %.0f um'
          % (out['c_sigma'], out['c_r'], out['d_sigma_ref']))


if __name__ == '__main__':
    main()
