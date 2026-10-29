# -*- coding: utf-8 -*-
"""Fine-pitch law and the sensor requirements for ocular-microtremor detection.

The single-subject precision sigma_within(r) at sigma_eff = 0.10, alpha = 0 is taken from
the grid-converged sweep (converged_os5.json, fine) for r >= 0.25 mm/line and from
the base-grid sweep (fine_pitch_base.json, fine_pitch_sweep.py) below that, where
refining the grid changes s by less than the statistical noise.
A power law sigma_within = A r^beta is fitted to it by least squares in log-log over
r = 0.10-0.50 mm/line.

The requirement is propagated with the lever arm L = 10.45 +- 0.5 mm drawn per trial:
a tremor of angular amplitude a displaces the pupil centre by d = a L; instantaneous
(waveform) detection needs sigma_within < d / k_det, narrowband (spectral) detection in
a band B at frame rate f_s needs sigma_within < (d / sqrt 2) / (k_det sqrt(2 B / f_s)),
and inverting the power law gives the required pitch and its 95% coverage interval.

    python tremor.py                   # seconds; writes results/tremor.json
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))

FINE = [0.500, 0.333, 0.250, 0.200, 0.167, 0.125, 0.100]
L0, UL = 10.45e-3, 0.50e-3            # m, centre of rotation to entrance pupil
K_DET, FS, B = 3.0, 2000.0, 1.0
AMPS = {'6 arcsec (peak)': 6.0, '17.5 arcsec (median)': 17.5, '60 arcsec (max)': 60.0}


def fine_sweep():
    """sigma_within(r) at sigma_eff = 0.10, alpha = 0, in um; the grid each value is on."""
    conv = json.load(open(os.path.join(RES, 'converged_os5.json')))
    base = json.load(open(os.path.join(RES, 'fine_pitch_base.json')))
    out = {}
    for r in FINE:
        k = 'fine/r%.3f' % r
        if k in conv:
            out[r] = (conv[k]['sigma_within'], 'converged (oversample 5)')
        else:
            row = min(base, key=lambda z: abs(z['r_mm'] - r))
            out[r] = (row['s_um'], 'base grid (oversample 1)')
    return out


def fine_pitch_law():
    """(A, beta) of sigma_within = A r^beta over the fine sweep."""
    sw = fine_sweep()
    rs = np.array(sorted(sw))
    b, a = np.polyfit(np.log(rs), np.log([sw[r][0] for r in rs]), 1)
    return float(math.exp(a)), float(b)


def targets(A, beta, nmc=200000, seed=4242):
    rng = np.random.default_rng(seed)
    L = L0 + UL * rng.standard_normal(nmc)
    L = L[(L > 8.5e-3) & (L < 12.5e-3)]
    arc = math.pi / (180.0 * 3600.0)
    out = {}
    for name, a in AMPS.items():
        d_um = a * arc * L * 1e6
        s_wave = d_um / K_DET
        s_spec = (d_um / math.sqrt(2.0)) / (K_DET * math.sqrt(2 * B / FS))
        r_spec = (s_spec / A) ** (1.0 / beta)
        out[name] = dict(
            A_um=[float(np.mean(d_um)), float(np.std(d_um))],
            s_req_waveform_um=float(np.mean(s_wave)),
            s_req_spectral_um=float(np.mean(s_spec)),
            r_spectral_mm=float(np.mean(r_spec)),
            r_spectral_ci95=[float(np.percentile(r_spec, 2.5)),
                             float(np.percentile(r_spec, 97.5))],
            pupil_px=float(np.mean(5.0 / r_spec)))
    return out


def main():
    sw = fine_sweep()
    A_fit, b_fit = fine_pitch_law()
    A, beta = A_fit, b_fit
    tg = targets(A, beta)

    w = sw[0.100][0]
    band = w * math.sqrt(2 * 12.0 / FS)                  # 78-90 Hz band, B = 12 Hz
    doc = dict(
        fine_sweep={'%.3f' % r: dict(sigma_within_um=v, grid=g) for r, (v, g) in sw.items()},
        law=dict(A=A, beta=beta),
        lever_arm_mm=[L0 * 1e3, UL * 1e3], k_det=K_DET, fs_hz=FS, band_hz=B,
        targets=tg,
        waveform_gap=w / tg['60 arcsec (max)']['s_req_waveform_um'],
        band_limited=dict(noise_um=band, rms_um=K_DET * band,
                          peak_arcsec=math.degrees(K_DET * band * math.sqrt(2) * 1e-6 / L0)
                          * 3600.0),
        band_exponent=0.5 / beta, band_halved=0.5 ** (0.5 / beta),
        band_doubled=2.0 ** (0.5 / beta))
    with open(os.path.join(RES, 'tremor.json'), 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1)

    print('fine sweep sigma_within [um]:',
          ', '.join('%.3f: %.1f' % (r, sw[r][0]) for r in FINE))
    print('fitted law: sigma_within = %.1f r^%.3f' % (A, beta))
    for k, v in tg.items():
        print('%-22s A = %.2f um  waveform %.2f um  spectral %.1f um  r = %.3f (%.3f-%.3f)'
              ' mm/line  ~%.0f px' % (k, v['A_um'][0], v['s_req_waveform_um'],
                                      v['s_req_spectral_um'], v['r_spectral_mm'],
                                      *v['r_spectral_ci95'], v['pupil_px']))
    print('waveform gap x%.1f; band-limited noise %.2f um -> %.2f um RMS (%.0f arcsec)'
          % (doc['waveform_gap'], band, K_DET * band, doc['band_limited']['peak_arcsec']))
    print('saved', os.path.join(RES, 'tremor.json'))


if __name__ == '__main__':
    main()
