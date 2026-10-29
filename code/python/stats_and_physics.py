# -*- coding: utf-8 -*-
"""Statistics and physical conversions that need no new sweep.

  - Rayleigh goodness-of-fit test of the radial error, as a parametric bootstrap;
  - variance shares of the three input quantities for three instrument classes, and
    logarithmic sensitivities;
  - lever arm from the centre of rotation to the entrance pupil, and the conversion
    of u to gaze angle;
  - imaging geometry r = p / m, with numerical examples.

Writes results/stats_physics.json.
"""
import json
import os

import numpy as np
from multiprocessing import Pool, cpu_count

import iris_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))
OUT = os.path.join(RES, 'stats_physics.json')
VERS = np.array([0.0, 1.0, 2.0])
OS = 5


# ============================================================== Rayleigh test
def _radial_trial(task):
    seed, dens, sn, angle = task
    rng = np.random.default_rng(int(seed))
    amp, ph = M.population_distortion(rng)
    ycR, zcR, _, _, ex, ey = M.single_trial(VERS, angle, amp, ph, dens, sn, rng,
                                            oversample=OS)
    if np.isnan(ex):
        return (np.nan, np.nan)
    return (ex - ycR, ey - zcR)


def ks_D(x, sig):
    """Kolmogorov-Smirnov statistic of x against Rayleigh(sig)."""
    x = np.sort(x)
    n = x.size
    F = 1.0 - np.exp(-x ** 2 / (2.0 * sig ** 2))
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))


def rayleigh_bootstrap(pool, n=3600, sn=0.50, dens=1.0, nboot=10000, seed=777):
    ss = np.random.SeedSequence(seed).spawn(n)
    v = np.array(pool.map(_radial_trial,
                          [(s.generate_state(1)[0], dens, sn, 0.0) for s in ss],
                          chunksize=32))
    v = v[~np.isnan(v).any(axis=1)]
    rad = np.hypot(v[:, 0], v[:, 1])
    N = rad.size
    sig = float(np.sqrt(np.mean(rad ** 2) / 2.0))          # maximum likelihood
    D = ks_D(rad, sig)

    rng = np.random.default_rng(20260819)
    Db = np.empty(nboot)
    sb = np.empty(nboot)
    for b in range(nboot):
        r = sig * np.sqrt(-2.0 * np.log(rng.random(N)))    # Rayleigh(sig) sample
        s_hat = np.sqrt(np.mean(r ** 2) / 2.0)             # refit on each sample
        sb[b] = s_hat
        Db[b] = ks_D(r, s_hat)
    p_boot = float((Db >= D).mean())
    return dict(n=int(N), sigma_mm=sig * 1e3,
                sigma_u_mm=float(np.std(sb, ddof=1)) * 1e3,
                D=D, D_crit_95=float(np.percentile(Db, 95)),
                D_crit_99=float(np.percentile(Db, 99)),
                p_bootstrap=p_boot, nboot=nboot,
                sx_um=float(v[:, 0].std(ddof=1) * 1e6),
                sy_um=float(v[:, 1].std(ddof=1) * 1e6),
                rho=float(np.corrcoef(v[:, 0], v[:, 1])[0, 1]))


# ============================================================== variance shares
def decompose(S):
    """Two-factor decomposition of the variance of a table S[factor, angle].

    Main effect of the swept factor = variance of the row means; main effect of the
    gaze angle = variance of the column means; the remainder is interaction. This is
    performed on the simulated grid itself.
    """
    S = np.asarray(S, float)
    g = S.mean()
    v_tot = ((S - g) ** 2).mean()
    v_f = ((S.mean(axis=1) - g) ** 2).mean()
    v_a = ((S.mean(axis=0) - g) ** 2).mean()
    v_i = max(v_tot - v_f - v_a, 0.0)
    return [100 * v_f / v_tot, 100 * v_a / v_tot, 100 * v_i / v_tot]


def instrument_classes(conv):
    ANG = np.array(range(-60, 61, 15), float)
    SN = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    RR = [0.500, 0.667, 1.000, 1.429, 2.000, 2.500]
    noise = np.array([[conv['noise/sn%.2f_a%+03d' % (s, a)]['s'] for a in ANG.astype(int)]
                   for s in SN])
    pitch = np.array([[conv['pitch/r%.3f_a%+03d' % (r, a)]['s'] for a in ANG.astype(int)]
                   for r in RR])

    classes = [('screen or smartphone camera', 20, 0.15, 1.429),
               ('head-mounted VOG', 40, 0.15, 1.429),
               ('wide-range remote tracker', 60, 0.15, 1.429)]
    out = {'classes': [], 'full_range': {}}
    out['full_range']['noise'] = decompose(noise)
    out['full_range']['pitch'] = decompose(pitch)
    for name, amax, snmax, rmax in classes:
        ja = np.abs(ANG) <= amax
        isn = [i for i, s in enumerate(SN) if s <= snmax]
        ir = [i for i, r in enumerate(RR) if r <= rmax]
        out['classes'].append(dict(
            name=name, gaze_max_deg=amax, sigma_n_max=snmax, r_max=rmax,
            noise=decompose(noise[np.ix_(isn, np.where(ja)[0])]),
            pitch=decompose(pitch[np.ix_(ir, np.where(ja)[0])])))

    # range-free logarithmic sensitivities at the reference point
    s_on = np.array([conv['noise/sn%.2f_a+00' % s]['s'] for s in SN])
    r_on = np.array([conv['pitch/r%.3f_a+00' % r]['s'] for r in RR])
    a_on = np.array([conv['noise/sn0.10_a%+03d' % a]['s'] for a in ANG.astype(int)])
    ln = lambda x: np.log(np.asarray(x, float))
    el_sn = float(np.polyfit(ln(SN[:3]), ln(s_on[:3]), 1)[0])
    el_r = float(np.polyfit(ln(RR), ln(r_on), 1)[0])
    # gaze is not a ratio scale, so its elasticity is quoted over cos(alpha), the
    # foreshortening factor the projection actually depends on
    ca = np.cos(np.radians(ANG))
    el_a = float(np.polyfit(ln(ca[ca > 0.4]), ln(a_on[ca > 0.4]), 1)[0])
    out['elasticity'] = dict(d_ln_s_d_ln_sigma_n=el_sn,
                             d_ln_s_d_ln_r=el_r,
                             d_ln_s_d_ln_cos_alpha=el_a)
    def span(sn):
        on = conv['noise/sn%.2f_a+00' % sn]['s']
        ext = max(conv['noise/sn%.2f_a-60' % sn]['s'], conv['noise/sn%.2f_a+60' % sn]['s'])
        return float(100 * (ext - on) / on)
    out['gaze_span_pct'] = dict(sn005=span(0.05), sn010=span(0.10))
    return out


# ============================================================== lever arm
def lever_arm(conv):
    """The lever arm and the gaze-angle equivalent of the on-axis uncertainty.

    The quantity that turns a gaze rotation into a translation of the pupil centre is
    the distance from the centre of rotation of the eye to the plane of the entrance
    pupil, not the radius of the globe. The centre of rotation lies about 13.5 mm
    behind the corneal vertex and the entrance pupil about 3.05 mm behind it, so the
    lever arm is about 10.4 mm, with a standard uncertainty of 0.5 mm. tremor.py
    propagates it into the tremor requirements.
    """
    L0, uL = 10.45e-3, 0.50e-3                 # m
    s_05 = conv['noise/sn0.05_a+00']['s'] * 1e-6
    s_10 = conv['noise/sn0.10_a+00']['s'] * 1e-6
    s_50 = conv['noise/sn0.50_a+00']['s'] * 1e-6
    return dict(L_mm=L0 * 1e3, uL_mm=uL * 1e3,
                deg_per_s=dict(sn005_deg=float(np.degrees(s_05 / L0)),
                               sn010_deg=float(np.degrees(s_10 / L0)),
                               sn050_deg=float(np.degrees(s_50 / L0))))


# ============================================================== imaging geometry
def imaging_geometry():
    """r = p / m. One worked example, so that a data sheet maps onto the tables."""
    ex = []
    for p_um, m in ((4.8, 0.100), (3.45, 0.100), (4.8, 0.048), (2.74, 0.0274)):
        ex.append(dict(sensor_pitch_um=p_um, magnification=m,
                       r_mm_per_line=p_um / m / 1e3,
                       pupil_px=5.0 / (p_um / m / 1e3)))
    return dict(relation='r = p / m', examples=ex)


def main():
    conv = json.load(open(os.path.join(RES, 'converged_os5.json')))
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {}
    doc['variance'] = instrument_classes(conv)
    doc['lever_arm'] = lever_arm(conv)
    doc['imaging'] = imaging_geometry()
    json.dump(doc, open(OUT, 'w'), indent=1)

    print('--- variance shares [factor, gaze, interaction] ---')
    print('full noise cross-section  :', np.round(doc['variance']['full_range']['noise'], 1))
    print('full pitch cross-section  :', np.round(doc['variance']['full_range']['pitch'], 1))
    for c in doc['variance']['classes']:
        print('%-28s noise %s   pitch %s'
              % (c['name'], np.round(c['noise'], 1), np.round(c['pitch'], 1)))
    print('elasticities:', {k: round(v, 2)
                            for k, v in doc['variance']['elasticity'].items()})
    print()
    d = doc['lever_arm']['deg_per_s']
    print('--- lever arm %.2f +- %.2f mm ---' % (doc['lever_arm']['L_mm'],
                                                doc['lever_arm']['uL_mm']))
    print('on-axis uncertainty in degrees: %.3f at sn=0.05, %.3f at sn=0.10, '
          '%.2f at sn=0.50' % (d['sn005_deg'], d['sn010_deg'], d['sn050_deg']))
    print()
    print('--- Rayleigh fit, parametric bootstrap ---')
    with Pool(cpu_count()) as pool:
        doc['rayleigh'] = rayleigh_bootstrap(pool)
    json.dump(doc, open(OUT, 'w'), indent=1)
    r = doc['rayleigh']
    print('N = %d, sigma = %.3f +- %.3f mm, D = %.4f, critical D(95%%) = %.4f, '
          'bootstrap p = %.3f' % (r['n'], r['sigma_mm'], r['sigma_u_mm'], r['D'],
                                  r['D_crit_95'], r['p_bootstrap']))
    print('sx = %.1f, sy = %.1f um, rho = %.3f, sy/sx = %.3f'
          % (r['sx_um'], r['sy_um'], r['rho'], r['sy_um'] / r['sx_um']))
    print('\nsaved', OUT)


if __name__ == '__main__':
    main()
