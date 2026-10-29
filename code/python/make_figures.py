# -*- coding: utf-8 -*-
"""Figures drawn from the analysis outputs in results/.

Each figure is written whole (figures/<name>.png) and panel by panel
(figures/<name>_a.png/.pdf, <name>_b.png/.pdf, ...):

    uncertainty_pitch_gaze   u over sampling pitch and gaze angle
    lpw_coverage_bias        coverage-fraction bias model on real (LPW) pupil geometry
    bias_correction          bias, prediction and residual; RMSE before and after
    bias_direction           direction of the bias vector over noise and gaze
    estimators               four estimators against noise and gaze angle
    tremor_requirements      sigma_within(r) against the tremor requirements

The bias vectors over noise levels and gaze angles, which no other script writes, are
computed here at the reporting grid and cached in results/bias_vectors.json.

    python make_figures.py            # compute what is missing, then draw
    python make_figures.py --draw     # draw only, from the cache
"""
import argparse
import json
import math
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from multiprocessing import Pool, cpu_count

import iris_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))
FIG = os.path.normpath(os.path.join(HERE, '..', '..', 'figures'))
CACHE = os.path.join(RES, 'bias_vectors.json')
VERS = np.array([0.0, 1.0, 2.0])
OS = 5

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.grid': True,
    'grid.alpha': 0.35, 'axes.axisbelow': True, 'figure.dpi': 160,
    'savefig.dpi': 600, 'legend.framealpha': 0.9,
})



C_SIM = '#1f4e79'
C_MOD = '#c0392b'
C_ALT = '#27ae60'
C_AUX = '#7f8c8d'


def load(name, where=RES):
    p = os.path.join(where, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def panel(ax, letter, threed=False):
    """Panel letter. Panels are stacked vertically, so the letter goes at the top
    left of each one; a 3-D axes carries a lot of internal padding, so its letter is
    placed inside the box rather than above it."""
    kw = dict(fontsize=12, fontweight='bold', va='bottom', ha='left')
    if threed:
        ax.text2D(0.02, 0.88, letter, transform=ax.transAxes, **kw)
    else:
        ax.text(-0.15, 1.02, letter, transform=ax.transAxes, **kw)


def save(fig, name):
    """Write a figure whole (name.png) and panel by panel (name_a.png, ...)."""
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + '.png'), bbox_inches='tight')
    print('  wrote %s.png' % name)
    save_panels(fig, name)
    plt.close(fig)


def save_panels(fig, name):
    """Write each panel of a stacked figure on its own, without its panel letter,
    as name_<letter>.png and .pdf."""
    main = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
    cbars = [ax for ax in fig.axes if ax.get_label() == '<colorbar>']
    letters = [t for ax in main for t in ax.texts
               if re.fullmatch(r'\([a-z]\)', t.get_text())]
    for t in letters:
        t.set_visible(False)
    rend = fig.canvas.get_renderer()
    for letter, ax in zip('abcdefgh', main):
        pos = ax.get_position()
        boxes = [ax.get_tightbbox(rend)] + [
            c.get_tightbbox(rend) for c in cbars
            if c.get_position().y0 < pos.y1 and c.get_position().y1 > pos.y0]
        bb = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(0.05)
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(FIG, '%s_%s.%s' % (name, letter, ext)),
                        bbox_inches=bb)
        print('  wrote %s_%s.png/.pdf' % (name, letter))
    for t in letters:
        t.set_visible(True)


# ============================================================ auxiliary runs
def _trial(task):
    seed, dens, sn, angle, shape = task
    rng = np.random.default_rng(int(seed))
    amp, ph = ((np.zeros(7), np.zeros(7)) if shape == 'circ'
               else M.population_distortion(rng))
    block = M.REDUCE * OS
    dens_hr = dens * M.RES_MULT * OS
    bnd, ycR, zcR = M.build_pupil_boundary(VERS, angle, amp, ph)
    Yb, Zb = bnd[:, 0], bnd[:, 1]
    mX = M.MARGP * (Yb.max() - Yb.min())
    mY = M.MARGP * (Zb.max() - Zb.min())
    x0 = Yb.min() - mX * (rng.random() + 1.0)
    x1 = Yb.max() + mX * (rng.random() + 1.0)
    y0 = Zb.min() - mY * (rng.random() + 1.0)
    y1 = Zb.max() + mY * (rng.random() + 1.0)
    nx = max(int(np.floor(1e3 * dens_hr * (x1 - x0))), block)
    ny = max(int(np.floor(1e3 * dens_hr * (y1 - y0))), block)
    PX, PY = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    ins = M.points_in_shape_radial(PX.ravel(), PY.ravel(), Yb, Zb,
                                   ycR, zcR).reshape(PX.shape).astype(float)
    XT, YT, V = M._block_average(PX, PY, ins, block)
    V = V + rng.standard_normal(V.shape) * sn
    B = V >= M.THRESH
    if B.sum() == 0:
        return (np.nan, np.nan)
    return (XT[B].mean() - ycR, YT[B].mean() - zcR)


def run(pool, rep, dens, sn, angle, shape='pop', seed=1):
    ss = np.random.SeedSequence(seed).spawn(rep)
    a = np.array(pool.map(_trial, [(s.generate_state(1)[0], dens, sn, angle, shape)
                               for s in ss], chunksize=16))
    return a[~np.isnan(a).any(axis=1)]


def compute(nproc=None):
    doc = load('bias_vectors.json') or {}
    with Pool(processes=nproc or cpu_count()) as pool:
        if 'biasvec' not in doc:
            rows = []
            for sn in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
                for al in (0.0, 30.0, 60.0):
                    a = run(pool, 1800, 1.0, sn, al, seed=int(4000 + 100 * sn + al))
                    bx, by = a[:, 0].mean() * 1e6, a[:, 1].mean() * 1e6
                    rows.append(dict(sigma_n=sn, alpha=al, bx=float(bx),
                                     by=float(by), mag=float(np.hypot(bx, by)),
                                     phi=float(np.degrees(np.arctan2(by, bx)))))
                    print('    biasvec sn=%.2f a=%.0f  |b|=%.1f phi=%.0f'
                          % (sn, al, rows[-1]['mag'], rows[-1]['phi']), flush=True)
            doc['biasvec'] = rows
            json.dump(doc, open(CACHE, 'w'))

    return doc


def resolvable(rows, conv=None, rep=1800, k=5.0):
    """Which bias vectors have a direction at all.

    The direction of a mean vector is meaningless once its magnitude falls to the
    Monte Carlo scatter of that mean, which is u/sqrt(rep) per axis. Points are kept
    only where |b| exceeds k times that scatter, so the dip below the transition knee,
    where the bias passes through zero, does not contribute a random angle.
    """
    conv = conv if conv is not None else load('converged_os5.json')
    out = []
    for r in rows:
        u = conv['noise/sn%.2f_a%+03d' % (r['sigma_n'], int(r['alpha']))]['s']
        out.append(r['mag'] > k * u / math.sqrt(rep))
    return np.array(out)


# ============================================================ the figures
ANG = list(range(-60, 61, 15))
SN_BUDGET = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
RR = [0.500, 0.667, 1.000, 1.429, 2.000, 2.500, 3.333]


def figure_uncertainty_pitch_gaze(conv):
    R = np.array(RR[:6])
    A = np.array(ANG, float)
    Z = np.array([[conv['pitch/r%.3f_a%+03d' % (r, int(a))]['s'] for a in A] for r in R])
    fig = plt.figure(figsize=(6.6, 8.8))
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    AA, RRg = np.meshgrid(A, R)
    ax.plot_surface(RRg, AA, Z, cmap='viridis', linewidth=0, antialiased=True,
                    alpha=0.95)
    ax.set_xlabel('$r$ [mm/line]'); ax.set_ylabel(r'$\alpha$ [deg]')
    ax.set_zlabel(r'$u$ [$\mu$m]')
    ax.view_init(24, -132)
    panel(ax, '(a)', threed=True)

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(R, Z[:, list(A).index(0)], 'o-', color=C_SIM, label=r'$\alpha = 0$')
    ax.plot(R, Z[:, 0], 's--', color=C_MOD, label=r'$\alpha = -60^\circ$')
    r33 = [conv['pitch/r3.333_a%+03d' % int(a)]['s'] for a in A]
    ax.plot([3.333], [r33[list(A).index(0)]], 'o', mfc='none', color=C_SIM)
    ax.plot([3.333], [r33[0]], 's', mfc='none', color=C_MOD)
    av = sorted(1 - conv['pitch/r3.333_a%+03d' % a]['discard'] for a in (-60, 60))
    ax.annotate('availability %.2f-%.2f' % tuple(av), xy=(3.333, r33[0]),
                xytext=(1.75, 700),
                fontsize=9, color=C_AUX,
                arrowprops=dict(arrowstyle='->', color=C_AUX, lw=0.8))
    ax.set_xlabel('$r$ [mm/line]'); ax.set_ylabel(r'$u$ [$\mu$m]')
    ax.legend(loc='upper left')
    inv = lambda x: 5.0 / np.maximum(x, 1e-9)
    sec = ax.secondary_xaxis('top', functions=(inv, inv))
    sec.set_xticks([10, 7, 5, 3.5, 2.5, 2, 1.5])    # fixed, or they crowd at the left
    sec.set_xlabel('pupil sampling [px]')
    panel(ax, '(b)')
    fig.tight_layout()
    save(fig, 'uncertainty_pitch_gaze')


def figure_lpw_coverage_bias(lpw5, lpwsum):
    s = lpw5['summary']
    SN = np.array(s['sigma_n'])
    p5 = lpwsum['px5']
    F = lpw5['frames']
    P = np.array([f['b_pred_um'] for f in F])
    Mm = np.array([f['b_meas_um'] for f in F])

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 8.8))
    ax = axes[0]
    meas = [q['mean_meas_um'] for q in s['per_sigma']]
    ax.axvspan(0.15, 0.20, color='0.9', zorder=0)
    ax.plot(SN, np.array(p5['curve_meas_um']), 'o-', color=C_SIM,
            label='measured on real frames')
    ax.plot(SN, np.array(p5['curve_pred_um']), 's--', color=C_MOD,
            label='coverage-fraction model on the measured coverage map')
    ax.set_xlabel(r'$\sigma_{\mathrm{eff}}$')
    ax.set_ylabel(r'threshold bias [$\mu$m]')
    ax.legend(fontsize=9)
    ax.set_title('frame-averaged, %d frames' % s['n_frames'], fontsize=10)
    panel(ax, '(a)')

    ax = axes[1]
    ok = np.isfinite(P) & np.isfinite(Mm)
    ax.plot(Mm[ok], P[ok], '.', ms=4, color=C_SIM, alpha=0.55)
    lim = [0, np.nanpercentile(Mm[ok], 99.5)]
    ax.plot(lim, lim, '-', color=C_MOD, lw=1.2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r'measured bias, per frame [$\mu$m]')
    ax.set_ylabel(r'predicted by the coverage-fraction model [$\mu$m]')
    ax.set_title('median |deviation| %.1f %%' % p5['frame_median_abs_rel_pct'],
                 fontsize=10)
    panel(ax, '(b)')
    fig.tight_layout()
    save(fig, 'lpw_coverage_bias')


def figure_bias_correction(alt):
    S = np.array(SN_BUDGET)
    c = [alt['noise']['sn%.2f' % s]['centroid'] for s in S]
    pr = [alt['noise']['sn%.2f' % s]['predicted_bias'] for s in S]
    co = [alt['noise']['sn%.2f' % s]['corrected'] for s in S]

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 8.8))
    ax = axes[0]
    ax.plot(S, [x['bias'] for x in c], 'o-', color=C_SIM, label='simulated bias')
    ax.plot(S, pr, 's--', color=C_MOD,
            label='coverage-fraction model, no fitted parameter')
    ax.plot(S, [x['bias'] for x in co], '^:', color=C_ALT, label='residual after correction')
    ax.set_xlabel(r'$\sigma_{\mathrm{eff}}$'); ax.set_ylabel(r'bias [$\mu$m]')
    ax.legend(fontsize=9)
    panel(ax, '(a)')

    ax = axes[1]
    ax.plot(S, [math.hypot(x['bias'], x['s']) for x in c], 'o-', color=C_SIM,
            label='RMSE, uncorrected')
    ax.plot(S, [math.hypot(x['bias'], x['s']) for x in co], 's--', color=C_ALT,
            label='RMSE, corrected')
    ax.plot(S, [x['s'] for x in c], ':', color=C_AUX, label=r'$\sigma_{\mathrm{within}}$')
    ax.set_xlabel(r'$\sigma_{\mathrm{eff}}$'); ax.set_ylabel(r'[$\mu$m]')
    ax.legend(fontsize=9)
    panel(ax, '(b)')
    fig.tight_layout()
    save(fig, 'bias_correction')


def figure_bias_direction(fd):
    rows = fd['biasvec']
    bx = np.array([r['bx'] for r in rows]); by = np.array([r['by'] for r in rows])
    sn = np.array([r['sigma_n'] for r in rows])
    phi = np.array([r['phi'] for r in rows])
    mag = np.array([r['mag'] for r in rows])
    # The bias reverses at the transition knee, so the two branches are summarised
    # separately; points whose magnitude is at the Monte Carlo scatter of the mean
    # have no direction at all and are shown but not used.
    KNEE = 0.20
    ok = resolvable(rows)
    above, below = ok & (sn >= KNEE), ok & (sn < KNEE)

    def circmean(a):
        return float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(a))),
                                           np.mean(np.cos(np.radians(a))))))

    phi_hi, phi_lo = circmean(phi[above]), circmean(phi[below])
    dev = np.degrees(np.angle(np.exp(1j * np.radians(phi[above] - phi_hi))))

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 8.8))
    ax = axes[0]
    sc = ax.scatter(bx[sn >= KNEE], by[sn >= KNEE], c=sn[sn >= KNEE], cmap='viridis',
                    s=34, zorder=3, label=r'$\sigma_{\mathrm{eff}} \geq %.2f$' % KNEE)
    ax.scatter(bx[sn < KNEE], by[sn < KNEE], facecolors='none', edgecolors=C_AUX,
               s=40, zorder=3, label=r'$\sigma_{\mathrm{eff}} < %.2f$' % KNEE)
    lim = max(abs(bx).max(), abs(by).max()) * 1.2
    t = np.linspace(-lim, lim, 2)
    ax.plot(t * math.cos(math.radians(phi_hi)), t * math.sin(math.radians(phi_hi)),
            '--', color=C_MOD, lw=1.1,
            label=r'$\varphi_0 = %.0f^\circ$ above the knee' % phi_hi)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axhline(0, color='0.8', lw=0.8, zorder=0)
    ax.axvline(0, color='0.8', lw=0.8, zorder=0)
    ax.set_xlabel(r'$b_x$ [$\mu$m]'); ax.set_ylabel(r'$b_y$ [$\mu$m]')
    ax.legend(fontsize=8, loc='upper left')
    fig.colorbar(sc, ax=ax, label=r'$\sigma_{\mathrm{eff}}$')
    panel(ax, '(a)')

    ax = axes[1]
    ax.plot(sn[above], phi[above], 'o', color=C_SIM, label='resolvable direction')
    ax.plot(sn[below], phi[below], 'o', mfc='none', color=C_AUX,
            label='below the knee')
    ax.plot(sn[ok & (sn < KNEE)], phi[ok & (sn < KNEE)], 'o', color=C_AUX)
    ax.axhline(phi_hi, color=C_MOD, ls='--',
               label=r'$%.0f^\circ$, r.m.s. %.0f$^\circ$'
                     % (phi_hi, float(np.sqrt((dev ** 2).mean()))))
    ax.axhline(phi_lo, color=C_ALT, ls=':', label=r'$%.0f^\circ$' % phi_lo)
    ax.axvline(KNEE, color='0.8', lw=0.8, zorder=0)
    ax.set_xlabel(r'$\sigma_{\mathrm{eff}}$')
    ax.set_ylabel(r'direction of $\mathbf{b}$ [deg]')
    ax.legend(fontsize=8)
    panel(ax, '(b)')
    fig.tight_layout()
    save(fig, 'bias_direction')


def figure_tremor_requirements(trem, fine_u):
    tg = trem['targets']
    rs = np.array(sorted(fine_u))
    us = np.array([fine_u[r] for r in rs])
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 8.8))
    ax = axes[0]
    ax.loglog(rs, us, 'o-', color=C_SIM, label=r'$\sigma_{\mathrm{within}}(r)$')
    cols = ['#8e44ad', '#e67e22', '#16a085']
    for (k, v), col in zip(tg.items(), cols):
        ax.axhline(v['s_req_spectral_um'], ls='--', color=col, lw=1.1,
                   label='spectral, %s' % k.split(' (')[0])
        ax.axhline(v['s_req_waveform_um'], ls=':', color=col, lw=1.1)
    ax.set_xlabel('$r$ [mm/line]'); ax.set_ylabel(r'$u$ [$\mu$m]')
    ax.legend(fontsize=8)
    panel(ax, '(a)')

    ax = axes[1]
    arc = np.array([6.0, 17.5, 60.0])
    rr = np.array([tg[k]['r_spectral_mm'] for k in tg])
    lo = np.array([tg[k]['r_spectral_ci95'][0] for k in tg])
    hi = np.array([tg[k]['r_spectral_ci95'][1] for k in tg])
    ax.errorbar(arc, rr, yerr=[rr - lo, hi - rr], fmt='o-', color=C_ALT, capsize=4,
                label='spectral detection')
    ax.axhspan(rs.min(), rs.max(), color='0.9', zorder=0, label='simulated pitch range')
    ax.set_xscale('log')
    ax.set_xlabel('tremor amplitude [arcsec]')
    ax.set_ylabel(r'required $r$ [mm/line]')
    ax.legend(fontsize=9)
    panel(ax, '(b)')
    fig.tight_layout()
    save(fig, 'tremor_requirements')


def figure_estimators(alt):
    S = np.array(SN_BUDGET)
    names = [('centroid', 'binary centroid', C_SIM, 'o-'),
             ('weighted_t', 'weighted, gated', C_ALT, 's--'),
             ('weighted', 'weighted, plain', '#8e44ad', 'v:'),
             ('ellipse', 'ellipse fit', C_MOD, '^-.')]
    def take(point, key, field):
        """A conic recovered in a few per cent of trials is not an estimate."""
        e = point.get(key)
        if e is None:
            return np.nan
        cs = point.get('centroid')
        if key == 'ellipse' and (point.get('ellipse_fit_rate', 1.0) < 0.5
                                 or (cs and e['s'] > 10 * cs['s'])):
            return np.nan
        return e[field]

    fig, axes = plt.subplots(3, 1, figsize=(6.6, 12.6))
    for key, lab, col, st in names:
        v = [alt['noise']['sn%.2f' % s] for s in S]
        axes[0].plot(S, [take(p, key, 's') for p in v], st, color=col, label=lab)
        axes[1].plot(S, [take(p, key, 'bias') for p in v], st, color=col, label=lab)
    axes[0].set_xlabel(r'$\sigma_{\mathrm{eff}}$')
    axes[0].set_ylabel(r'$u$ [$\mu$m]'); axes[0].legend(fontsize=8)
    panel(axes[0], '(a)')
    axes[1].set_xlabel(r'$\sigma_{\mathrm{eff}}$')
    axes[1].set_ylabel(r'$|\mathbf{b}|$ [$\mu$m]'); axes[1].legend(fontsize=8)
    panel(axes[1], '(b)')

    A = [0, 30, 60]
    for key, lab, col, st in names:
        v = [alt['gaze'].get('a%02d' % a, {}) for a in A]
        axes[2].plot(A, [take(p, key, 's') for p in v], st, color=col, label=lab)
    axes[2].set_xlabel(r'$\alpha$ [deg]'); axes[2].set_ylabel(r'$u$ [$\mu$m]')
    axes[2].legend(fontsize=8)
    panel(axes[2], '(c)')
    fig.tight_layout()
    save(fig, 'estimators')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draw', action='store_true', help='do not compute, draw only')
    ap.add_argument('-j', '--nproc', type=int, default=None)
    args = ap.parse_args()

    fd = (load('bias_vectors.json') or {}) if args.draw else compute(args.nproc)
    conv = load('converged_os5.json')
    trem = load('tremor.json')
    alt = load('alternatives.json')
    lpw5 = load('lpw_coverage_bias_5px.json')
    lpwsum = load('lpw_coverage_bias_summary.json')
    fine_u = {float(r): v['sigma_within_um']
              for r, v in (trem or {}).get('fine_sweep', {}).items()}

    print('drawing:')
    if conv:
        figure_uncertainty_pitch_gaze(conv)
    if lpw5 and lpwsum:
        figure_lpw_coverage_bias(lpw5, lpwsum)
    if alt:
        figure_bias_correction(alt); figure_estimators(alt)
    if fd.get('biasvec'):
        figure_bias_direction(fd)
    if trem and fine_u:
        figure_tremor_requirements(trem, fine_u)
    print('done')


if __name__ == '__main__':
    main()
