# -*- coding: utf-8 -*-
"""Grid convergence of the internal rendering grid.

Three levels of the rendering grid, run with identical seed streams so that only the
discretisation of the coverage fraction differs, at every point computed on all three:

    OS = 1   coarse    h3 = 1
    OS = 3   medium    h2 = 1/3
    OS = 5   fine      h1 = 1/5      <- the grid all results are reported on

Richardson/GCI analysis (Roache): observed order of convergence p, extrapolated value
and the Grid Convergence Index of the finest solution. The GCI is an approximate 95 %
band, so it enters the budget as u_num = GCI * f1 / 2.

Writes results/gci.json.
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))

R21 = (1.0 / 3.0) / (1.0 / 5.0)     # h2/h1 = 5/3
R32 = 1.0 / (1.0 / 3.0)             # h3/h2 = 3


def observed_order(e21, e32, r21=R21, r32=R32, tol=1e-10, itmax=500):
    """Roache's iterative solution for p with unequal refinement ratios."""
    if e21 == 0 or e32 == 0:
        return None
    s = np.sign(e32 / e21)
    p = 2.0
    for _ in range(itmax):
        q = np.log((r21 ** p - s) / (r32 ** p - s))
        pn = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if not np.isfinite(pn):
            return None
        if abs(pn - p) < tol:
            return float(pn)
        p = 0.5 * (p + pn)          # damped, the raw iteration can oscillate
    return float(p)


def gci_point(f1, f2, f3, p_fixed=None):
    """f1 finest (OS=5), f2 medium (OS=3), f3 coarsest (OS=1)."""
    e21, e32 = f2 - f1, f3 - f2
    p_obs = observed_order(e21, e32)
    p = p_fixed if p_fixed is not None else p_obs
    out = dict(f1=f1, f2=f2, f3=f3, e21=e21, e32=e32, p_obs=p_obs, p_used=p)
    if p is None or not np.isfinite(p) or p <= 0:
        return out
    f_ext = (R21 ** p * f1 - f2) / (R21 ** p - 1.0)
    ea = abs((f1 - f2) / f1)
    gci = 1.25 * ea / (R21 ** p - 1.0)
    out.update(f_ext=f_ext,
               ea_pct=100 * ea,
               gci_pct=100 * gci,
               u_num_um=abs(gci * f1) / 2.0,
               ext_rel_pct=100 * (f_ext - f1) / f1)
    return out


def main():
    d = {os_: json.load(open(os.path.join(RES, 'converged_os%d.json' % os_)))
         for os_ in (1, 3, 5)}
    for os_ in (1, 3, 5):
        assert d[os_]['meta']['oversample'] == os_

    keys = [k for k in d[5] if k != 'meta' and k in d[3] and k in d[1]]
    keys = [k for k in keys if all(d[o][k].get('s') for o in (1, 3, 5))]

    # ---- pass 1: observed order where the two differences are well separated
    rows = {}
    ps = []
    for k in keys:
        f1, f2, f3 = (d[o][k]['s'] for o in (5, 3, 1))
        g = gci_point(f1, f2, f3)
        rows[k] = g
        # keep only points where both differences are clearly above the paired
        # statistical floor (the runs share seeds, so that floor is small; 0.5 %
        # of the value is a conservative cut) and monotone
        if (g['p_obs'] is not None and np.isfinite(g['p_obs'])
                and abs(g['e21']) > 0.005 * f1 and abs(g['e32']) > 0.005 * f1
                and g['e21'] * g['e32'] > 0 and 0.3 < g['p_obs'] < 6.0):
            ps.append(g['p_obs'])

    p_bar = float(np.median(ps)) if ps else 1.0

    # ---- pass 2: one global order for every point, which is the usual practice
    # when the point-by-point p scatters. Report both.
    for k in keys:
        f1, f2, f3 = (d[o][k]['s'] for o in (5, 3, 1))
        rows[k].update({('g_' + a): b for a, b in gci_point(f1, f2, f3, p_bar).items()
                        if a in ('f_ext', 'gci_pct', 'u_num_um', 'ext_rel_pct', 'ea_pct')})
        rows[k].update(study=d[5][k]['study'], r_mm=d[5][k]['r_mm'],
                       sigma_n=d[5][k]['sigma_n'], alpha=d[5][k]['alpha'])

    gcis = [rows[k]['g_gci_pct'] for k in keys if 'g_gci_pct' in rows[k]]
    out = dict(meta=dict(levels={'coarse': 1, 'medium': 3, 'fine': 5},
                         r21=R21, r32=R32, Fs=1.25,
                         p_points_used=len(ps), p_observed_median=p_bar,
                         p_observed_iqr=[float(np.percentile(ps, 25)),
                                         float(np.percentile(ps, 75))] if ps else None,
                         n_points=len(keys),
                         gci_median_pct=float(np.median(gcis)),
                         gci_p95_pct=float(np.percentile(gcis, 95)),
                         gci_max_pct=float(np.max(gcis))),
               points=rows)
    json.dump(out, open(os.path.join(RES, 'gci.json'), 'w'), indent=1)

    print('points with a usable three-level triplet : %d' % len(keys))
    print('points contributing to the order estimate: %d' % len(ps))
    print('observed order of convergence p = %.2f  (IQR %.2f - %.2f)'
          % (p_bar, np.percentile(ps, 25), np.percentile(ps, 75)))
    print('GCI on the finest grid: median %.1f %%, 95th pct %.1f %%, max %.1f %%'
          % (np.median(gcis), np.percentile(gcis, 95), np.max(gcis)))
    print()
    hdr = '%-18s %8s %8s %8s %8s %8s %8s' % ('point', 'OS1', 'OS3', 'OS5',
                                             'ext', 'GCI%', 'u_num')
    for study in ('budget', 'pitch', 'fine'):
        print('--- %s ---' % study)
        print(hdr)
        for k in keys:
            if rows[k]['study'] != study:
                continue
            if study == 'pitch' and rows[k]['alpha'] != 0.0:
                continue
            g = rows[k]
            print('%-18s %8.1f %8.1f %8.1f %8.1f %8.2f %8.1f'
                  % (k, g['f3'], g['f2'], g['f1'], g.get('g_f_ext', float('nan')),
                     g.get('g_gci_pct', float('nan')), g.get('g_u_num_um', float('nan'))))
    print('--- noise on axis ---')
    print(hdr)
    for k in keys:
        if rows[k]['study'] == 'noise' and rows[k]['alpha'] == 0.0:
            g = rows[k]
            print('%-18s %8.1f %8.1f %8.1f %8.1f %8.2f %8.1f'
                  % (k, g['f3'], g['f2'], g['f1'], g.get('g_f_ext', float('nan')),
                     g.get('g_gci_pct', float('nan')), g.get('g_u_num_um', float('nan'))))
    print('\nsaved results/gci.json')


if __name__ == '__main__':
    main()
