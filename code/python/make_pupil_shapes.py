# -*- coding: utf-8 -*-
"""Examples of pupil shapes drawn from the population model.

The shapes are drawn with iris_model.population_distortion, the function every
simulation calls. Each panel carries the noncircularity of the shape it shows
(Wyatt's equation (3)).

Usage:
    python make_pupil_shapes.py     # six shapes -> figures/pupil_shapes_a.png ... _f.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import iris_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))
RPFIG = os.path.join(REPO, 'figures')

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.grid': True,
    'grid.alpha': 0.35, 'axes.axisbelow': True, 'figure.dpi': 160,
    'savefig.dpi': 600,
})

PANELS = 'abcdef'
SEED = 20260815


def outline(dist_amp, dist_phase, n=721):
    """Pupil outline in mm, as build_pupil_boundary constructs it (IrisC.m l.93)."""
    fi = np.arange(n) / (n - 1) * 360.0
    diam = np.full(n, M.IRIS_D)
    for j in range(len(dist_amp)):
        diam += dist_amp[j] * np.sin(np.deg2rad(fi * (j + 2) + dist_phase[j]))
    r = diam / 2.0 * 1e3
    th = np.deg2rad(fi)
    return r * np.cos(th), r * np.sin(th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=None, help='override the output directory')
    args = ap.parse_args()

    os.makedirs(RPFIG, exist_ok=True)
    rng = np.random.default_rng(SEED)
    r_ave = M.IRIS_D / 2.0 * 1e3
    ncs = []

    for letter in PANELS:
        amp, phase = M.population_distortion(rng)
        nc = M.noncircularity(amp)
        ncs.append(nc)
        x, y = outline(amp, phase)

        fig, ax = plt.subplots(figsize=(3.1, 3.1))
        th = np.linspace(0, 2 * np.pi, 721)
        ax.plot(r_ave * np.cos(th), r_ave * np.sin(th), ':', color='tab:red', lw=1.4)
        ax.plot(x, y, '-', color='black', lw=1.6)
        ax.set_aspect('equal')
        lim = 3.0
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
        ax.text(0.03, 0.97, 'NC = %.4f' % nc, transform=ax.transAxes,
                va='top', ha='left', fontsize=9)

        name = 'pupil_shapes_%s.png' % letter
        target = args.outdir or RPFIG
        os.makedirs(target, exist_ok=True)
        fig.savefig(os.path.join(target, name), bbox_inches='tight')
        plt.close(fig)

    ncs = np.array(ncs)
    print('condition        : %s' % M.SHAPE_CONDITION)
    print('NC of the 6 shown: %.4f to %.4f (mean %.4f)'
          % (ncs.min(), ncs.max(), ncs.mean()))
    print('Wyatt (1995)     : 0.0176 +- 0.0062 (dark), 0.0166 +- 0.0041 (light)')
    print('written to       : %s' % (args.outdir or RPFIG))


if __name__ == '__main__':
    main()
