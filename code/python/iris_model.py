# -*- coding: utf-8 -*-
"""
Faithful Python port of the eye-tracker pupil-uncertainty model.

Original MATLAB/Octave sources (R. Szewczyk / A. Ostaszewska-Lizewska, MIT):
    IrisC.m, CheckIsInT.m / CheckIsInF.m, reduce5.m, RotVect*.m, VectNorm.m

Unlike the MATLAB version, the Monte Carlo trials, not the inner grid scan, are
parallelised across CPU cores (see run_converged.py).

The point-in-shape test is a vectorised transcription of CheckIsInF.m, not a
straight-edge polygon test; points_in_polygon() below implements the ray-casting
alternative and is used only to check that the two converge on a common low-noise
asymptote once the boundary is finely sampled.

Everything else (dimensions, grid construction, 5x5 block averaging, Gaussian
noise, 0.5 threshold, centroid) reproduces the MATLAB code. The harmonic pupil-shape
distortion does not: the MATLAB code applies Wyatt's normalised amplitudes as absolute
diameter modulations in metres, which inflates them about twentyfold;
population_distortion() applies them to the pupil diameter (see code/matlab/README.md).
"""
import numpy as np


def points_in_shape_radial(px, py, vx, vy, cx, cy):
    """Exact port of CheckIsInF.m: radial (angle->length) interpolation test.

    A query point is INSIDE if its distance from the reference centre (cx,cy)
    is smaller than the boundary distance interpolated LINEARLY IN ANGLE between
    the two adjacent boundary vertices that bracket the query's angle, as the
    MATLAB code does (not a straight-edge polygon test). boundary_vertices.py
    compares it with ray casting.

    cx,cy : reference centre = projected ideal pupil centre (IrisC), as in
            CheckIsInT.m where Iris2DCX/Y = IrisC(2)/IrisC(3).
    """
    px = np.asarray(px); py = np.asarray(py)
    # boundary in polar coords about the reference centre
    bx = vx - cx; by = vy - cy
    th_b = np.degrees(np.arctan2(by, bx)) % 360.0       # [0,360)
    r_b = np.hypot(bx, by)
    order = np.argsort(th_b)
    th_s = th_b[order]; r_s = r_b[order]
    # query polar coords
    pth = np.degrees(np.arctan2(py - cy, px - cx)) % 360.0
    pr = np.hypot(px - cx, py - cy)
    # boundary radius at the query angle, linear-in-angle, periodic 360
    r_at = np.interp(pth, th_s, r_s, period=360.0)
    return pr < r_at


def points_in_polygon(px, py, vx, vy):
    """Vectorised crossing-number (ray-casting) point-in-polygon test.

    px,py : 1D arrays of query-point coordinates.
    vx,vy : closed-polygon vertices (first==last is fine).
    Returns boolean array (True = inside). Pure NumPy, no C extensions.
    """
    px = np.asarray(px); py = np.asarray(py)
    vx = np.asarray(vx); vy = np.asarray(vy)
    n = len(vx)
    inside = np.zeros(px.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = vx[i], vy[i], vx[j], vy[j]
        # does edge (j->i) straddle the horizontal ray from the point?
        cond = ((yi > py) != (yj > py))
        # x-coordinate of edge at the point's y
        with np.errstate(divide='ignore', invalid='ignore'):
            xcross = (xj - xi) * (py - yi) / (yj - yi) + xi
        inside ^= cond & (px < xcross)
        j = i
    return inside

# ----------------------------------------------------------------- constants
EYEBALL_D = 0.025          # m, eyeball diameter            (IrisC.m)
IRIS_D    = 0.005          # m, pupil diameter
DENS      = 100            # boundary points per full turn
MARGP     = 0.5            # margin parameter
THRESH    = 0.5            # detection threshold t
RES_MULT  = 5             # DensCCD is multiplied by 5 internally (IrisC.m l.45)
REDUCE    = 5             # reduce5 block size


def sensor_pitch_mm(dens_ccd):
    """Physical pitch of ONE low-resolution sensor pixel, in mm per line.

    DensCCD is the line density of the CMOS sensor itself. The model renders the
    pupil on a grid RES_MULT (x oversample) times finer and then averages blocks
    of the same size, so the rendered grid cancels exactly and the sensor pitch
    is 1/DensCCD, independent of RES_MULT and of the oversampling factor. This
    is the quantity called r throughout.

    Do NOT use 1/(DensCCD*RES_MULT): that is the pitch of the internal rendering
    grid, five times finer than any pixel the centroid ever sees.
    """
    return 1.0 / dens_ccd


# ----------------------------------------------------------------- vector ops
def rot_x(vec, deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    M = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return M @ vec


def rot_vers(vec, vers, deg):
    """Rodrigues rotation of vec around unit axis vers by deg degrees."""
    r = np.deg2rad(deg)
    vers = vers / np.sqrt(np.sum(vers**2))
    return (vec * np.cos(r)
            + np.cross(vers, vec) * np.sin(r)
            + vers * np.dot(vers, vec) * (1.0 - np.cos(r)))


# ----------------------------------------------------------------- core model
def build_pupil_boundary(iris_rot_vers, iris_rot_deg, dist_amp, dist_phase):
    """Return projected boundary (Y,Z) and projected centre (yc,zc).

    Mirrors the iris-construction loop in IrisC.m. X is the optical axis;
    the CMOS sensor sees the projection onto the Y-Z plane, i.e. components
    (index 1,2) of each 3D point (0-based) == MATLAB rows (2,3).
    """
    iris_x = np.sqrt((EYEBALL_D / 2) ** 2 - (IRIS_D / 2) ** 2)
    n = DENS + 1
    fi = np.arange(n) / DENS * 360.0                     # degrees

    # harmonic-distorted radius (vectorised over fi)
    diam = np.full(n, IRIS_D)
    for j in range(len(dist_amp)):
        diam += dist_amp[j] * np.sin(np.deg2rad(fi * (j + 2) + dist_phase[j]))
    # note: MATLAB uses (j+1) with 1-based j=1..numel -> harmonic orders 2..N+1;
    # here 0-based j=0..N-1 with (j+2) gives the identical orders 2..N+1.

    bnd = np.empty((n, 3))
    for i in range(n):
        p0 = rot_x(np.array([iris_x, 0.0, diam[i] / 2.0]), fi[i])
        bnd[i] = rot_vers(p0, iris_rot_vers, iris_rot_deg)

    centre = rot_vers(np.array([iris_x, 0.0, 0.0]), iris_rot_vers, iris_rot_deg)

    bndYZ = bnd[:, 1:3]              # (n,2) -> (Y,Z)
    return bndYZ, centre[1], centre[2]


def single_trial(iris_rot_vers, iris_rot_deg, dist_amp, dist_phase,
                 dens_ccd, noise_l, rng, oversample=1):
    """One Monte-Carlo evaluation. Returns (xcR,ycR,xcM,ycM,xcMr,ycMr) in metres.

    Reproduces IrisC.m: build shape -> high-res grid inside/outside ->
    high-res centroid (xcM,ycM); reduce5 -> +Gaussian noise -> threshold ->
    low-res noisy centroid (xcMr,ycMr); ideal centre (xcR,ycR).

    oversample (OS): integer >=1. The PHYSICAL low-resolution sensor pixel is
    held FIXED (pitch = 1/(1000*dens_ccd) m, exactly as MATLAB). OS only refines
    the sub-pixel anti-aliasing used to compute each low-res pixel's pupil-
    coverage fraction: the boundary is rendered on a grid REDUCE*OS finer than
    the low-res pixel, then block-averaged back. OS=1 reproduces the original
    MATLAB 5x5 sub-sampling; larger OS -> the converged (grid-independent)
    coverage fraction, i.e. the true CMOS area integral. This removes the
    boundary-rasterisation artefact that dominates the low-noise regime.
    """
    # high-res lines/mm so that after reducing by (REDUCE*OS) the low-res pixel
    # pitch is independent of OS and equals the MATLAB value (dens_ccd*RES_MULT
    # reduced by REDUCE).
    block = REDUCE * oversample
    dens_hr = dens_ccd * RES_MULT * oversample   # high-res lines/mm

    bndYZ, ycR, zcR = build_pupil_boundary(
        iris_rot_vers, iris_rot_deg, dist_amp, dist_phase)
    Yb, Zb = bndYZ[:, 0], bndYZ[:, 1]

    # grid extent with randomised margins (IrisC.m l.127-133)
    margX = MARGP * (Yb.max() - Yb.min())
    startX = Yb.min() - margX * (rng.random() + 1.0)
    stopX  = Yb.max() + margX * (rng.random() + 1.0)
    margY = MARGP * (Zb.max() - Zb.min())
    startY = Zb.min() - margY * (rng.random() + 1.0)
    stopY  = Zb.max() + margY * (rng.random() + 1.0)

    nx = int(np.floor(1e3 * dens_hr * (stopX - startX)))
    ny = int(np.floor(1e3 * dens_hr * (stopY - startY)))
    nx = max(nx, block); ny = max(ny, block)
    x = np.linspace(startX, stopX, nx)
    y = np.linspace(startY, stopY, ny)
    PX, PY = np.meshgrid(x, y)

    # inside/outside via radial interpolation about the ideal centre,
    # exactly as CheckIsInF.m (the reference MATLAB test)
    inside = points_in_shape_radial(PX.ravel(), PY.ravel(),
                                    Yb, Zb, ycR, zcR).reshape(PX.shape).astype(float)

    # high-resolution (optimised) centroid
    s = inside.sum()
    xcM = (PX * inside).sum() / s
    ycM = (PY * inside).sum() / s

    # low-resolution camera: average (REDUCE*OS) blocks, add noise, threshold
    PXr, PYr, Vr = _block_average(PX, PY, inside, block)
    Vr = Vr + rng.standard_normal(Vr.shape) * noise_l
    Vr = (Vr >= THRESH).astype(float)

    sr = Vr.sum()
    if sr > 0:
        xcMr = (PXr * Vr).sum() / sr
        ycMr = (PYr * Vr).sum() / sr
    else:                                   # all-background after threshold
        xcMr = np.nan
        ycMr = np.nan

    return ycR, zcR, xcM, ycM, xcMr, ycMr
    # NB: ordering matches MATLAB [xcR ycR xcM ycM xcMr ycMr], where the
    # MATLAB "x"/"y" are the projected Y/Z components (rows 2,3).


def _block_average(XT, YT, VT, block):
    """Average non-overlapping block x block cells, as reduce5.m does.

    block = REDUCE * oversample. Partial border cells are dropped, so every
    output pixel is formed from a full block. The leftover cells fall at the
    high-index end of each axis, which sets the offset of the analysed window
    (see window_offset.py).
    """
    h = block // 2
    rows, cols = VT.shape
    ri = np.arange(h, rows - h, block)
    ci = np.arange(h, cols - h, block)
    XTr = XT[np.ix_(ri, ci)]
    YTr = YT[np.ix_(ri, ci)]
    VTr = np.empty((len(ri), len(ci)))
    for ii, c0 in enumerate(ri):
        for jj, d0 in enumerate(ci):
            VTr[ii, jj] = VT[c0 - h:c0 - h + block, d0 - h:d0 - h + block].mean()
    return XTr, YTr, VTr


# --------------------------------------------------------- pupil-shape model
# Harmonic amplitudes of the pupil margin, harmonics 2..8, from Wyatt (1995).
# Read off his Fig. 4, whose x-axis is 10^4 x the amplitude in NORMALISED form,
# r_n / r_ave in his equation (1). Mean and population spread, in those units:
_WYATT_SHAPE_BASE   = np.array([150., 100., 50., 50., 20., 20., 10.]) * 1e-4
_WYATT_SHAPE_SPREAD = np.array([75.,  50., 25., 25., 10., 10.,  5.]) * 1e-4

# Wyatt's measured noncircularity, his equation (3), mean over 23 subjects.
# The model pupil is 5 mm, i.e. his dark condition (he measured 4.93 mm dark,
# 3.09 mm light), so 'dark' is the matched value; 'light' is offered because
# fixation on a visible target is photopic.
WYATT_NC = {'dark': 0.0176, 'light': 0.0166}
SHAPE_CONDITION = 'dark'

# The seven numbers above give a mean NC of 0.01484 as read; one overall factor
# calibrates that onto Wyatt's measured mean while leaving the harmonic energy
# shares (0.586 for the ellipse, against his 0.596 in the dark) untouched. The
# constant is the mean NC of the uncalibrated draw; test_shape_model() checks it.
_NC_AS_READ = 0.014836



def _shape_arrays(condition=None):
    """Normalised radius amplitudes (mean, spread), calibrated onto Wyatt's NC."""
    cond = SHAPE_CONDITION if condition is None else condition
    k = WYATT_NC[cond] / _NC_AS_READ
    return _WYATT_SHAPE_BASE * k, _WYATT_SHAPE_SPREAD * k



def population_distortion(rng, condition=None):
    """Random population-based pupil-shape distortion, after Wyatt (1995).

    Returns (dist_amp, dist_phase) where dist_amp is the amplitude of the
    modulation of the pupil DIAMETER in metres, as consumed by
    build_pupil_boundary, and dist_phase is in degrees.

    A normalised radius amplitude r_n/r_ave corresponds to a diameter amplitude
    of (r_n/r_ave) * 2 * r_ave = (r_n/r_ave) * IRIS_D, which is the conversion
    applied here and the one missing from the original MATLAB.
    """
    dist_phase = rng.random(7) * 358.0 + 1.0
    l = np.clip(rng.standard_normal(7), -2, 2)
    base, spread = _shape_arrays(condition)
    return (base + l * spread) * IRIS_D, dist_phase


def noncircularity(dist_amp):
    """Wyatt's noncircularity, equation (3): NC^2 = 1/2 sum (r_n / r_ave)^2.

    dist_amp is the DIAMETER modulation in metres, so r_n = dist_amp/2 and
    r_ave = IRIS_D/2, giving r_n/r_ave = dist_amp/IRIS_D.
    """
    return float(np.sqrt(np.sum((np.asarray(dist_amp) / IRIS_D) ** 2) / 2.0))


def test_shape_model(n=20000, seed=0, verbose=True):
    """Check the pupil-shape model against the three numbers Wyatt reports.

    Applying the normalised amplitudes as absolute metres, as the original MATLAB
    does, inflates NC about 20-fold and fails this check. Returns True on success.
    """
    rng = np.random.default_rng(seed)
    amps = np.array([population_distortion(rng)[0] for _ in range(n)])
    nc = np.sqrt((amps / IRIS_D) ** 2 @ np.ones(7) / 2.0)
    energy = (amps ** 2).mean(axis=0)
    ellipse_share = energy[0] / energy.sum()
    target = WYATT_NC[SHAPE_CONDITION]
    # Wyatt: NC = 0.0176 +- 0.0062 (dark), 0.0166 +- 0.0041 (light);
    # ellipse contributes 0.596 of the energy in the dark, 0.477 in the light.
    checks = [('mean NC', nc.mean(), target, 0.05 * target),
              ('SD of NC', nc.std(), 0.0062 if SHAPE_CONDITION == 'dark' else 0.0041, 0.0020),
              ('ellipse energy share', ellipse_share, 0.596, 0.12)]
    ok = True
    for name, got, want, tol in checks:
        good = abs(got - want) <= tol
        ok &= good
        if verbose:
            print('  %-22s %8.4f   Wyatt %7.4f  +-%6.4f   %s'
                  % (name, got, want, tol, 'ok' if good else 'FAIL'))
    if verbose:
        print('  %-22s %8.6f' % ('_NC_AS_READ (recalc)',
                                 nc.mean() * _NC_AS_READ / target))
    return bool(ok)


if __name__ == '__main__':
    print('pupil-shape model against Wyatt (1995), condition = %r:' % SHAPE_CONDITION)
    raise SystemExit(0 if test_shape_model() else 1)
