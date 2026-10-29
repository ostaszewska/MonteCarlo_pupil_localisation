# The original MATLAB/Octave model

The implementation the Python code in `../python/` was ported from (R. Szewczyk and
A. Ostaszewska-Liżewska, MIT licence, as stated in the file headers), deposited for
provenance. Every published result comes from the Python implementation; nothing in
`../../results/` was produced with this code.

Written for GNU Octave; it runs in MATLAB unchanged except that `pkg load parallel`
and `pararrayfun` have no MATLAB equivalent, so set `fast = 1` there.

| File | Purpose |
|------|---------|
| `IrisC.m` | The forward model of one trial: eyeball and pupil geometry, harmonic shape distortion, rotation, projection, rendering grid, 5x5 block averaging, noise, threshold, centroid. Returns `[xcR ycR xcM ycM xcMr ycMr]`. |
| `CheckIsInT.m` | Inside/outside test over the grid; `fast` selects `CheckIsIn` (0), serial `CheckIsInF` (1) or `pararrayfun` over `CheckIsInF` (2). |
| `CheckIsInF.m` | Radial angle-to-length interpolation between the two boundary vertices bracketing the query angle; ported as `iris_model.points_in_shape_radial`. |
| `CheckIsIn.m` | The original slower inside test. |
| `reduce5.m` | Averages 5x5 blocks of the rendering grid into one sensor pixel. |
| `ArcDeg.m`, `VectL.m`, `VectNorm.m`, `RotVect*.m` | Geometry helpers. |
| `PopulationDistortion.m` | The pupil-shape population after Wyatt (1995). |
| `iris14.m` | Noise sweep driver: `DensCCD = 1`, ten noise levels, nine gaze angles. |
| `iris15_resolution.m` | Resolution sweep driver: seven `DensCCD` values at `sigma_n = 0.10`. |

## Differences from the code as originally written

- Pupil-shape amplitudes: The original drivers applied Wyatt's normalised harmonic
  amplitudes as absolute diameter modulations in metres (noncircularity 0.30 instead
  of 0.0176). Both drivers now draw the shape from `PopulationDistortion.m`, which
  converts them to a diameter modulation, as `iris_model.population_distortion` does.
- Failed trials: A trial whose thresholded image has no foreground pixel returns
  NaN; both drivers discard such trials before taking `std`.
- Sensor pitch: `IrisC.m` renders on a grid of `5*DensCCD` lines per mm and
  `reduce5.m` averages it 5x5, so one sensor pixel has the pitch `r = 1/DensCCD`
  mm/line. `iris15_resolution.m` reports and saves that pitch (`rSensorT`) alongside
  the rendering pitch `rT = 1/(5*DensCCD)` of the original header.

Edits other than the shape model are marked `[mod]` in the source. Line endings were
normalised to LF.

## Running it

```
octave-cli iris14.m              # noise sweep
octave-cli iris15_resolution.m   # resolution sweep
```

`fast = 2` needs the Octave `parallel` package. With `fast = 1`, one trial at
`DensCCD = 1` takes about 1.4 s.
