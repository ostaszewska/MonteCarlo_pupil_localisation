# Monte Carlo uncertainty evaluation of binary-centroid pupil localisation

Measurement model, analysis scripts, input data and Monte Carlo results for a JCGM 101
uncertainty evaluation of binary-centroid pupil localisation in video-oculography.

## Contents

```
code/python/     measurement model (iris_model.py) and analysis scripts
code/matlab/     the original MATLAB/Octave model the Python code was ported from
results/         output of the analysis scripts (JSON)
figures/static/  model schematics, which no script draws
```

## Data

| Data | Where | Notes |
|---|---|---|
| Inputs of the measurement model | constants in `code/python/iris_model.py` | Pupil-shape population after Wyatt (1995) |
| Real-image verification | LPW dataset | Not redistributed; see below |
| Monte Carlo and analysis results | `results/` | One JSON per script |

## The model in brief

`iris_model.py` simulates one trial at a time:

1. A 5 mm pupil, with harmonic shape distortions drawn from the Wyatt population, is
   rotated through the gaze angle α and projected.
2. It is rendered on a fine grid and area-averaged onto sensor pixels of pitch
   r = 1/DensCCD (mm per line).
3. Gaussian noise of standard deviation σ_eff is added.
4. The image is thresholded at t = 0.5, and the centroid of the foreground pixels is
   taken.

The error is the centroid minus the centre of the noise-free projected boundary.

The coverage-fraction bias model predicts the mean of that error from the
coverage fraction c_k of every pixel of the analysed region:

```
E{x} = Σ_k x_k p_k / Σ_k p_k,    p_k = Φ((c_k − t) / σ_eff)
```

For c_k ∈ {0, 1} it reduces to the two-class model of Van Rheeden and Jones (1988).
`vrcheck.py` states both models.

## Analysis → script → output

All commands run from `code/python/`; outputs go to `results/`.

| Analysis | Command | Output |
|---|---|---|
| Shape population against Wyatt (1995) | `python iris_model.py` | console |
| Design-space sweeps on three rendering grids (noise, pitch and gaze; budget point; fine pitch) | `python run_converged.py --oversample 1`, `3`, `5` | `converged_os{1,3,5}.json` |
| Grid convergence (observed order, GCI, u_num) | `python gci.py` | `gci.json` |
| Fine-pitch sweep on the base grid below r = 0.25 mm/line | `python fine_pitch_sweep.py` | `fine_pitch_base.json` |
| Boundary vertices and point-in-shape test | `python boundary_vertices.py` | `boundary_vertices.json` |
| Rayleigh test, variance shares, lever arm, imaging geometry | `python stats_and_physics.py` | `stats_physics.json` |
| Model-incompleteness runs (shot noise, contrast, optics, occlusion, diameter) | `python run_umodel.py` | `umodel.json` |
| Bias correction, four estimators, variable noise, processing cost | `python alternatives.py` | `alternatives.json` |
| Uncertainty budget and model-incompleteness components | `python budget.py` | `budget.json` |
| Coverage-fraction against two-class bias model (four runs, see the script header) | `python vrcheck.py` | `vrcheck.json` |
| Bias against the offset of the analysed window | `python window_offset.py` | `window_offset.json` |
| On-line estimation of the bias-model inputs | `python estimator.py` | `estimator.json` |
| Detection-threshold sensitivity | `python threshold_sweep.py` | `threshold.json` |
| Bias at fixed against randomised subpixel phase | `python phase_test.py` | `phase_test.json` |
| Closed-form surrogate of u(σ_eff, r) | `python surrogate.py` | `surrogate.json` |
| Adaptive JCGM 101 procedure, coverage intervals | `python run_jcgm101.py --all` | `jcgm101.json` |
| Cramér–Rao bound | `python crlb.py` | `crlb.json` |
| Estimators under illumination gradient and catch-light | `python robustness.py` | `robustness.json` |
| Fine-pitch law and tremor-detection requirements | `python tremor.py` | `tremor.json` |
| Precision on real pupil geometry | `python lpw_precision.py <LPW> 3 5`, `... 3 25` | `lpw_precision_{5,25}px.json` |
| Coverage-fraction model on real coverage maps | `python lpw_coverage_bias.py <LPW> 3 5`, `... 3 25`; `python lpw_summary.py` | `lpw_coverage_bias_{5,25}px.json`, `lpw_coverage_bias_summary.json` |
| Whiteness of the localisation error | `python lpw_autocorr.py <LPW> 200 5` | `lpw_autocorr.json` |
| Ratios, ranges and directions combining several results | `python derived.py` | `derived.json` |
| Bias vectors over noise and gaze; all figures | `python make_figures.py` | `bias_vectors.json`, `figures/*.png` |
| Example pupil shapes | `python make_pupil_shapes.py` | `figures/pupil_shapes_a-f.png` |

The sweeps of `run_converged.py` are stored under the keys `noise/`, `pitch/`,
`budget/`, `fine/` and `finenoise/`; `python run_converged.py --help` lists them.

## Reproducibility

Every Monte Carlo script spawns its seeds from a fixed `numpy.random.SeedSequence`, so
a re-run reproduces the stored JSON. The seed streams of `run_converged.py`,
`run_umodel.py` and `alternatives.py` are identical by construction. That makes their
comparisons paired and exact, so change the seed arithmetic in all three or in none.

Order of execution:

1. `run_converged.py` at the three grid levels, then `fine_pitch_sweep.py`.
2. The independent Monte Carlo scripts above, in any order.
3. The scripts that only read `results/`: `gci.py`, `budget.py`, `surrogate.py`,
   `tremor.py`, `derived.py`, `make_figures.py --draw`.

Run times on 12 cores:

| Script | Time |
|---|---|
| `run_converged.py`, one grid level | about 30 min |
| `run_umodel.py` | 15 min |
| `alternatives.py` | 10 min |
| `estimator.py` | 30 min |
| `crlb.py` | 20 min |
| `window_offset.py` | about 1 h |
| `phase_test.py` | a few minutes |
| `run_jcgm101.py --all` | about 85 min |

The LPW scripts take minutes; the read-only scripts take seconds.

## Requirements

- Python 3.10 or later with NumPy, SciPy and Matplotlib (`pip install -r requirements.txt`).
  The results were produced with Python 3.14, NumPy 2.4 and SciPy 1.17.
- `opencv-python` in addition for the LPW scripts.
- GNU Octave or MATLAB only for `code/matlab/`, which no result depends on.

## Third-party data and licensing

- LPW dataset (Tonsen et al., ETRA 2016; Bulling, DaRUS 2023, DOI
  [10.18419/darus-3237](https://doi.org/10.18419/darus-3237)), licensed under
  CC BY-NC-SA 4.0.
  - The dataset is not redistributed. Download it after accepting its data agreement;
    see `code/python/LPW_REPRODUCE.md`.
  - The repository includes quantities measured on the dataset: aggregate curves and,
    in `lpw_coverage_bias_{5,25}px.json`, per-frame derived values.
  - No image, crop, mask or annotated coordinate is included, and none can be
    reconstructed from these files.
  - These files are derived from LPW, so its licence and citation apply to them.
- The code and the simulation results are released under the MIT License (`LICENSE`).

## Funding

Centre for Priority Research Area Artificial Intelligence and Robotics of Warsaw
University of Technology, Excellence Initiative: Research University (IDUB) programme,
project no. 504/04496/1142/45.010202.

## Citing

See `CITATION.cff`. Cite the LPW dataset as well for anything derived from the
real-image verification.
