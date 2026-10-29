LPW real-image verification: how to reproduce
=============================================

Three scripts use real eye images from the Labelled Pupils in the Wild (LPW) dataset
(Tonsen et al., ETRA 2016). The dataset is NOT redistributed with this repository;
download it yourself after accepting the LPW data agreement.

1. DOWNLOAD (requires accepting the LPW data agreement)
   - DaRUS, DOI 10.18419/darus-3237:
       https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3237
   Unzip so the layout is:  LPW/<subject>/<video>.avi  +  LPW/<subject>/<video>.txt

2. INSTALL the one extra dependency:
       python -m pip install opencv-python

3. RUN, from code/python/. The second argument is the number of frames per video
   (the deposited results use 3 frames in one video per participant, 66 frames in
   all); the third is the pupil diameter in sensor pixels, which fixes the pitch
   because the pupil is 5 mm: 5 px is r = 1.00 mm/line, 25 px is r = 0.20 mm/line.

   Precision on real pupil geometry:
       python lpw_precision.py <path-to-LPW> 3 5
       python lpw_precision.py <path-to-LPW> 3 25
   -> results/lpw_precision_{5,25}px.json

   Coverage-fraction bias model on the coverage maps of real pupils:
       python lpw_coverage_bias.py <path-to-LPW> 3 5
       python lpw_coverage_bias.py <path-to-LPW> 3 25
       python lpw_summary.py
   -> results/lpw_coverage_bias_{5,25}px.json, results/lpw_coverage_bias_summary.json

   Whiteness of the localisation error:
       python lpw_autocorr.py <path-to-LPW> 200 5
   -> results/lpw_autocorr.json

   lpw_precision.py checkpoints per video in .lpw_cache/ at the repository root, so an
   interrupted run resumes; delete that folder to start afresh.

Notes
-----
* Segmentation is anchored on the LPW annotation of the pupil centre: a dark-region
  threshold with hole filling inside a window centred on it.
* lpw_precision.py also writes diagnostic figures (lpw_frames_*.png, lpw_curves_*.png)
  to .lpw_cache/. The frame montage shows real eye frames; .lpw_cache/ is excluded by
  .gitignore, and the montage must not be published without the dataset's
  CC BY-NC-SA 4.0 credit.
* The repository deposits numbers measured on LPW, never imagery or annotated
  coordinates; see the licensing section of the top-level README.
