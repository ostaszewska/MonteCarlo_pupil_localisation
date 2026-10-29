# -*- coding: utf-8 -*-
"""Segmentation and sampling helpers shared by the LPW scripts.

Used by `lpw_precision.py` and `lpw_autocorr.py`. Importing this module has no side
effects; it neither reads the dataset nor writes anything. See LPW_REPRODUCE.md.
"""
import numpy as np
from scipy import ndimage as ndi

try:
    import cv2
except ImportError:                                   # pragma: no cover
    raise SystemExit('opencv-python is required: pip install opencv-python')

WIN = 90          # half-size of the crop window around the annotated centre [hi-res px]


def segment_pupil_local(g, gc_xy, win=WIN):
    """Dark-pupil mask in a window centred on the annotated centre gc_xy = (x, y).

    Returns a boolean mask in full-frame coordinates, or None if no component
    survives the plausibility tests. The threshold percentile is relaxed in steps
    because illumination varies widely across the LPW recordings; holes are filled
    to close corneal catch-lights.
    """
    H, W = g.shape
    cx, cy = gc_xy
    x0, x1 = max(int(cx - win), 0), min(int(cx + win), W)
    y0, y1 = max(int(cy - win), 0), min(int(cy + win), H)
    sub = g[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    lcx, lcy = cx - x0, cy - y0
    for pct in (8, 12, 16, 20, 25):
        thr = np.percentile(sub, pct)
        m = ndi.binary_fill_holes(sub <= thr)
        lab, n = ndi.label(m)
        if n == 0:
            continue
        gi = lab[int(round(lcy)), int(round(lcx))] \
            if 0 <= int(round(lcy)) < lab.shape[0] and 0 <= int(round(lcx)) < lab.shape[1] \
            else 0
        cand = [gi] if gi > 0 else list(range(1, n + 1))
        best = None
        for i in cand:
            comp = lab == i
            area = comp.sum()
            if area < 0.01 * sub.size or area > 0.7 * sub.size:
                continue
            ys, xs = np.nonzero(comp)
            if np.hypot(xs.mean() - lcx, ys.mean() - lcy) > 0.4 * win:
                continue
            if best is None or area > best[1]:
                best = (comp, area)
        if best is not None:
            full = np.zeros((H, W), bool)
            full[y0:y1, x0:x1] = best[0]
            return full
    return None


def block_downsample(mask, factor):
    """Average factor x factor blocks, the sensor-pixel model of iris_model."""
    H, W = mask.shape
    nh, nw = H // factor, W // factor
    m = mask[:nh * factor, :nw * factor].astype(float)
    return m.reshape(nh, factor, nw, factor).mean(axis=(1, 3))


def centroid_roi(binimg, c0, half):
    """Centre of mass of the foreground within +-half of c0, or None if empty."""
    H, W = binimg.shape
    x0, y0 = c0
    xa, xb = max(int(x0 - half), 0), min(int(x0 + half) + 1, W)
    ya, yb = max(int(y0 - half), 0), min(int(y0 + half) + 1, H)
    sub = binimg[ya:yb, xa:xb]
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        return None
    return np.array([xs.mean() + xa, ys.mean() + ya])


def read_annotation(txt):
    """One (x, y) per frame from an LPW .txt; unparsable lines become (0, 0)."""
    out = []
    with open(txt) as f:
        for line in f:
            parts = line.replace(',', ' ').split()
            if len(parts) >= 2:
                try:
                    out.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    out.append((0.0, 0.0))
    return out


def sample_frame(avi, idx):
    """One frame as greyscale in [0, 1], or None if it cannot be read."""
    cap = cv2.VideoCapture(avi)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.astype(float) / 255.0
