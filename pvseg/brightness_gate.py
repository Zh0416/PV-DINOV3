"""Brightness-conditioned attenuation of a cached greenhouse penalty.

Uses original display RGB, never encoder-normalized tensors. Calibration masks
fit the brightness reference; query masks are not part of the inference API.
This is apparent sRGB brightness, not radiometrically calibrated reflectance.
"""

import math

import numpy as np
import torch
from torch.nn import functional as F


def brightness_grid(rgb, grid_size=(56, 56)):
    """Return area-mean brightness and image-wise robust exposure normalization.

    RGB is a H,W,3 uint8 array. The 10/90 percentiles are calculated on the
    original brightness pixels, with a one-DN denominator floor. For 896x896
    and 56x56 this is exact 16x16 pooling, aligned to the DINO patch grid.
    Normalized values are deliberately not clipped before applying the gate.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected original H,W,3 uint8 RGB')
    if len(grid_size) != 2 or any(v <= 0 for v in grid_size):
        raise ValueError('Expected positive grid dimensions')
    if any(n % g for n, g in zip(rgb.shape[:2], grid_size)):
        raise ValueError('Image dimensions must divide exactly into the patch grid')
    values = rgb.astype(np.float32)
    y = .2126 * values[..., 0] + .7152 * values[..., 1] + .0722 * values[..., 2]
    lo, hi = np.quantile(y, [.1, .9])
    scale = max(float(hi - lo), 1.)
    raw = F.adaptive_avg_pool2d(torch.from_numpy(y)[None, None], grid_size)[0, 0].numpy()
    normalized = (raw - float(lo)) / scale
    return raw, normalized, dict(scene_p10=float(lo), scene_p90=float(hi), scale_dn=scale)


def weighted_quantile(values, weights, quantiles):
    values, weights = np.asarray(values).ravel(), np.asarray(weights).ravel()
    quantiles = np.asarray(quantiles)
    if (values.shape != weights.shape or not values.size
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0
            or not np.isfinite(quantiles).all() or (quantiles < 0).any() or (quantiles > 1).any()):
        raise ValueError('Invalid weighted quantile inputs')
    valid = weights > 0
    order = np.argsort(values[valid], kind='stable')
    v, w = values[valid][order], weights[valid][order]
    index = np.searchsorted(w.cumsum(), quantiles * w.sum(), side='left')
    return v[np.clip(index, 0, len(v) - 1)]


def fit_brightness_reference(normalized_grids, foreground_area_grids, quantiles=(.75, .95)):
    """Equal weight per positive calibration image, PV area weight within it.

    Each foreground-area grid contains the mean of the binary PV mask in each
    patch. Partial foreground patches are retained, including small targets.
    No negative image contributes to the PV reference. The quantile pair is
    a fixed design choice for the pilot, not selected on query performance.
    """
    if (len(normalized_grids) != len(foreground_area_grids) or not normalized_grids
            or len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1):
        raise ValueError('Invalid calibration/reference configuration')
    values, weights, image_count = [], [], 0
    for z, area in zip(normalized_grids, foreground_area_grids):
        z, area = np.asarray(z), np.asarray(area)
        if (z.shape != area.shape or not np.isfinite(z).all() or not np.isfinite(area).all()
                or (area < 0).any() or (area > 1).any()):
            raise ValueError('Invalid calibration grid/foreground weights')
        if area.sum() > 0:
            values.append(z.ravel())
            weights.append((area.astype(np.float64) / area.sum()).ravel())
            image_count += 1
    if not image_count:
        raise ValueError('Brightness calibration requires positive foreground')
    lo, hi = weighted_quantile(np.concatenate(values), np.concatenate(weights), quantiles)
    # Fixed numerical fallback for a degenerate/constant reference.
    hi = max(float(hi), float(lo) + .001)
    return dict(lower=float(lo), upper=hi, quantiles=list(quantiles),
                positive_calibration_images=image_count,
                weighting='Equal image weight; foreground patch area within each positive image')


def brightness_gate(normalized_grid, reference):
    lo, hi = reference['lower'], reference['upper']
    z = np.asarray(normalized_grid, dtype=np.float32)
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo and np.isfinite(z).all()):
        raise ValueError('Invalid brightness/reference values')
    return np.clip((z - lo) / (hi - lo), 0, 1)


def apply_brightness_gate(pv_semantic, semantic_penalty, gate):
    """Apply the gate before interpolation, preserving the A3 branch weights.

    All three arrays have the same patch grid. Gate=0 recovers the original
    semantic response; gate=1 recovers the preceding greenhouse module.
    """
    pv, penalty, gate = [np.asarray(v, dtype=np.float32) for v in
                         (pv_semantic, semantic_penalty, gate)]
    if (pv.shape != penalty.shape or pv.shape != gate.shape
            or not all(np.isfinite(v).all() for v in (pv, penalty, gate))
            or (penalty < 0).any() or (gate < 0).any() or (gate > 1).any()):
        raise ValueError('Invalid semantic penalty/gate grids')
    effective = penalty * gate
    return np.clip(pv - effective, -1, 1), effective
