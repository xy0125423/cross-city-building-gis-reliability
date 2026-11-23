"""Frozen native-grid inference helpers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import rasterio
import torch
from torch.nn import functional as F


def preprocess(native: dict, norm: dict) -> torch.Tensor:
    """Apply the frozen source preprocessing without estimating target statistics."""
    data, valid = native['data'], native['valid']
    x = F.interpolate(torch.from_numpy(data.astype(np.float32))[None], size=(128, 128),
                      mode='bilinear', align_corners=False)[0]
    support = F.adaptive_avg_pool2d(torch.from_numpy(valid.astype(np.float32))[None, None],
                                  (128, 128))[0] >= 1 - 1e-6
    if not support.any():
        raise ValueError('STOP: no valid model-input pixels')
    lo = torch.tensor(norm['band_p1'], dtype=torch.float32)[:, None, None]
    hi = torch.tensor(norm['band_p99'], dtype=torch.float32)[:, None, None]
    x = torch.clamp((x - lo) / torch.clamp(hi - lo, min=1), 0, 1)
    x[:, ~support[0]] = 0
    return x[None]


def error_classes(prediction: np.ndarray, label: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Encode TP1/FP2/FN3/TN4 with invalid pixels explicitly set to 255."""
    result = np.full(label.shape, 255, dtype=np.uint8)
    truth = label > 0
    result[valid & prediction & truth] = 1
    result[valid & prediction & ~truth] = 2
    result[valid & ~prediction & truth] = 3
    result[valid & ~prediction & ~truth] = 4
    return result


def export_raster(path: Path, values: np.ndarray, native: dict, *, nodata: int,
                  kind: str, tags: dict) -> None:
    """Write a new compressed, tiled, georeferenced single-band result."""
    if path.exists():
        raise FileExistsError('Preserve existing raster: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', driver='GTiff', width=values.shape[1], height=values.shape[0],
                       count=1, dtype=values.dtype, crs=native['crs'],
                       transform=rasterio.Affine(*native['transform'][:6]), nodata=nodata,
                       compress='deflate', predictor=3 if values.dtype == np.float32 else 2,
                       tiled=True, blockxsize=256, blockysize=256, BIGTIFF='IF_SAFER') as dst:
        dst.write(values, 1)
        dst.set_band_description(1, kind)
        dst.update_tags(**{k: str(v) for k, v in tags.items()}, kind=kind,
                        support_rule='native Support == 1', threshold='0.5',
                        invalid_region='NoData excluded from metrics')
        if kind == 'error_classes':
            dst.write_colormap(1, {1: (0, 158, 115, 255), 2: (213, 94, 0, 255),
                                   3: (204, 121, 167, 255), 4: (235, 235, 235, 255),
                                   255: (0, 0, 0, 0)})
            dst.update_tags(class_1='TP', class_2='FP', class_3='FN', class_4='TN')
