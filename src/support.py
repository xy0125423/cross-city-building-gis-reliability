"""Original multi-product conservative Support generation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from src.p1_spatial import classify_footprint, RULE
ROOT = Path('.')


def observation_mask(paths: dict, output: Path) -> tuple[np.ndarray, dict, dict]:
    """Use co-registered products and native-MUL evidence, preserving source files."""
    with rasterio.open(ROOT / paths["RGB-PanSharpen"]) as rgb:
        rgb_values = rgb.read()
        rgb_zero = np.all(rgb_values == 0, axis=0)
        valid = np.all(rgb.read_masks() > 0, axis=0)
        profile = rgb.profile.copy()
        transform_rgb, crs_rgb, shape_rgb = rgb.transform, rgb.crs, rgb.shape
        metadata = {"nodata": str(rgb.nodata),
                    "rgb_mask_flags": str(rgb.mask_flag_enums),
                    "rgb_min": int(rgb_values.min()), "rgb_max": int(rgb_values.max()),
                    "rgb_zero_fraction": float(rgb_zero.mean()),
                    "rgb_internal_mask_invalid_pixels": int(np.sum(~valid))}
    products = []
    for name in ("PAN", "MUL-PanSharpen"):
        with rasterio.open(ROOT / paths[name]) as dataset:
            if dataset.crs != crs_rgb or dataset.transform != transform_rgb or dataset.shape != shape_rgb:
                raise ValueError(f"High-resolution product grid differs: {paths[name]}")
            products.append(np.all(dataset.read() == 0, axis=0))
            valid &= np.all(dataset.read_masks() > 0, axis=0)
            metadata[f"{name}_mask_flags"] = str(dataset.mask_flag_enums)
    with rasterio.open(ROOT / paths["MUL"]) as native:
        native_zero = np.all(native.read() == 0, axis=0).astype(np.uint8)
        native_zero[np.any(native.read_masks() == 0, axis=0)] = 255
        native_on_rgb = np.full(shape_rgb, 255, dtype=np.uint8)
        reproject(native_zero, native_on_rgb, src_transform=native.transform, src_crs=native.crs,
                  dst_transform=transform_rgb, dst_crs=crs_rgb, src_nodata=255, dst_nodata=255,
                  resampling=Resampling.nearest)
        metadata["native_MUL_zero_fraction"] = float((native_zero == 1).mean())
        metadata["native_MUL_mask_flags"] = str(native.mask_flag_enums)
    classified, evidence = classify_footprint(rgb_zero, products[0], products[1], native_on_rgb, valid)
    metadata.update(evidence)
    for code, name in RULE["classes"].items():
        metadata[f"{name}_pixels"] = int(np.sum(classified == int(code)))
        metadata[f"{name}_fraction"] = float(np.mean(classified == int(code)))
    output.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="uint8", nodata=None, compress="deflate", predictor=1)
    with rasterio.open(output, "w", **profile) as dataset:
        dataset.write(classified, 1)
        dataset.update_tags(rule=RULE["version"], classes=json.dumps(RULE["classes"]))
    return classified, metadata, profile
