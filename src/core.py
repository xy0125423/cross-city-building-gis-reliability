"""Frozen FIX03 numerical core; extracted without numerical changes."""
from __future__ import annotations
from typing import Any, Callable
import random
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import transform
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from src.p3_01_validation import geographic_row_areas, metrics, scene_counts, summarize
EXCLUDED_ID = 'AOI_5_Khartoum_img894'
VERSION = 'P3-01-FIX03'
from pathlib import Path
import hashlib
_CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
LOCK = hashlib.sha256((_CONFIGS / 'P3_01_FIX03_effective_manifest.csv').read_bytes()).hexdigest()
ORIGINAL_LOCK = hashlib.sha256((_CONFIGS / 'P3_00_split_manifest.csv').read_bytes()).hexdigest()


def configure_determinism(config: dict) -> None:
    """Reset all three generators independently for every source run."""
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(True)


def rasterize_scene(image: Any, support: Any, payload: dict) -> dict:
    """Shared real/MemoryFile raster interface with exact native grid checks."""
    if image.count != 3 or support.count != 1:
        raise ValueError('STOP: RGB/Support channel mismatch')
    if (image.shape != support.shape or image.transform != support.transform
            or image.crs != support.crs or image.crs != CRS.from_epsg(4326)):
        raise ValueError('STOP: grid/CRS mismatch')
    mask = support.read(1)
    if not set(np.unique(mask)).issubset({0, 1, 2, 3}) or support.nodata == 1:
        raise ValueError('STOP: invalid Support codes/NoData')
    valid = mask == 1
    data = image.read().astype(np.float32)
    if (not valid.any() or not np.isfinite(data).all()
            or np.any(valid & np.any(image.read_masks() == 0, axis=0))
            or np.any(valid & (support.read_masks(1) == 0))):
        raise ValueError('STOP: invalid RGB/Support/NoData')
    label_crs = payload.get('crs', {}).get('properties', {}).get('name', 'OGC:CRS84')
    converter = Transformer.from_crs(label_crs, image.crs, always_xy=True)
    geometries = [transform(converter.transform, shape(f['geometry']))
                  for f in payload['features']]
    if any(g.is_empty or not g.is_valid for g in geometries):
        raise ValueError('STOP: invalid label geometry')
    label = rasterize([(g, 1) for g in geometries], out_shape=image.shape,
                      transform=image.transform, dtype='uint8', all_touched=False
                      ) if geometries else np.zeros(image.shape, dtype='uint8')
    area = geographic_row_areas(image.transform, *image.shape)
    return dict(data=data, label=label, valid=valid, area=area,
                transform=tuple(image.transform), crs=str(image.crs),
                label_crs=label_crs, assumed_label_crs='crs' not in payload)


def require_rows(rows: list[dict], city: str, split: str) -> None:
    """Reject city mixing, wrong splits, missing or duplicate scenes."""
    if not rows or len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('STOP: empty or duplicate cohort')
    if any(r['image_id'] == EXCLUDED_ID or r['city'] != city or r['split'] != split
           or r.get('effective_split', split) != split for r in rows):
        raise PermissionError('STOP: source/split isolation failed')


def normalization(rows: list[dict], city: str, loader: Callable) -> dict:
    """Compute P1/P99 from source TRAIN valid native row-major [::16] only."""
    require_rows(rows, city, 'train')
    ordered = sorted(rows, key=lambda row: row['image_id'])
    chunks = []
    for row in ordered:
        sample = loader(row)
        chunks.append(sample['data'][:, sample['valid']][:, ::16].astype(np.float32))
    pixels = np.concatenate(chunks, axis=1)
    lo, hi = np.percentile(pixels, [1, 99], axis=1, method='linear')
    if not np.all(np.isfinite(lo)) or not np.all(hi > lo):
        raise ValueError('STOP: degenerate normalization')
    return dict(source_city=city, split='train', scene_ids=[r['image_id'] for r in ordered],
                manifest_sha256=LOCK, manifest_version=VERSION, original_manifest_sha256=ORIGINAL_LOCK, band_p1=lo.tolist(), band_p99=hi.tolist(),
                sampled_pixels_per_band=pixels.shape[1], stride=16,
                scene_order='image_id_lexicographic', percentile_method='linear')


class SourceDataset(Dataset):
    """On-demand source-only samples, with native targets retained for validation."""

    def __init__(self, rows: list[dict], city: str, split: str,
                 loader: Callable, norm: dict) -> None:
        require_rows(rows, city, split)
        if split not in ('train', 'validation'):
            raise PermissionError('STOP: this dataset cannot evaluate tests')
        if norm['source_city'] != city or norm['split'] != 'train':
            raise PermissionError('STOP: normalization must come from source train')
        self.rows = sorted(rows, key=lambda row: row['image_id'])
        self.city, self.split, self.loader, self.norm = city, split, loader, norm

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        native = self.loader(row)
        data, label, valid = native['data'], native['label'], native['valid']
        if data.shape != (3, *label.shape) or valid.shape != label.shape:
            raise ValueError('STOP: sample dimensions')
        if not valid.any() or not np.isfinite(data).all():
            raise ValueError('STOP: empty support or nonfinite sample')
        if bool(label[valid].any()) != (int(row['label_empty']) == 0):
            raise ValueError(f'STOP: {self.split} effective/vector stratum mismatch: {row["image_id"]}')
        x = F.interpolate(torch.from_numpy(data.astype(np.float32))[None], size=(128, 128),
                          mode='bilinear', align_corners=False)[0]
        y = F.interpolate(torch.from_numpy(label.astype(np.float32))[None, None],
                          size=(128, 128), mode='nearest')[0]
        v = F.adaptive_avg_pool2d(torch.from_numpy(valid.astype(np.float32))[None, None],
                                 (128, 128))[0] >= 1-1e-6
        if not v.any():
            raise ValueError('STOP: zero resized valid pixels')
        lo = torch.tensor(self.norm['band_p1'], dtype=torch.float32)[:, None, None]
        hi = torch.tensor(self.norm['band_p99'], dtype=torch.float32)[:, None, None]
        x = torch.clamp((x-lo) / torch.clamp(hi-lo, min=1), 0, 1)
        x[:, ~v[0]] = 0
        return dict(x=x, y=y, v=v, row=row, native=native)


def collate_training(batch: list[dict]) -> dict:
    """Stack tensors only; no accidental native-grid collation or augmentation."""
    return {key: torch.stack([sample[key] for sample in batch]) for key in ('x', 'y', 'v')}


def objective(p: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Exact frozen probability BCE + batch-global masked soft Dice."""
    valid = v.float()
    if not bool(valid.sum() > 0):
        raise ValueError('STOP: zero loss support')
    bce = (F.binary_cross_entropy(p.clamp(1e-6, 1-1e-6), y, reduction='none') * valid).sum()/valid.sum()
    dice = 1-(2*(p*y*valid).sum()+1)/((p*valid).sum()+(y*valid).sum()+1)
    return bce+dice


def upsample_probability(p: torch.Tensor, native_shape: tuple) -> np.ndarray:
    """Interpolate sigmoid probabilities before applying any threshold."""
    native = F.interpolate(p, size=native_shape, mode='bilinear', align_corners=False)
    result = native[0, 0].detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError('STOP: nonfinite native probability')
    return result


def validate(model: torch.nn.Module, dataset: SourceDataset) -> dict:
    """Complete one-city native validation; no test dataset is accepted."""
    if dataset.split != 'validation':
        raise PermissionError('STOP: validation requires source validation split')
    model.eval()
    total, pixels, details = 0., 0, []
    with torch.no_grad():
        for sample in dataset:
            p = model(sample['x'][None])
            loss = objective(p, sample['y'][None], sample['v'][None])
            if not torch.isfinite(loss):
                raise ValueError('STOP: nonfinite validation loss')
            n = int(sample['v'].sum())
            total += float(loss)*n
            pixels += n
            native, row = sample['native'], sample['row']
            probability = upsample_probability(p, native['label'].shape)
            detail = scene_counts(row['city'], row['image_id'], row['spatial_group_id'],
                                  probability >= .5, native['label'], native['valid'], native['area'])
            detail.update(label_crs=native.get('label_crs'),
                          assumed_label_crs=native.get('assumed_label_crs', False),
                          transform=native.get('transform'), crs=native.get('crs'),
                          per_scene_metrics=metrics([detail]))
            details.append(detail)
    expected_ids = {r['image_id'] for r in dataset.rows}
    if {r['image_id'] for r in details} != expected_ids or len(details) != len(expected_ids):
        raise ValueError('STOP: incomplete validation')
    result = summarize(details)
    if len(result['spatial_groups']) != len({r['spatial_group_id'] for r in dataset.rows}):
        raise ValueError('STOP: missing validation groups')
    result.update(validation_loss=total/pixels, loss_valid_pixels_128=pixels,
                  per_scene=details)
    return result
