"""Pure count-based P3 validation; no model loading, inference or file access."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from pyproj import Geod


def ratio(numerator: float, denominator: float) -> float | None:
    """Return null for undefined metrics."""
    return numerator / denominator if denominator else None


def metrics(rows: list[dict]) -> dict:
    """Aggregate native valid confusion counts before computing micro scores."""
    tp, fp, fn, tn = (sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn', 'tn'))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, valid_pixels=tp+fp+fn+tn,
                IoU=ratio(tp, tp+fp+fn), F1=ratio(2*tp, 2*tp+fp+fn),
                Precision=ratio(tp, tp+fp), Recall=ratio(tp, tp+fn))


def mean_defined(values: list) -> float | None:
    """Exclude undefined values explicitly; caller also reports defined n."""
    kept = [v for v in values if v is not None]
    return float(np.mean(kept)) if kept else None


def geographic_row_areas(transform: Any, height: int, width: int) -> np.ndarray:
    """WGS84 ellipsoidal m2/pixel for north-up geographic native grids.

    Each row has constant area; do not treat degree squared as square metres.
    """
    if transform.b != 0 or transform.d != 0:
        raise ValueError('Rotated grid needs per-pixel geodesic area')
    geod = Geod(ellps='WGS84')
    areas = []
    for row in range(height):
        x0, y0 = transform * (0, row)
        x1, y1 = transform * (1, row+1)
        area, _ = geod.polygon_area_perimeter([x0, x1, x1, x0], [y0, y0, y1, y1])
        areas.append(abs(area))
    return np.broadcast_to(np.array(areas)[:, None], (height, width))


def scene_counts(city: str, image_id: str, group: str, prediction: np.ndarray,
                 label: np.ndarray, valid: np.ndarray, pixel_area_m2: np.ndarray) -> dict:
    """Count a supplied native binary prediction; this does not run inference."""
    if not (prediction.shape == label.shape == valid.shape == pixel_area_m2.shape):
        raise ValueError('Native grid shapes differ')
    if prediction.dtype != bool or valid.dtype != bool:
        raise ValueError('Prediction and validity must be boolean')
    if not valid.any() or not np.all(np.isfinite(pixel_area_m2)) or np.any(pixel_area_m2 <= 0):
        raise ValueError('Invalid support/area')
    p, y = prediction[valid], label[valid] > 0
    fp_mask = valid & prediction & ~(label > 0)
    return dict(city=city, image_id=image_id, spatial_group_id=group,
                tp=int(np.sum(p & y)), fp=int(np.sum(p & ~y)),
                fn=int(np.sum(~p & y)), tn=int(np.sum(~p & ~y)),
                fp_area_m2=float(pixel_area_m2[fp_mask].sum()),
                valid_area_m2=float(pixel_area_m2[valid].sum()))


def summarize(rows: list[dict]) -> dict:
    """One city's validation only; group-balanced primary score is predeclared."""
    if not rows or len({r['city'] for r in rows}) != 1:
        raise ValueError('Require exactly one nonempty city cohort')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate scenes')
    for row in rows:
        if any(row[k] < 0 or int(row[k]) != row[k] for k in ('tp', 'fp', 'fn', 'tn')):
            raise ValueError('Invalid confusion counts')
        if not sum(row[k] for k in ('tp', 'fp', 'fn', 'tn')):
            raise ValueError('Zero valid pixels')
    positive = [r for r in rows if r['tp']+r['fn'] > 0]
    empty = [r for r in rows if r['tp']+r['fn'] == 0]
    groups = defaultdict(list)
    for row in rows:
        groups[row['spatial_group_id']].append(row)
    group_reports, pos_scores, empty_rates = [], [], []
    for group, cohort in sorted(groups.items()):
        pos = [r for r in cohort if r['tp']+r['fn'] > 0]
        bg = [r for r in cohort if r['tp']+r['fn'] == 0]
        score = mean_defined([metrics([r])['IoU'] for r in pos])
        rate = ratio(sum(r['fp'] for r in bg), sum(r['fp']+r['tn'] for r in bg))
        if score is not None:
            pos_scores.append(score)
        if rate is not None:
            empty_rates.append(rate)
        group_reports.append(dict(spatial_group_id=group, scenes=len(cohort),
                                  positive_scenes=len(pos), empty_scenes=len(bg), micro=metrics(cohort),
                                  positive_macro_iou=score, empty_fp_pixel_fraction=rate))
    macro = {}
    for key in ('IoU', 'F1', 'Precision', 'Recall'):
        values = [metrics([r])[key] for r in positive]
        macro[key] = mean_defined(values)
        macro[key+'_defined_scenes'] = sum(v is not None for v in values)
    pscore, erate = mean_defined(pos_scores), mean_defined(empty_rates)
    if pscore is None:
        raise ValueError('STOP: no nonempty validation groups')
    score = pscore if erate is None else (pscore + 1-erate)/2
    return dict(city=rows[0]['city'], scenes=len(rows), valid_spatial_groups=len(groups),
                nonempty_scenes=len(positive), nonempty_spatial_groups=len(pos_scores),
                empty_scenes=len(empty), empty_spatial_groups=len(empty_rates),
                nonempty_macro=macro, micro_all=metrics(rows), spatial_groups=group_reports,
                empty=dict(status='available' if empty else 'NA_NO_EMPTY_VALIDATION_SCENES',
                           fp_pixels=sum(r['fp'] for r in empty) if empty else None,
                           fp_area_m2=sum(r['fp_area_m2'] for r in empty) if empty else None,
                           fp_pixel_fraction=ratio(sum(r['fp'] for r in empty), sum(r['fp']+r['tn'] for r in empty)),
                           fp_area_fraction=ratio(sum(r['fp_area_m2'] for r in empty), sum(r['valid_area_m2'] for r in empty)),
                           mean_fp_area_m2_per_scene=ratio(sum(r['fp_area_m2'] for r in empty), len(empty))),
                positive_group_balanced_iou=pscore, empty_group_balanced_fp_fraction=erate,
                checkpoint_score=score)


def select_checkpoint(epoch_summaries: list[dict]) -> int:
    """Highest score rounded to 12 decimals, ties earliest epoch; never test data."""
    if not epoch_summaries or len({r['epoch'] for r in epoch_summaries}) != len(epoch_summaries):
        raise ValueError('Missing/duplicate epoch summaries')
    if any(not np.isfinite(r['checkpoint_score']) for r in epoch_summaries):
        raise ValueError('STOP: nonfinite validation score')
    return max(epoch_summaries, key=lambda r: (round(r['checkpoint_score'], 12), -r['epoch']))['epoch']
