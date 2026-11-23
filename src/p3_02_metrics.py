"""Count-based paired test summaries and group bootstrap; no model or disk IO."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.p3_01_validation import metrics, ratio

SCORES = ('IoU', 'Precision', 'Recall', 'F1')
COUNTS = ('tp', 'fp', 'fn', 'tn')


def sufficient_stats(rows: list[dict]) -> np.ndarray:
    """Preserve count totals and all defined nonempty scene-score sums/counts."""
    values = [sum(r[k] for r in rows) for k in COUNTS]
    positive = [r for r in rows if r['tp'] + r['fn'] > 0]
    empty = [r for r in rows if r['tp'] + r['fn'] == 0]
    for key in SCORES:
        scores = [metrics([r])[key] for r in positive]
        defined = [x for x in scores if x is not None]
        values.extend((sum(defined), len(defined)))
    values.extend((len(rows), len(positive), len(empty),
                   sum(r['fp'] for r in empty),
                   sum(r['fp_area_m2'] for r in empty),
                   sum(r['fp'] + r['tn'] for r in empty),
                   sum(r['valid_area_m2'] for r in empty)))
    return np.asarray(values, dtype=np.float64)


def values_from_stats(values: np.ndarray) -> dict[str, float | None]:
    """Recompute nonlinear scores after aggregation, including resampled groups."""
    tp, fp, fn, tn = values[:4]
    output = dict(tp=float(tp), fp=float(fp), fn=float(fn), tn=float(tn),
                  micro_IoU=ratio(tp, tp + fp + fn),
                  micro_Precision=ratio(tp, tp + fp), micro_Recall=ratio(tp, tp + fn),
                  micro_F1=ratio(2 * tp, 2 * tp + fp + fn))
    for index, key in enumerate(SCORES):
        output[f'nonempty_macro_{key}'] = ratio(values[4 + 2 * index], values[5 + 2 * index])
    nempty, efp, earea, epix, evarea = values[14:19]
    output.update(empty_fp_pixels=float(efp) if nempty else None,
                  empty_fp_area_m2=float(earea) if nempty else None,
                  empty_fp_pixel_fraction=ratio(efp, epix),
                  empty_fp_area_fraction=ratio(earea, evarea),
                  empty_mean_fp_area_m2_per_scene=ratio(earea, nempty))
    return {k: (float(v) if v is not None else None) for k, v in output.items()}


def summarize_test(rows: list[dict]) -> dict[str, Any]:
    """Summarize a complete single-target cohort without checkpoint selection."""
    if not rows or len({r['city'] for r in rows}) != 1:
        raise ValueError('Require a single nonempty target cohort')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate test scene')
    if any(sum(r[k] for k in COUNTS) <= 0 for r in rows):
        raise ValueError('Empty evaluation support')
    stats = sufficient_stats(rows)
    values = values_from_stats(stats)
    groups = defaultdict(list)
    for row in rows:
        groups[row['spatial_group_id']].append(row)
    nonempty_macro = {}
    for index, key in enumerate(SCORES):
        nonempty_macro[key] = values[f'nonempty_macro_{key}']
        nonempty_macro[f'{key}_defined_scenes'] = int(stats[5 + 2 * index])
    return dict(
        target_city=rows[0]['city'], scenes=len(rows), spatial_group_count=len(groups),
        nonempty_scenes=int(stats[13]), empty_scenes=int(stats[14]),
        nonempty_groups=sum(any(r['tp'] + r['fn'] > 0 for r in g) for g in groups.values()),
        empty_groups=sum(any(r['tp'] + r['fn'] == 0 for r in g) for g in groups.values()),
        micro=metrics(rows), nonempty_macro=nonempty_macro,
        empty=dict(status='available' if stats[14] else 'NA_NO_EMPTY_TEST_SCENES',
                   **{k.removeprefix('empty_'): v for k, v in values.items() if k.startswith('empty_')}),
        spatial_groups=[dict(spatial_group_id=group,
                             **summarize_group(cohort)) for group, cohort in sorted(groups.items())],
        paired_metric_values=values,
    )


def summarize_group(rows: list[dict]) -> dict:
    """Return a group's raw counts and stratum scores without recursion."""
    stats = sufficient_stats(rows)
    return dict(scenes=len(rows), nonempty_scenes=int(stats[13]), empty_scenes=int(stats[14]),
                valid_area_m2=sum(r['valid_area_m2'] for r in rows),
                micro=metrics(rows), **values_from_stats(stats))


def pair_records(within: list[dict], cross: list[dict]) -> list[tuple[dict, dict]]:
    """Require identical target membership, GT, support and group assignments."""
    a, b = ({r['image_id']: r for r in cohort} for cohort in (within, cross))
    if len(a) != len(within) or len(b) != len(cross) or not a or a.keys() != b.keys():
        raise ValueError('Paired target scene mismatch')
    paired = [(a[k], b[k]) for k in sorted(a)]
    for first, second in paired:
        if (first['city'] != second['city'] or
                first['spatial_group_id'] != second['spatial_group_id'] or
                first['tp'] + first['fn'] != second['tp'] + second['fn'] or
                sum(first[k] for k in COUNTS) != sum(second[k] for k in COUNTS)):
            raise ValueError('Paired GT/support/group mismatch')
    return paired


def paired_comparison(within: list[dict], cross: list[dict], *,
                      repeats: int = 2000, seed: int = 20261115) -> dict:
    """Use shared cluster draws for within-minus-cross descriptive intervals."""
    paired = pair_records(within, cross)
    groups = sorted({r['spatial_group_id'] for r in within})
    first = np.stack([sufficient_stats([a for a, _ in paired if a['spatial_group_id'] == g])
                      for g in groups])
    second = np.stack([sufficient_stats([b for _, b in paired if b['spatial_group_id'] == g])
                       for g in groups])
    va, vb = values_from_stats(first.sum(axis=0)), values_from_stats(second.sum(axis=0))
    samples = {key: [] for key in va}
    rng = np.random.Generator(np.random.PCG64(seed))
    draw_hash = __import__('hashlib').sha256()
    for _ in range(repeats):
        draw = rng.integers(0, len(groups), size=len(groups))
        draw_hash.update(draw.astype('<i8').tobytes())
        weights = np.bincount(draw, minlength=len(groups))
        sa, sb = values_from_stats(weights @ first), values_from_stats(weights @ second)
        for key in samples:
            if sa[key] is not None and sb[key] is not None:
                samples[key].append(sa[key] - sb[key])
    results = {}
    for key, observations in samples.items():
        difference = va[key] - vb[key] if va[key] is not None and vb[key] is not None else None
        interval = np.percentile(observations, [2.5, 97.5], method='linear').tolist() if observations else [None, None]
        results[key] = dict(within=va[key], cross=vb[key], within_minus_cross=difference,
                            lower_95=interval[0], upper_95=interval[1],
                            valid_repeats=len(observations), invalid_repeats=repeats - len(observations))
    scene_differences = []
    for a, b in paired:
        ma, mb = metrics([a]), metrics([b])
        differences = {key: (ma[key] - mb[key] if ma[key] is not None and mb[key] is not None else None)
                       for key in SCORES}
        scene_differences.append(dict(image_id=a['image_id'], spatial_group_id=a['spatial_group_id'],
                                      nonempty=a['tp'] + a['fn'] > 0, within=ma, cross=mb,
                                      within_minus_cross=differences,
                                      fp_difference=a['fp'] - b['fp'], fn_difference=a['fn'] - b['fn']))
    common = [r['within_minus_cross']['Precision'] for r in scene_differences
              if r['nonempty'] and r['within_minus_cross']['Precision'] is not None]
    return dict(target_city=within[0]['city'], scenes=len(paired), groups=len(groups),
                direction='within_minus_cross', seed=seed, repeats=repeats,
                bootstrap='paired spatial-group percentile, descriptive, conditional on fixed models',
                draw_sha256=draw_hash.hexdigest(), metrics=results,
                common_defined_positive_precision=dict(scenes=len(common),
                    mean_paired_difference=float(np.mean(common)) if common else None),
                per_scene=scene_differences)
