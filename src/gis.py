"""Frozen projected-area accounting helpers."""
from __future__ import annotations
import numpy as np
from pyproj import Transformer
from rasterio.features import shapes
import shapely
from shapely.geometry import shape
from shapely.ops import transform


def project(geo: dict, converter: Transformer):
    """Densify geographic edges before transforming; repair derived polygons only."""
    poly = shape(geo)
    if not poly.is_valid:
        poly = shapely.make_valid(poly)
    return transform(converter.transform, shapely.segmentize(poly, 0.00001))


def vectorize(array: np.ndarray, valid: np.ndarray, affine, converter, connectivity=4):
    """Yield projected native pixel regions, without resampling classifications."""
    for geo, value in shapes(array, mask=valid, transform=affine, connectivity=connectivity):
        yield project(geo, converter), int(value)


def derive(tp: float, fp: float, fn: float, tn: float, observed: float) -> dict:
    """Separate total wrong area, net bias, and observed-part coverage rates."""
    gt, pred = tp + fn, tp + fp
    total, net = fp + fn, fp - fn
    return dict(tp_m2=tp, fp_m2=fp, fn_m2=fn, tn_m2=tn,
                gt_m2=gt, pred_m2=pred, error_m2=total, net_m2=net, abs_bias_m2=abs(net),
                gt_cover=gt / observed, pred_cover=pred / observed,
                cover_bias=net / observed, abs_cover_error=abs(net) / observed,
                wrong_fraction=total / observed,
                relative_area_error=net / gt if gt > 1e-8 else None,
                cancellation=1 - abs(net) / total if total > 1e-8 else None,
                reference_nonempty=int(gt > 1e-8))


def summary(frame, city: str, route: str, scale: int, stratum='all') -> dict:
    """Summarize cell distributions and pooled areas on the same observed support."""
    area = frame.observed_m2.to_numpy()
    weights = area / area.sum()
    gt, pred = frame.gt_m2.sum(), frame.pred_m2.sum()
    return dict(city=city, route=route, scale_m=scale, stratum=stratum, cells=len(frame),
                observed_m2=float(area.sum()), gt_m2=float(gt), pred_m2=float(pred),
                fp_m2=float(frame.fp_m2.sum()), fn_m2=float(frame.fn_m2.sum()),
                total_wrong_m2=float(frame.error_m2.sum()), net_m2=float(frame.net_m2.sum()),
                sum_abs_cell_bias_m2=float(frame.abs_bias_m2.sum()),
                area_MAE_m2=float(frame.abs_bias_m2.mean()),
                area_weighted_area_MAE_m2=float(weights @ frame.abs_bias_m2),
                mean_signed_area_bias_m2=float(frame.net_m2.mean()),
                cover_MAE=float(frame.abs_cover_error.mean()),
                weighted_cover_MAE=float(weights @ frame.abs_cover_error),
                mean_cover_bias=float(frame.cover_bias.mean()),
                weighted_cover_bias=float((pred - gt) / area.sum()),
                pooled_gt_cover=float(gt / area.sum()), pooled_pred_cover=float(pred / area.sum()),
                pooled_relative_area_error=float((pred - gt) / gt) if gt > 1e-8 else None,
                spatial_cancellation=1 - float(frame.abs_bias_m2.sum() / frame.error_m2.sum()) if frame.error_m2.sum() > 1e-8 else None,
                support_fraction_min=float(frame.observed_fraction.min()),
                support_fraction_median=float(frame.observed_fraction.median()),
                support_fraction_max=float(frame.observed_fraction.max()),
                partial_cells=int(frame.partial_cell.sum()),
                cells_support_below_20pct=int((frame.observed_fraction < .2).sum()))
