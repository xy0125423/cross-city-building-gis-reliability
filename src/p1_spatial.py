"""Geometry and conservative observation-footprint rules for P1-01."""

from __future__ import annotations

import numpy as np
import shapely
from shapely.affinity import translate
from scipy import ndimage

RULE = {
    "version": "P1_01_v1",
    "classes": {"0": "supported_edge_padding", "1": "conservative_observed",
                "2": "uncertain", "3": "source_mask_invalid"},
    "connectivity": 4,
    "minimum_padding_component_pixels": 64,
    "native_support_interior_erosion_pixels": 2,
    "minimum_native_zero_agreement": 0.99,
    "padding_guard_pixels": 1,
    "minimum_building_covered_area_fraction": 0.999,
    "density_grid_metres": 500,
    "authority": "local evidence-based operational footprint, not publisher ground truth",
}


def polygon_parts(geometry) -> tuple[list, list]:
    """Keep polygon parts and separately retain evidence of collapsed remnants."""
    if geometry.is_empty:
        return [], []
    if geometry.geom_type == "Polygon":
        return [geometry], []
    if hasattr(geometry, "geoms"):
        polygons, remnants = [], []
        for part in geometry.geoms:
            sub_polygons, sub_remnants = polygon_parts(part)
            polygons.extend(sub_polygons)
            remnants.extend(sub_remnants)
        return polygons, remnants
    return [], [geometry]


def polygonal_valid(geometry):
    """Return polygonal output; never use this to silently repair source labels."""
    candidate = geometry if geometry.is_valid else shapely.make_valid(geometry)
    polygons, _ = polygon_parts(candidate)
    return shapely.union_all(polygons) if polygons else shapely.Polygon()


def classify_footprint(
    rgb_zero: np.ndarray,
    pan_zero: np.ndarray,
    ms_zero: np.ndarray,
    native_zero: np.ndarray,
    mask_valid: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Require boundary, component size and native evidence before inferring fill.

    native_zero is nearest-neighbour mapped onto the RGB grid: 0=nonzero,
    1=all native bands zero, 255=outside native support. All unresolved zero
    evidence remains uncertain, rather than being called NoData.
    """
    joint_zero = rgb_zero & pan_zero & ms_zero
    components, count = ndimage.label(joint_zero)
    boundary_ids = set(np.unique(np.concatenate((components[0], components[-1],
                                                 components[:, 0], components[:, -1])))) - {0}
    padding = np.zeros(joint_zero.shape, dtype=bool)
    accepted, rejected = 0, 0
    agreements = []
    sizes = np.bincount(components.ravel())
    for component_id in boundary_ids:
        if sizes[component_id] < RULE["minimum_padding_component_pixels"]:
            rejected += 1
            continue
        component = components == component_id
        interior = ndimage.binary_erosion(
            component, iterations=RULE["native_support_interior_erosion_pixels"])
        # Thin components without a supported interior remain uncertain.
        agreement = float(np.mean(native_zero[interior] == 1)) if interior.any() else 0.0
        agreements.append(agreement)
        if agreement >= RULE["minimum_native_zero_agreement"]:
            padding |= component
            accepted += 1
        else:
            rejected += 1
    guard = ndimage.binary_dilation(padding, iterations=RULE["padding_guard_pixels"]) & ~padding
    uncertain = ((rgb_zero | pan_zero | ms_zero | (native_zero != 0)) & ~padding) | guard
    classified = np.ones(joint_zero.shape, dtype=np.uint8)
    classified[uncertain] = 2
    classified[padding] = 0
    classified[~mask_valid] = 3
    return classified, {
        "joint_zero_pixels": int(joint_zero.sum()),
        "edge_components_accepted": accepted,
        "edge_components_rejected": rejected,
        "native_zero_agreement_min": min(agreements) if agreements else 1.0,
        "interior_zero_pixels": int(np.sum(joint_zero & (components > 0) &
                                          ~np.isin(components, list(boundary_ids)))),
        "rgb_zero_not_padding_pixels": int(np.sum(rgb_zero & ~padding)),
        "rgb_pan_zero_disagreement_pixels": int(np.sum(rgb_zero != pan_zero)),
        "rgb_ms_zero_disagreement_pixels": int(np.sum(rgb_zero != ms_zero)),
    }


def shape_metrics(geometry) -> dict:
    """Compute planar metrics only after the caller projects geometry to metres."""
    area, perimeter = geometry.area, geometry.length
    # GEOS oriented envelopes lose precision with small features at large UTM
    # coordinates. Translation preserves shape while conditioning this operation.
    centroid = geometry.centroid
    local = translate(geometry, xoff=-centroid.x, yoff=-centroid.y)
    rectangle = local.minimum_rotated_rectangle
    coordinates = np.asarray(rectangle.exterior.coords)[:, :2]
    lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    short, long = float(lengths.min()), float(lengths.max())
    polygons, _ = polygon_parts(geometry)
    solidity = local.area / local.convex_hull.area
    rectangularity = local.area / rectangle.area
    if solidity > 1 + 1e-9 or rectangularity > 1 + 1e-9:
        raise ValueError("Shape containment ratio exceeds one beyond floating tolerance")
    return {
        "area_m2": area, "perimeter_m": perimeter,
        "compactness": 4 * np.pi * area / perimeter**2 if perimeter else 0,
        "solidity": min(solidity, 1.0),
        "elongation": long / short if short else None,
        "rectangularity": min(rectangularity, 1.0),
        "solidity_roundoff_adjusted": int(solidity > 1),
        "rectangularity_roundoff_adjusted": int(rectangularity > 1),
        "polygon_parts": len(polygons),
        "holes": sum(len(p.interiors) for p in polygons),
    }
