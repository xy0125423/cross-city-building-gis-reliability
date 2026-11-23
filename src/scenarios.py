"""Original P5-03 27-scenario strata and error-budget calculation."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import geopandas as gpd
from src.diagnosis import MAIN, classify, metrics as diag_metrics, summarize as diag_summarize

def _evaluate_27_and_budgets(
    frame: pd.DataFrame | gpd.GeoDataFrame,
    arch: str,
    seed: int,
    origin: str,
    source_tag: str,
    scenarios: list[dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """对单组 (arch, city, route, seed, scale, origin) 计算全 27 种情景与误差预算。"""
    city = str(frame["city"].iloc[0])
    route = str(frame["route"].iloc[0])
    scale_m = int(frame["scale_m"].iloc[0])
    met = diag_metrics(frame)

    all_27, main_rows, b_all, b_main = [], [], [], []
    for scene in scenarios:
        net_tol, spat_tol, q_min = scene["net"], scene["spatial"], scene["Q"]
        classified = classify(met, net_tol, spat_tol, q_min)
        strata = {
            "all": classified,
            "nonempty": classified[classified.reference_nonempty == 1],
            "empty": classified[classified.reference_nonempty == 0],
            "low_support": classified[classified.support_low],
            "sufficient_support": classified[~classified.support_low],
        }
        is_main = (net_tol, spat_tol, q_min) == MAIN

        for label, sub in strata.items():
            both = sub.net_pass & sub.spatial_pass
            rec = {
                "architecture": arch,
                "seed": seed,
                "origin": origin,
                "source_type": source_tag,
                "city": city,
                "route": route,
                "scale_m": scale_m,
                "stratum": label,
                "net_tolerance": net_tol,
                "spatial_tolerance": spat_tol,
                "Q_min": q_min,
                **diag_summarize(sub),
                "baseline2_pass_cells": int(both.sum()),
                "baseline1_pass_area_m2": float(sub.loc[sub.net_pass, "observed_m2"].sum()),
                "baseline2_pass_area_m2": float(sub.loc[both, "observed_m2"].sum()),
                "joint_pass_area_m2": float(sub.loc[sub.joint_scenario_pass, "observed_m2"].sum()),
                "support_only_restriction_cells": int((both & sub.support_low).sum()),
            }
            all_27.append(rec)
            if is_main:
                main_rows.append(rec)

        for label in ("all", "nonempty"):
            sub = strata[label]
            low = sub.net_pass
            hidden = sub.hidden_spatial_error
            pure = np.minimum(sub.fp_m2, sub.fn_m2) <= 1e-8
            wrong = float((sub.loc[low, "fp_m2"] + sub.loc[low, "fn_m2"]).sum())
            net = float(abs(sub.loc[low, "pred_m2"] - sub.loc[low, "gt_m2"]).sum())
            brec = {
                "architecture": arch,
                "seed": seed,
                "origin": origin,
                "source_type": source_tag,
                "city": city,
                "route": route,
                "scale_m": scale_m,
                "stratum": label,
                "net_tolerance": net_tol,
                "spatial_tolerance": spat_tol,
                "Q_min": q_min,
                "hidden_cells": int(hidden.sum()),
                "hidden_pure_or_zero_cancellation_cells": int((hidden & pure).sum()),
                "hidden_positive_cancellation_cells": int((hidden & ~pure).sum()),
                "net_pass_spatial_error_m2": wrong,
                "net_pass_abs_net_error_m2": net,
                "net_pass_cancelled_spatial_error_m2": wrong - net,
                "hidden_grid_observed_area_m2": float(sub.loc[hidden, "observed_m2"].sum()),
                "actual_spatial_error_in_hidden_grids_m2": float((sub.loc[hidden, "fp_m2"] + sub.loc[hidden, "fn_m2"]).sum()),
            }
            b_all.append(brec)
            if is_main:
                b_main.append(brec)

    return all_27, main_rows, b_all, b_main
