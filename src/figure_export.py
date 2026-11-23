"""Convert newly computed experiment outputs to the existing formal plot schema."""

from __future__ import annotations
import itertools
import json
from pathlib import Path
import shutil
import geopandas as gpd
import numpy as np
import pandas as pd
from src import diagnosis, gis
from src.pipeline import ROOT, ROUTES, SEEDS, ORIGINS, rows
from src.scenarios import _evaluate_27_and_budgets
from src.p3_02_metrics import paired_comparison
from src.p3_01_validation import metrics


def write_csv(root: Path, relative: str, records) -> None:
    """Write a derived table inside a fresh figure project."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    (
        records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    ).to_csv(path, index=False)


def scene_asset_fields(manifest: list[dict], image_id: str) -> dict[str, str]:
    """Carry the prepared imagery and support into the critical-scene catalog."""
    scene = next(row for row in manifest if row["image_id"] == image_id)
    return {"RGB": scene["rgb_path"], "support": scene["support_path"]}


def gis_tables(root: Path, combinations) -> None:
    """Rebuild all 288/27 tables and origin ranges using the original formulas."""
    all27, main, b27, bmain, summaries = [], [], [], [], []
    scenarios = [
        dict(net=n, spatial=s, Q=q)
        for n, s, q in itertools.product(
            (0.01, 0.02, 0.05), (0.02, 0.05, 0.10), (0.2, 0.5, 0.8)
        )
    ]
    for arch, seed, route, scale, origin, frame in combinations:
        city = ROUTES[route][1]
        frame = frame.copy()
        frame["city"] = city
        frame["route"] = route
        frame["scale_m"] = scale
        a, m, b, bm = _evaluate_27_and_budgets(
            frame, arch, seed, origin, "new_run", scenarios
        )
        all27.extend(a)
        main.extend(m)
        b27.extend(b)
        bmain.extend(bm)
        rec = dict(
            architecture=arch,
            seed=seed,
            origin=origin,
            **gis.summary(frame, city, route, scale),
        )
        rec["E_spatial_pooled"] = rec["total_wrong_m2"] / rec["observed_m2"]
        summaries.append(rec)
    sumdf = pd.DataFrame(summaries)
    maindf = pd.DataFrame(main)
    assert len(sumdf) == 288 and len(maindf) == 1440
    for name, data in [
        ("all_288_spatial_summaries", sumdf),
        ("all_288_main_scenario", maindf),
        ("all_288_27_scenarios", all27),
        ("all_288_error_budgets_main", bmain),
        ("all_288_error_budgets_27", b27),
    ]:
        write_csv(root, f"results/P5_03/tables/{name}.csv", data)
    mainall = maindf[maindf.stratum == "all"].merge(
        sumdf,
        on=["architecture", "seed", "origin", "city", "route", "scale_m"],
        suffixes=("", "_summary"),
    )
    mainall["hidden_area_pct"] = 100 * mainall.hidden_area_fraction_total
    mainall["hidden_cond_pct"] = 100 * mainall.hidden_fraction_of_net_low
    mainall["joint_pass_pct"] = 100 * mainall.joint_pass_cells / mainall.cells
    tracks = [
        "hidden_cells",
        "hidden_area_pct",
        "hidden_cond_pct",
        "baseline_net_low_cells",
        "baseline2_pass_cells",
        "joint_pass_cells",
        "joint_pass_pct",
        "low_support_cells",
        "weighted_cover_MAE",
        "E_spatial_pooled",
        "spatial_cancellation",
        "weighted_cover_bias",
    ]
    sensitivity = []
    for key, sub in mainall.groupby(
        ["architecture", "city", "route", "seed", "scale_m"]
    ):
        item = dict(
            zip(["architecture", "city", "route", "seed", "scale_m"], key)
        )
        by = sub.set_index("origin")
        for column in tracks:
            for origin in ORIGINS:
                item[f"{column}_{origin}"] = by.loc[origin, column]
            item[f"{column}_min"] = sub[column].min()
            item[f"{column}_max"] = sub[column].max()
            item[f"{column}_range"] = sub[column].max() - sub[column].min()
        item["zero_in_original"] = int(by.loc["original", "hidden_cells"] == 0)
        item["zero_across_all_4_origins"] = int((sub.hidden_cells == 0).all())
        item["shifted_max_hidden_when_original_zero"] = (
            sub[sub.origin != "original"].hidden_cells.max()
            if item["zero_in_original"]
            else np.nan
        )
        sensitivity.append(item)
    ranges = pd.DataFrame(sensitivity)
    write_csv(
        root, "results/P5_03/tables/origin_sensitivity_ranges.csv", ranges
    )
    write_csv(
        root,
        "results/P5_03/tables/zero_hidden_origin_stability.csv",
        ranges[ranges.zero_in_original == 1],
    )
    original = sumdf[sumdf.origin == "original"].copy()
    original["fp_ha"] = original.fp_m2 / 10000
    original["fn_ha"] = original.fn_m2 / 10000
    original["cover_bias_pp"] = original.weighted_cover_bias * 100
    original["rel_area_pct"] = original.pooled_relative_area_error * 100
    original["E_spatial_pct"] = original.E_spatial_pooled * 100
    original["E_net_pct"] = original.weighted_cover_MAE * 100
    original["cancel_pct"] = original.spatial_cancellation * 100
    for name in ("F3", "F4"):
        write_csv(root, f"results/P5_02/figures/{name}_source.csv", original)


def export(work: Path, output: Path) -> None:
    """Export new run results; never substitute recorded metrics for missing outputs."""
    output.mkdir(parents=True, exist_ok=False)
    manifest = rows(work / "manifest.csv")
    write_csv(output, "configs/P3_01_FIX03_effective_manifest.csv", manifest)
    shutil.copytree(ROOT / "figure_inputs", output / "figure_inputs")
    records = []
    catalog = []
    ledgers = {}
    combinations = []
    density = []
    for arch, seed, route in itertools.product(
        ("UNet", "DeepLabV3Plus"), SEEDS, ROUTES
    ):
        pred = work / "predictions" / arch / str(seed) / route
        ledger = pd.read_csv(pred / "per_scene.csv").to_dict("records")
        ledgers[arch, seed, route] = ledger
        records.append(
            dict(architecture=arch, seed=seed, route=route, **metrics(ledger))
        )
        for scale, origin in itertools.product((100, 250, 500), ORIGINS):
            frame = pd.read_csv(
                work
                / "gis"
                / arch
                / str(seed)
                / route
                / f"{scale}m_{origin}_cells.csv"
            )
            combinations.append((arch, seed, route, scale, origin, frame))
        if arch == "UNet" and seed == 20261115:
            for row in manifest:
                if (
                    row["city"] == ROUTES[route][1]
                    and row["effective_split"] == "test"
                ):
                    dest = (
                        output
                        / f"results/P3_02/error_maps/{route}/{row['image_id']}_errors.tif"
                    )
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(
                        pred / f"{row['image_id']}_errors.tif", dest
                    )
                    catalog.append(
                        dict(
                            row,
                            route=route,
                            error_path=dest.relative_to(output).as_posix(),
                        )
                    )
            frame = pd.read_csv(
                work
                / "gis"
                / arch
                / str(seed)
                / route
                / "100m_original_cells.csv"
            )
            for low, high, label in [
                (1e-10, 0.05, "0-5%"),
                (0.05, 0.15, "5-15%"),
                (0.15, 0.30, "15-30%"),
                (0.30, 1.000001, ">30%"),
            ]:
                sub = frame[(frame.gt_cover > low) & (frame.gt_cover <= high)]
                density.append(
                    dict(
                        city=ROUTES[route][1],
                        route=route,
                        band=label,
                        cells=len(sub),
                        observed_area_share=sub.observed_m2.sum()
                        / frame.observed_m2.sum(),
                        FN_area_share=sub.fn_m2.sum() / frame.fn_m2.sum(),
                        GT_area_share=sub.gt_m2.sum() / frame.gt_m2.sum(),
                        conditional_FN_fraction=sub.fn_m2.sum()
                        / sub.gt_m2.sum()
                        if sub.gt_m2.sum()
                        else np.nan,
                    )
                )
    write_csv(output, "results/P5_02/figures/F1_source.csv", records)
    write_csv(output, "results/P3_02/GIS_output_catalog.csv", catalog)
    write_csv(
        output,
        "results/P4_01/tables/FN_concentration_vs_GT_exposure.csv",
        density,
    )
    comparisons = []
    for within, cross in [("A", "C"), ("D", "B")]:
        result = paired_comparison(
            ledgers["UNet", 20261115, within], ledgers["UNet", 20261115, cross]
        )
        for metric, values in result["metrics"].items():
            comparisons.append(
                dict(target_city=ROUTES[within][1], metric=metric, **values)
            )
    write_csv(
        output, "results/P3_02/metrics/paired_differences.csv", comparisons
    )
    gis_tables(output, combinations)
    city = "AOI_5_Khartoum"
    source = work / "gis/UNet/20261115/D/spatial.gpkg"
    target = output / f"results/P4_01/gpkg/{city}_P4_01.gpkg"
    target.parent.mkdir(parents=True, exist_ok=True)
    for layer in ["scene_ownership", "test_observed_region"]:
        gpd.read_file(source, layer=layer).to_file(
            target, layer=layer, driver="GPKG"
        )
    for scale in [100, 250, 500]:
        frame = gpd.read_file(source, layer=f"D_{scale}m_original")
        frame.to_file(target, layer=f"D_grid_{scale}m", driver="GPKG")
        if scale == 100:
            target2 = output / f"results/P4_02/gpkg/{city}_P4_02.gpkg"
            target2.parent.mkdir(parents=True, exist_ok=True)
            diagnosis.classify(
                diagnosis.metrics(frame), *diagnosis.MAIN
            ).to_file(target2, layer="D_diagnosis_100m", driver="GPKG")
            gpd.read_file(source, layer="test_observed_region").to_file(
                target2, layer="test_observed_region", driver="GPKG"
            )
    source = work / "gis/DeepLabV3Plus/20261121/D/spatial.gpkg"
    for origin in ORIGINS:
        rel = (
            f"results/P5_02/gpkg/{city}_seed20261121_P5_02.gpkg"
            if origin == "original"
            else f"results/P5_03/gpkg/DeepLabV3Plus_{city}_seed20261121_shifted_P5_03.gpkg"
        )
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        gpd.read_file(source, layer=f"D_500m_{origin}").to_file(
            target, layer=f"D_500m_{origin}", driver="GPKG"
        )
        if origin == "original":
            gpd.read_file(source, layer="test_observed_region").to_file(
                target, layer="test_observed_region", driver="GPKG"
            )
    assets = []
    for route in ["B", "D"]:
        src = (
            work
            / f"predictions/UNet/20261121/{route}/{city}_img1028_errors.tif"
        )
        dest = output / f"results/critical_scene/{route}_errors.tif"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        assets.append(
            dict(
                architecture="UNet",
                seed=20261121,
                route=route,
                kind="error",
                path=dest.relative_to(output).as_posix(),
                **scene_asset_fields(manifest, f"{city}_img1028"),
            )
        )
    write_csv(
        output, "results/P5_02/diagnostics/img1028_GIS_assets.csv", assets
    )
    (output / "export_status.json").write_text(
        json.dumps(
            dict(
                source="new experiment outputs only",
                formal_figures="1–5",
                supplementary="use recorded efficiency inputs or add actual runtime profiling",
            )
        ),
        encoding="utf-8",
    )
