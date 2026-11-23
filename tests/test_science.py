"""Mathematical identities and frozen design invariants."""

import itertools
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
import geopandas as gpd
from src import diagnosis, gis
from src.grids import shifted_grid_p501
from src.p3_01_validation import select_checkpoint


@pytest.mark.parametrize(
    "tp,fp,fn,tn",
    [(10, 0, 0, 90), (10, 20, 20, 50), (0, 0, 40, 60), (0, 40, 0, 60)],
)
def test_area_identities(tp, fp, fn, tn):
    record = gis.derive(tp, fp, fn, tn, 100)
    record.update(observed_m2=100, scale_m=100)
    frame = diagnosis.metrics(pd.DataFrame([record]))
    assert frame.E_spatial.iloc[0] - frame.E_net.iloc[0] == pytest.approx(
        2 * min(fp, fn) / 100
    )
    assert record["pred_m2"] - record["gt_m2"] == fp - fn
    if fp + fn == 0:
        assert np.isnan(frame.C.iloc[0])


def test_threshold_boundaries_and_support():
    frame = pd.DataFrame(
        [
            dict(E_net=0.02, E_spatial=0.05, Q=0.50),
            dict(E_net=0.01, E_spatial=0.1, Q=0.49),
        ]
    )
    result = diagnosis.classify(frame, 0.02, 0.05, 0.50)
    assert result.joint_scenario_pass.tolist() == [True, False]
    assert result.hidden_spatial_error.tolist() == [False, True]


def test_shift_preserves_support():
    region = box(25, 30, 170, 190)
    grid = gpd.GeoDataFrame(
        [
            dict(
                grid_id=f"{x}_{y}",
                geometry=box(x * 100, y * 100, (x + 1) * 100, (y + 1) * 100),
            )
            for x, y in itertools.product(range(2), repeat=2)
        ],
        crs=32636,
    )
    for dx, dy in ((0.5, 0), (0, 0.5), (0.5, 0.5)):
        shifted = shifted_grid_p501(grid, region, 100, dx, dy)
        assert shifted.area.sum() == pytest.approx(region.area)


def test_checkpoint_tie():
    assert (
        select_checkpoint(
            [
                dict(epoch=1, checkpoint_score=0.8),
                dict(epoch=2, checkpoint_score=0.80000000000001),
            ]
        )
        == 1
    )


def test_frozen_cohort_and_factorial():
    root = Path(__file__).resolve().parents[1]
    with (root / "configs/P3_01_FIX03_effective_manifest.csv").open(
        encoding="utf-8"
    ) as h:
        rows = list(csv.DictReader(h))
    for city, expected in [
        ("AOI_3_Paris", (428, 150, 111)),
        ("AOI_5_Khartoum", (384, 141, 131)),
    ]:
        for split, count in zip(("train", "validation", "test"), expected):
            assert (
                sum(
                    r["city"] == city and r["effective_split"] == split
                    for r in rows
                )
                == count
            )
    excluded = [r for r in rows if r["image_id"] == "AOI_5_Khartoum_img894"][0]
    assert excluded["effective_train"] == "0"
    assert (
        len(
            list(
                itertools.product(
                    range(2), range(3), range(4), range(3), range(4)
                )
            )
        )
        == 288
    )


def test_prepared_figure_table_does_not_require_original_label_tree(
    tmp_path, monkeypatch
):
    """Prepared assets must not trigger resolution of unused archival paths."""
    from src.figure_paths import read_table

    monkeypatch.delenv("DATA_ROOT", raising=False)
    rgb = tmp_path / "rgb.tif"
    rgb.touch()
    table = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            dict(
                rgb_path=str(rgb),
                original_geojson_path="original/unused.geojson",
            )
        ]
    ).to_csv(table, index=False)
    result = read_table(table)
    assert result.rgb_path.iloc[0] == str(rgb)
