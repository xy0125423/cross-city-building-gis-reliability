"""Check portable GIS export against the recorded 288 sets of cell statistics."""

from pathlib import Path
import argparse, itertools, json, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from src.pipeline import ROOT, ROUTES, SEEDS, ORIGINS
from src.figure_export import gis_tables


def run(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    combinations = []
    for arch, seed, route, scale, origin in itertools.product(
        ("UNet", "DeepLabV3Plus"), SEEDS, ROUTES, (100, 250, 500), ORIGINS
    ):
        city = ROUTES[route][1]
        if arch == "UNet" and seed == 20261115 and origin == "original":
            path = (
                ROOT
                / f"results/P4_01/tables/{city}_{route}_{scale}m_cells.csv"
            )
        elif arch == "UNet" and (seed == 20261115 or origin == "original"):
            path = (
                ROOT
                / f"results/P5_01/tables/cells/{city}_{route}_{seed}_{scale}m_{origin}.csv"
            )
        elif origin == "original":
            path = (
                ROOT
                / f"results/P5_02/tables/cells/{city}_{route}_{seed}_{scale}m_original.csv"
            )
        else:
            path = (
                ROOT
                / f"results/P5_03/tables/cells/{arch}_{city}_{route}_{seed}_{scale}m_{origin}.csv"
            )
        combinations.append(
            (arch, seed, route, scale, origin, pd.read_csv(path))
        )
    gis_tables(output, combinations)
    keys = [
        "architecture",
        "city",
        "route",
        "seed",
        "scale_m",
        "origin",
        "stratum",
    ]
    expected = (
        pd.read_csv(ROOT / "results/P5_03/tables/all_288_main_scenario.csv")
        .sort_values(keys)
        .reset_index(drop=True)
    )
    actual = (
        pd.read_csv(output / "results/P5_03/tables/all_288_main_scenario.csv")
        .sort_values(keys)
        .reset_index(drop=True)
    )
    assert expected[keys].equals(actual[keys])
    compared = []
    for column in expected.select_dtypes(include="number").columns:
        if column in actual:
            assert np.allclose(
                expected[column],
                actual[column],
                equal_nan=True,
                rtol=1e-9,
                atol=1e-7,
            ), column
            compared.append(column)
    result = dict(
        status="PASS",
        combinations=288,
        main_rows=len(actual),
        numeric_columns=compared,
        scope="Recomputed from packaged recorded cell statistics; not new model training",
    )
    (output / "schema_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
