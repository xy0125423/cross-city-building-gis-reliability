"""Regenerate the unchanged formal layouts from packaged derived plot inputs."""

import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=("1", "2", "3", "4", "5", "supplement", "all"),
        required=True,
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        help="New figure project from export_figure_inputs.py",
    )
    args = parser.parse_args()
    if args.inputs:
        os.environ["FIGURE_INPUT_ROOT"] = str(args.inputs.resolve())
    names = {
        "1": "build_fig1_study_area.py",
        "2": "build_fig2_cross_city_degradation.py",
        "3": "build_fig3_multiscale_gis_errors.py",
        "4": "build_fig4_dual_error_diagnosis.py",
        "5": "build_fig5_maup_origin_sensitivity.py",
        "supplement": "build_supplementary_figures_v2.py",
    }
    for key in names if args.figure == "all" else [args.figure]:
        runpy.run_path(
            str(ROOT / "scripts/figures_original" / names[key]),
            run_name="__main__",
        )
