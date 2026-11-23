"""Run all twelve source models and all 24 inference/GIS routes."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import CITIES, SEEDS, ROUTES, train, evaluate, spatial

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("train", "evaluate", "gis"), required=True
    )
    args = parser.parse_args()
    for architecture in ("UNet", "DeepLabV3Plus"):
        for seed in SEEDS:
            if args.stage == "train":
                for city in CITIES:
                    train(args.work, architecture, city, seed)
            else:
                for route in ROUTES:
                    (evaluate if args.stage == "evaluate" else spatial)(
                        args.work, architecture, seed, route
                    )
