"""Synthetic disk-to-model-to-GIS acceptance; no historical assets are read."""

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping
import torch

from src import core, inference
from src.pipeline import model, grid_statistics, spatial_frames, save


def run(out: Path) -> dict:
    """Exercise real raster/vector IO, both model updates and area conservation."""
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20261115)
    rng = np.random.default_rng(20261115)
    affine = from_origin(32.5, 15.6, 0.000027, 0.000027)
    profile = dict(
        driver="GTiff", width=32, height=32, crs="EPSG:4326", transform=affine
    )
    image = rng.integers(1, 2000, (3, 32, 32), dtype="uint16")
    support = np.ones((32, 32), dtype="uint8")
    support[:2] = 2
    with rasterio.open(
        out / "rgb.tif", "w", count=3, dtype="uint16", **profile
    ) as ds:
        ds.write(image)
    with rasterio.open(
        out / "support.tif", "w", count=1, dtype="uint8", **profile
    ) as ds:
        ds.write(support, 1)
    geom = box(*(affine * (8, 24)), *(affine * (24, 8)))
    payload = dict(
        type="FeatureCollection",
        features=[dict(type="Feature", properties={}, geometry=mapping(geom))],
    )
    (out / "label.geojson").write_text(json.dumps(payload), encoding="utf-8")
    with (
        rasterio.open(out / "rgb.tif") as rgb,
        rasterio.open(out / "support.tif") as mask,
    ):
        native = core.rasterize_scene(
            rgb, mask, json.loads((out / "label.geojson").read_text())
        )
    city = "AOI_5_Khartoum"
    row = dict(
        city=city,
        image_id="synthetic",
        spatial_group_id="synthetic_group",
        split="train",
        effective_split="train",
        label_empty="0",
    )
    norm = core.normalization([row], city, lambda _: native)
    sample = core.SourceDataset([row], city, "train", lambda _: native, norm)[
        0
    ]
    reports = []
    for architecture in ("UNet", "DeepLabV3Plus"):
        net = model(architecture)
        initial = next(net.parameters()).detach().clone()
        opt = torch.optim.Adam(net.parameters(), lr=0.001)
        # Batch two also exercises DeepLab training-mode batch normalization.
        net.train()
        opt.zero_grad()
        x = sample["x"][None].repeat(2, 1, 1, 1)
        loss = core.objective(
            net(x),
            sample["y"][None].repeat(2, 1, 1, 1),
            sample["v"][None].repeat(2, 1, 1, 1),
        )
        loss.backward()
        opt.step()
        assert torch.isfinite(loss) and not torch.equal(
            initial, next(net.parameters()).detach()
        )
        torch.save(
            dict(state_dict=net.state_dict()), out / f"{architecture}.pt"
        )
        restored = model(architecture)
        restored.load_state_dict(
            torch.load(out / f"{architecture}.pt", weights_only=True)[
                "state_dict"
            ]
        )
        restored.eval()
        with torch.no_grad():
            probability = core.upsample_probability(
                restored(inference.preprocess(native, norm)), (32, 32)
            )
        pred = probability >= 0.5
        errors = inference.error_classes(
            pred, native["label"], native["valid"]
        )
        inference.export_raster(
            out / f"{architecture}_errors.tif",
            errors,
            native,
            nodata=255,
            kind="error_classes",
            tags={},
        )
        with rasterio.open(out / f"{architecture}_errors.tif") as ds:
            assert np.array_equal(ds.read(1), errors)
        from src.p3_01_validation import scene_counts, metrics

        detail = scene_counts(
            city,
            "synthetic",
            "synthetic_group",
            pred,
            native["label"],
            native["valid"],
            native["area"],
        )
        region, parts, owners = spatial_frames(
            [
                ("a", errors, native["valid"], affine),
                ("b", errors, native["valid"], affine),
            ],
            32636,
        )
        assert owners.geometry.iloc[1].area < 1e-8
        cells = []
        for scale in (100, 250, 500):
            for origin in ("original", "x_half", "y_half", "xy_half"):
                frame = grid_statistics(region, parts, scale, origin)
                assert abs(frame.observed_m2.sum() - region.area) < 0.05
                frame.drop(columns="geometry").to_csv(
                    out / f"{architecture}_{scale}_{origin}.csv", index=False
                )
                cells.append(len(frame))
        reports.append(
            dict(
                architecture=architecture,
                loss=float(loss.detach()),
                micro=metrics([detail]),
                grid_cells=cells,
            )
        )
    result = dict(
        status="PASS",
        synthetic_only=True,
        formal_results_reproduced=False,
        coverage=[
            "GeoTIFF/GeoJSON",
            "both model forward/backward",
            "checkpoint reload",
            "native inference/evaluation/export",
            "overlap ownership",
            "3 scales × 4 origins",
        ],
        models=reports,
    )
    save(out / "smoke_report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
