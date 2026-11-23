"""Portable orchestration for the frozen Paris–Khartoum experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
import shapely
from shapely.geometry import box
import torch
from torch.utils.data import DataLoader
import yaml

from src import core, diagnosis, gis, inference, support
from src.grids import shifted_grid_p501
from src.p3_01_validation import metrics, scene_counts, select_checkpoint
from src.P5_02.deeplabv3plus import DeepLabV3Plus
from src.vendor.p2_unet.unet import UNet

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261115, 20261121, 20261122)
CITIES = ("AOI_3_Paris", "AOI_5_Khartoum")
ROUTES = {
    "A": (CITIES[0], CITIES[0]),
    "B": (CITIES[0], CITIES[1]),
    "C": (CITIES[1], CITIES[0]),
    "D": (CITIES[1], CITIES[1]),
}
ORIGINS = {
    "original": (0, 0),
    "x_half": (0.5, 0),
    "y_half": (0, 0.5),
    "xy_half": (0.5, 0.5),
}


def save(path: Path, value: object) -> None:
    """Create a finite JSON artifact, preserving any existing output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path: Path) -> str:
    """Stream a reproducibility fingerprint."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    """Read a UTF-8 manifest."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def model(architecture: str) -> torch.nn.Module:
    """Instantiate the actual frozen architectures without pretrained weights."""
    return (
        UNet(in_channels=3, out_channels=1, init_features=8)
        if architecture == "UNet"
        else DeepLabV3Plus()
    )


def prepare(data_root: Path, work: Path) -> None:
    """Resolve public products and regenerate conservative support into runs/."""
    cohort = rows(ROOT / "configs/P3_01_FIX03_effective_manifest.csv")
    support_checks = json.loads(
        (ROOT / "configs/support_array_sha256.json").read_text(
            encoding="utf-8"
        )
    )
    index: dict[str, list[Path]] = {}
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".tif", ".geojson"):
            index.setdefault(p.name, []).append(p)

    def resolve(relative: str) -> Path:
        candidate = data_root / relative
        if candidate.is_file():
            return candidate.resolve()
        candidates = index.get(Path(relative).name, [])
        if len(candidates) != 1:
            raise ValueError(f"Missing or ambiguous public asset: {relative}")
        return candidates[0].resolve()

    work.mkdir(parents=True, exist_ok=False)
    prepared = []
    for row in cohort:
        rgb = resolve(row["rgb_path"])
        if digest(rgb) != row["rgb_sha256"]:
            raise ValueError("RGB hash mismatch: " + row["image_id"])
        label = ROOT / "configs/label_repairs" / Path(row["geojson_path"]).name
        if row["label_source_kind"] != "P1_derived_fixed":
            label = resolve(row["original_geojson_path"])
        if digest(label) != row["label_sha256"]:
            raise ValueError("Label hash mismatch: " + row["image_id"])
        identifier = row["image_id"]
        products = {"RGB-PanSharpen": str(rgb)}
        for product in ("PAN", "MUL-PanSharpen", "MUL"):
            products[product] = str(resolve(f"{product}_{identifier}.tif"))
        mask_path = work / "support" / f"{identifier}_support.tif"
        classified, evidence, _ = support.observation_mask(products, mask_path)
        if (
            hashlib.sha256(classified.tobytes()).hexdigest()
            != support_checks[identifier]
        ):
            raise ValueError("Support rule drift: " + identifier)
        new = dict(
            row,
            rgb_path=str(rgb),
            geojson_path=str(label.resolve()),
            support_path=str(mask_path.resolve()),
        )
        prepared.append(new)
    with (work / "manifest.csv").open("x", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(prepared[0]))
        writer.writeheader()
        writer.writerows(prepared)
    save(
        work / "preparation.json",
        dict(
            scenes=len(prepared),
            manifest_sha256=digest(
                ROOT / "configs/P3_01_FIX03_effective_manifest.csv"
            ),
            support_rule=support.RULE,
            excluded="AOI_5_Khartoum_img894",
            splits="Frozen effective_split, including quality/buffer/development exclusions; never re-search split.",
        ),
    )


def load_scene(row: dict) -> dict:
    """Read the actual native GeoTIFF, Support and GeoJSON inputs."""
    with (
        rasterio.open(row["rgb_path"]) as rgb,
        rasterio.open(row["support_path"]) as mask,
    ):
        payload = json.loads(
            Path(row["geojson_path"]).read_text(encoding="utf-8")
        )
        return core.rasterize_scene(rgb, mask, payload)


def train(work: Path, architecture: str, city: str, seed: int) -> None:
    """Run the frozen 12-epoch source-only training and checkpoint rule."""
    cohort = rows(work / "manifest.csv")
    training = [
        r
        for r in cohort
        if r["city"] == city and r["effective_split"] == "train"
    ]
    validation = [
        r
        for r in cohort
        if r["city"] == city and r["effective_split"] == "validation"
    ]
    if set(r["spatial_group_id"] for r in training) & set(
        r["spatial_group_id"] for r in validation
    ):
        raise ValueError("Train/validation spatial group overlap")
    cfg = yaml.safe_load(
        (ROOT / f"configs/unet_{city.split('_')[-1]}.yaml").read_text(
            encoding="utf-8"
        )
    )
    cfg.update(seed=seed, architecture=architecture)
    core.configure_determinism(cfg)
    norm = core.normalization(training, city, load_scene)
    train_set = core.SourceDataset(training, city, "train", load_scene, norm)
    val_set = core.SourceDataset(
        validation, city, "validation", load_scene, norm
    )
    out = work / "models" / architecture / str(seed) / city
    out.mkdir(parents=True, exist_ok=False)
    save(out / "normalization.json", norm)
    net = model(architecture)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    log = []
    for epoch in range(1, 13):
        order = list(range(len(train_set)))
        random.shuffle(order)
        loader = DataLoader(
            train_set,
            batch_size=4,
            sampler=order,
            num_workers=0,
            drop_last=False,
            collate_fn=core.collate_training,
        )
        net.train()
        total = pixels = 0
        for batch in loader:
            optimizer.zero_grad()
            loss = core.objective(net(batch["x"]), batch["y"], batch["v"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            n = int(batch["v"].sum())
            total += float(loss.detach()) * n
            pixels += n
        report = core.validate(net, val_set)
        report.update(epoch=epoch, train_loss=total / pixels)
        save(out / f"epoch_{epoch:02d}.json", report)
        torch.save(
            dict(
                state_dict=net.state_dict(),
                optimizer=optimizer.state_dict(),
                normalization=norm,
                config=cfg,
                epoch=epoch,
            ),
            out / f"epoch_{epoch:02d}.pt",
        )
        log.append(
            dict(
                epoch=epoch,
                checkpoint_score=report["checkpoint_score"],
                train_loss=report["train_loss"],
                validation_loss=report["validation_loss"],
            )
        )
        print(architecture, city, seed, log[-1], flush=True)
    best = select_checkpoint(log)
    shutil.copyfile(out / f"epoch_{best:02d}.pt", out / "best.pt")
    save(out / "training_log.json", log)
    save(
        out / "completion.json",
        dict(best_epoch=best, best_sha256=digest(out / "best.pt"), epochs=12),
    )


def evaluate(work: Path, architecture: str, seed: int, route: str) -> None:
    """Infer frozen tests using only source-training normalization."""
    source, target = ROUTES[route]
    cp = work / "models" / architecture / str(seed) / source / "best.pt"
    checkpoint = torch.load(cp, map_location="cpu", weights_only=True)
    net = model(architecture)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    torch.set_num_threads(4)
    out = work / "predictions" / architecture / str(seed) / route
    out.mkdir(parents=True, exist_ok=False)
    ledger = []
    for row in rows(work / "manifest.csv"):
        if row["city"] != target or row["effective_split"] != "test":
            continue
        native = load_scene(row)
        with torch.no_grad():
            probability = core.upsample_probability(
                net(inference.preprocess(native, checkpoint["normalization"])),
                native["label"].shape,
            )
        pred = probability >= 0.5
        detail = scene_counts(
            target,
            row["image_id"],
            row["spatial_group_id"],
            pred,
            native["label"],
            native["valid"],
            native["area"],
        )
        ledger.append(detail)
        outputs = {
            "probability": np.where(
                native["valid"], probability, -9999
            ).astype("float32"),
            "prediction": np.where(native["valid"], pred, 255).astype("uint8"),
            "errors": inference.error_classes(
                pred, native["label"], native["valid"]
            ),
        }
        for kind, values in outputs.items():
            inference.export_raster(
                out / f"{row['image_id']}_{kind}.tif",
                values,
                native,
                nodata=-9999 if kind == "probability" else 255,
                kind=kind,
                tags=dict(route=route, architecture=architecture, seed=seed),
            )
    pd.DataFrame(ledger).to_csv(out / "per_scene.csv", index=False)
    save(
        out / "summary.json",
        dict(
            architecture=architecture,
            seed=seed,
            route=route,
            checkpoint_sha256=digest(cp),
            micro=metrics(ledger),
        ),
    )


def spatial_frames(
    records: list[tuple[str, np.ndarray, np.ndarray, object]], epsg: int
):
    """Assign lexicographic scene ownership then intersect native classes in UTM."""
    converter = Transformer.from_crs(4326, epsg, always_xy=True)
    union = shapely.GeometryCollection()
    parts, ownership = [], []
    for identifier, errors, valid, affine in sorted(
        records, key=lambda r: r[0]
    ):
        footprint = shapely.union_all(
            [
                g
                for g, _ in gis.vectorize(
                    valid.astype("uint8"), valid, affine, converter
                )
            ]
        )
        owned = footprint.difference(union)
        union = shapely.union_all([union, footprint])
        ownership.append(dict(image_id=identifier, geometry=owned))
        for geom, code in gis.vectorize(errors, valid, affine, converter):
            clipped = geom.intersection(owned)
            if clipped.area > 1e-10:
                parts.append(dict(code=code, geometry=clipped))
    return (
        union,
        gpd.GeoDataFrame(parts, crs=epsg),
        gpd.GeoDataFrame(ownership, crs=epsg),
    )


def grid_statistics(
    region, parts: gpd.GeoDataFrame, scale: int, origin: str
) -> gpd.GeoDataFrame:
    """Conserve unique observed area across absolute UTM grid origins."""
    xmin, ymin, xmax, ymax = region.bounds
    cells = []
    for ix in range(math.floor(xmin / scale), math.floor(xmax / scale) + 1):
        for iy in range(
            math.floor(ymin / scale), math.floor(ymax / scale) + 1
        ):
            geom = box(
                ix * scale, iy * scale, (ix + 1) * scale, (iy + 1) * scale
            ).intersection(region)
            if geom.area > 1e-6:
                cells.append(dict(grid_id=f"{ix}_{iy}", geometry=geom))
    grid = gpd.GeoDataFrame(cells, crs=parts.crs)
    dx, dy = ORIGINS[origin]
    if dx or dy:
        grid = shifted_grid_p501(grid, region, scale, dx, dy)
    grid["observed_m2"] = grid.area
    grid["observed_fraction"] = grid.observed_m2 / scale**2
    grid["partial_cell"] = (grid.observed_fraction < 1 - 1e-6).astype(int)
    grid["scale_m"] = scale
    tree = parts.sindex
    areas = np.zeros((len(grid), 4))
    for i, geom in enumerate(grid.geometry):
        for j in tree.query(geom, predicate="intersects"):
            areas[i, int(parts.iloc[j].code) - 1] += geom.intersection(
                parts.geometry.iloc[j]
            ).area
    if not np.allclose(areas.sum(axis=1), grid.observed_m2, atol=0.05, rtol=0):
        raise ValueError("Class area conservation failed")
    if abs(grid.observed_m2.sum() - region.area) >= 0.05:
        raise ValueError("Grid support conservation failed")
    for key in gis.derive(*areas[0], float(grid.observed_m2.iloc[0])):
        grid[key] = [
            gis.derive(*a, float(s))[key]
            for a, s in zip(areas, grid.observed_m2)
        ]
    return diagnosis.metrics(grid)


def spatial(work: Path, architecture: str, seed: int, route: str) -> None:
    """Regenerate 3 scales × 4 origins × 27 threshold scenarios for one route."""
    city = ROUTES[route][1]
    pred = work / "predictions" / architecture / str(seed) / route
    out = work / "gis" / architecture / str(seed) / route
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for row in rows(work / "manifest.csv"):
        if row["city"] == city and row["effective_split"] == "test":
            with rasterio.open(pred / f"{row['image_id']}_errors.tif") as ds:
                error = ds.read(1)
                records.append(
                    (row["image_id"], error, error != 255, ds.transform)
                )
    region, parts, ownership = spatial_frames(
        records, 32631 if city == CITIES[0] else 32636
    )
    package = out / "spatial.gpkg"
    ownership.to_file(package, layer="scene_ownership", driver="GPKG")
    gpd.GeoDataFrame([dict(geometry=region)], crs=parts.crs).to_file(
        package, layer="test_observed_region", driver="GPKG"
    )
    scenarios = []
    for scale, origin in itertools.product((100, 250, 500), ORIGINS):
        frame = grid_statistics(region, parts, scale, origin)
        frame["city"] = city
        frame["route"] = route
        frame.to_file(
            package, layer=f"{route}_{scale}m_{origin}", driver="GPKG"
        )
        frame.drop(columns="geometry").to_csv(
            out / f"{scale}m_{origin}_cells.csv", index=False
        )
        for net, wrong, q in itertools.product(
            (0.01, 0.02, 0.05), (0.02, 0.05, 0.10), (0.20, 0.50, 0.80)
        ):
            classified = diagnosis.classify(frame, net, wrong, q)
            scenarios.append(
                dict(
                    architecture=architecture,
                    seed=seed,
                    route=route,
                    city=city,
                    scale_m=scale,
                    origin=origin,
                    net_tolerance=net,
                    spatial_tolerance=wrong,
                    Q_min=q,
                    **diagnosis.summarize(classified),
                )
            )
    pd.DataFrame(scenarios).to_csv(out / "scenarios.csv", index=False)


def main() -> None:
    """Expose real and inspectable reproduction commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser(
        "prepare", help="Resolve public data and regenerate Support"
    )
    prep.add_argument(
        "--data-root", type=Path, default=os.environ.get("DATA_ROOT")
    )
    prep.add_argument("--work", type=Path, required=True)
    for command in ("train", "evaluate", "gis"):
        p = sub.add_parser(command)
        p.add_argument("--work", type=Path, required=True)
        p.add_argument(
            "--architecture", choices=("UNet", "DeepLabV3Plus"), required=True
        )
        p.add_argument("--seed", type=int, choices=SEEDS, required=True)
        if command == "train":
            p.add_argument("--city", choices=CITIES, required=True)
        else:
            p.add_argument("--route", choices=ROUTES, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.data_root is None:
            parser.error("--data-root or DATA_ROOT is required")
        prepare(Path(args.data_root), args.work)
    elif args.command == "train":
        train(args.work, args.architecture, args.city, args.seed)
    elif args.command == "evaluate":
        evaluate(args.work, args.architecture, args.seed, args.route)
    else:
        spatial(args.work, args.architecture, args.seed, args.route)


if __name__ == "__main__":
    main()
