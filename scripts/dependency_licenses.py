"""Report the installed distribution metadata without granting new licenses."""

from importlib.metadata import metadata, version
from pathlib import Path
import argparse

PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "PyYAML",
    "torch",
    "rasterio",
    "geopandas",
    "shapely",
    "pyproj",
    "matplotlib",
    "Pillow",
    "pytest",
    "pyogrio",
)


def main() -> None:
    """Write a compact, source-attributed dependency license inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = [
        "# Installed dependency license metadata",
        "",
        "This records the accepted environment's own distribution metadata. "
        "Bundled binaries may include additional notices; consult each upstream "
        "distribution. No third-party dependency is relicensed by this project.",
        "",
        "| Distribution | Version | Reported license |",
        "|---|---|---|",
    ]
    for package in PACKAGES:
        meta = metadata(package)
        declared = meta.get("License-Expression") or meta.get("License")
        if not declared:
            declared = (
                "; ".join(
                    v
                    for v in meta.get_all("Classifier", [])
                    if v.startswith("License ::")
                )
                or "Not declared in this field; inspect upstream notices"
            )
        declared = declared.strip().splitlines()[0].replace("|", "/")
        lines.append(
            f"| [{package}](https://pypi.org/project/{package}/) "
            f"| {version(package)} | {declared} |"
        )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
