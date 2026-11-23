"""Resolve external imagery for archived formal plotting code explicitly."""

import os
from pathlib import Path
import pandas as pd


def read_table(path, *args, **kwargs):
    """Read a provenance table; resolve only public data asset columns."""
    frame = pd.read_csv(path, *args, **kwargs)
    data_root = os.environ.get("DATA_ROOT")
    for column in (
        "rgb_path",
        "geojson_path",
        "support_path",
        "RGB",
        "support",
    ):
        if column not in frame:
            continue

        def resolve(value):
            if not isinstance(value, str):
                return value
            p = Path(value)
            if p.is_absolute():
                return str(p)
            if data_root is None:
                raise ValueError(
                    "Set DATA_ROOT to the external data directory before plotting imagery panels"
                )
            root = Path(data_root)
            candidate = root / p
            if candidate.is_file():
                return str(candidate)
            candidates = list(root.rglob(p.name))
            if len(candidates) != 1:
                raise ValueError(
                    "Missing or ambiguous figure input: " + p.name
                )
            return str(candidates[0])

        frame[column] = frame[column].map(resolve)
    return frame
