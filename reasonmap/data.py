"""Dataset access: benchmark samples, metro station tables and map images."""

import base64
import json
import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATIONS_DIR = REPO_ROOT / "stations"
DEFAULT_MAPS_DIR = REPO_ROOT / "maps"

# task -> (HF dataset id, default split)
DATASETS = {
    "reasonmap": ("FSCCS/ReasonMap", "validation"),
    "plus": ("FSCCS/ReasonMap-Plus", "test"),
}

# Questions asked per sample, keyed by the name used in the prediction files.
QUESTION_FIELDS = {
    "reasonmap": {"short": "question_short", "long": "question_long"},
    "plus": {"question": "question"},
}

# Columns that are not needed for scoring (large blobs).
_HEAVY_COLUMNS = ("figure", "json")

_STATION_MARKERS = ("（换乘站）", "（支线起始站）", " (Transfer Station)", " (Branch-starting Station)")


def load_benchmark(task, split=None, cache_dir=None, data_dir=None):
    """Load the HF dataset for ``task`` ("reasonmap" or "plus")."""
    from datasets import load_dataset

    repo_id, default_split = DATASETS[task]
    return load_dataset(data_dir or repo_id, split=split or default_split, cache_dir=cache_dir)


def sample_metadata(sample):
    """Everything in a sample except the image / raw json blobs (stored next to predictions)."""
    return {k: v for k, v in sample.items() if k not in _HEAVY_COLUMNS}


@lru_cache(maxsize=None)
def _load_metro_data_cached(path):
    with open(path, "r", encoding="utf-8") as f:
        metro_data = json.load(f)
    cleaned = {}
    for route_name, stations in metro_data.items():
        names = []
        for s in stations:
            for marker in _STATION_MARKERS:
                s = s.replace(marker, "")
            names.append(s)
        cleaned[route_name] = names
    return cleaned


def load_metro_data(country, city, stations_dir=None):
    """Route name -> ordered list of station names, with transfer / branch markers removed."""
    path = Path(stations_dir or DEFAULT_STATIONS_DIR) / country / f"{city}.json"
    if not path.exists():
        raise FileNotFoundError(f"Station table not found: {path}")
    # copy so callers can't mutate the cache
    return {k: list(v) for k, v in _load_metro_data_cached(str(path)).items()}


def get_map_image_path(sample, task, maps_dir=None):
    """Return a local PNG path for the sample's metro map, materialising it if needed.

    * ReasonMap stores the map as a base64 PNG in ``figure``; it is decoded once per city
      into ``maps/{country}/{city}.png``.
    * ReasonMap-Plus stores a relative path such as ``./maps/portugal/lisboa.png`` which
      points into this repository's ``maps/`` folder.
    """
    maps_dir = Path(maps_dir or DEFAULT_MAPS_DIR)
    figure = sample.get("figure")
    target = maps_dir / sample["country"] / f"{sample['city']}.png"

    if isinstance(figure, str) and figure.lstrip("./").startswith("maps/"):
        rel = Path(figure.lstrip("./")).relative_to("maps")
        path = maps_dir / rel
        if not path.exists():
            raise FileNotFoundError(
                f"Map image {path} not found. ReasonMap-Plus reuses the maps shipped in this repo "
                f"(or produced by running the ReasonMap task once)."
            )
        return str(path)

    if target.exists():
        return str(target)

    if isinstance(figure, str):
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(f".tmp{os.getpid()}")
        tmp.write_bytes(base64.b64decode(figure))
        os.replace(tmp, target)
        return str(target)

    if figure is not None and hasattr(figure, "save"):  # PIL image (HF Image feature)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.save(target)
        return str(target)

    raise ValueError(f"Cannot resolve the map image for sample {sample.get('country')}/{sample.get('city')}")
