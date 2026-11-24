"""Aggregate per-sample results into the benchmark metrics (shared by score.py / cal_metrics.py)."""

import json
import os
from collections import defaultdict

# Cities of the ReasonMap validation split.
REASONMAP_EVAL_CITIES = (
    "los_angeles", "miami", "toronto", "singapore", "auckland",
    "rome", "lisboa", "beijing", "hangzhou", "budapest", "geneva",
)

PLUS_DIFFICULTY_WEIGHTS = {"easy": 1.0, "middle": 1.5, "hard": 2.0}
PLUS_TYPE_WEIGHTS = {"Counting1": 1.0, "Counting2": 1.0, "Counting3": 1.0, "TorF1": 1.0, "TorF2": 1.0}

_REASONMAP_FIELDS = {
    "acc_short": "acc short",
    "acc_long": "acc long",
    "token_short": "short token count",
    "token_long": "long token count",
    "via_stops_score": "save via stops score",
    "num_via_stop_score": "save num via stop score",
    "map_score_short": "best map api score short",
    "map_score_long": "best map api score long",
}


def _safe_div(n, d):
    return n / d if d else 0.0


# ----------------------------------------------------------------------------- loading


def iter_result_files(root_dir, cities=None):
    """Yield (model_shortname, result_dict) from ``root_dir/{country}/{city}/{idx}_{model}.json``."""
    for dirpath, _, filenames in os.walk(root_dir):
        parts = os.path.relpath(dirpath, root_dir).split(os.sep)
        if len(parts) < 2:
            continue
        if cities is not None and parts[1] not in cities:
            continue
        for filename in sorted(filenames):
            if not filename.endswith(".json") or "_" not in filename:
                continue
            _, model_shortname = filename[:-5].split("_", 1)
            path = os.path.join(dirpath, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    yield model_shortname, json.load(f)
            except Exception as e:  # keep going on a corrupt file
                print(f"❌ Error processing {filename} in {dirpath}: {e}")


# ----------------------------------------------------------------------------- ReasonMap


def summarize_reasonmap(pairs):
    """``pairs``: iterable of (model_name, result_dict). Returns {model: {metric: value}}."""
    sums = defaultdict(lambda: defaultdict(float))
    for model, data in pairs:
        stats = sums[model]
        stats["count"] += 1
        for key, field in _REASONMAP_FIELDS.items():
            stats[key] += data.get(field, 0) or 0
    summary = {}
    for model, stats in sums.items():
        n = stats["count"]
        summary[model] = {"count": int(n), **{k: _safe_div(stats[k], n) for k in _REASONMAP_FIELDS}}
    return summary


def print_reasonmap_summary(summary):
    if not summary:
        print("⚠️ 没有找到任何符合格式的JSON文件")
        return
    for model, s in summary.items():
        print(f"📊 Model: {model}")
        print(f"  Total files: {s['count']}")
        print(f"  Average acc_short: {s['acc_short']:.4f}")
        print(f"  Average acc_long: {s['acc_long']:.4f}")
        print(f"  Average token_short: {s['token_short']:.2f}")
        print(f"  Average token_long: {s['token_long']:.2f}")
        print(f"  Average save via stops score: {s['via_stops_score']:.4f}")
        print(f"  Average save num via stop score: {s['num_via_stop_score']:.4f}")
        print(f"  Average best map api score short: {s['map_score_short']:.4f}")
        print(f"  Average best map api score long: {s['map_score_long']:.4f}")
        print()


# ----------------------------------------------------------------------------- ReasonMap-Plus


def _plus_group(q_type):
    ql = q_type.lower()
    if ql.startswith("count"):
        return ["counting"]
    if ql == "torf1":
        return ["torf1", "torf"]
    if ql == "torf2":
        return ["torf2", "torf"]
    if ql.startswith("torf"):
        return ["torf"]
    return []


def summarize_plus(results):
    """``results``: iterable of Plus result dicts (grouped by their "model" field)."""
    def new_group():
        return {"count": 0, "sum": 0.0, "w_sum": 0.0, "w_total": 0.0}

    stats = defaultdict(lambda: {
        "count": 0, "sum": 0.0, "w_sum": 0.0, "w_total": 0.0, "token_sum": 0.0,
        "groups": {g: new_group() for g in ("counting", "torf", "torf1", "torf2")},
    })
    for data in results:
        model = data.get("model", "unknown_model")
        acc = float(data.get("acc short", 0.0))
        q_type = str(data.get("type", "TorF"))
        difficulty = str(data.get("difficulty city", "easy")).lower()
        weight = PLUS_DIFFICULTY_WEIGHTS.get(difficulty, 1.0) * PLUS_TYPE_WEIGHTS.get(q_type, 1.0)

        s = stats[model]
        s["count"] += 1
        s["sum"] += acc
        s["w_sum"] += acc * weight
        s["w_total"] += weight
        s["token_sum"] += float(data.get("token count", 0) or 0)
        for g in _plus_group(q_type):
            grp = s["groups"][g]
            grp["count"] += 1
            grp["sum"] += acc
            grp["w_sum"] += acc * weight
            grp["w_total"] += weight

    summary = {}
    for model, s in stats.items():
        summary[model] = {
            "count": s["count"],
            "acc": _safe_div(s["sum"], s["count"]),
            "weighted_acc": _safe_div(s["w_sum"], s["w_total"]),
            "token_count": _safe_div(s["token_sum"], s["count"]),
            "groups": {
                g: {
                    "count": grp["count"],
                    "acc": _safe_div(grp["sum"], grp["count"]),
                    "weighted_acc": _safe_div(grp["w_sum"], grp["w_total"]),
                }
                for g, grp in s["groups"].items()
            },
        }
    return summary


def print_plus_summary(summary):
    if not summary:
        print("⚠️ 没有找到任何符合格式的JSON文件")
        return
    for model, s in summary.items():
        print(f"📊 Model: {model}")
        print(f"  Total files: {s['count']}")
        print(f"  Average acc_short: {s['acc']:.4f}")
        print(f"  Weighted acc_short: {s['weighted_acc']:.4f}")
        print(f"  Average token count: {s['token_count']:.2f}")
        for g, label in (("counting", "Counting"), ("torf", "TorF"), ("torf1", "TorF1"), ("torf2", "TorF2")):
            grp = s["groups"][g]
            print(f"  [{label}]")
            print(f"    files: {grp['count']}")
            print(f"    avg acc_short: {grp['acc']:.4f}")
            print(f"    weighted acc_short: {grp['weighted_acc']:.4f}")
        print()
