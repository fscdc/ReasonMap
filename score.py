#!/usr/bin/env python
"""Scoring only: turn the generations written by ``infer.py`` into per-sample results and metrics.

No model, GPU or image is needed. Per-sample results are written to
``results/{country}/{city}/{idx}_{model_tag}.json`` (ReasonMap) or ``results_plus/...``
(ReasonMap-Plus) -- the same layout and fields as the original scripts, so
``cal_metrics.py`` keeps working -- and a summary is printed for the scored models.

Examples:
    python score.py outputs/reasonmap/Qwen2.5-VL-7B-Instruct.jsonl
    python score.py outputs/plus/*.jsonl --summary-json summary_plus.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from reasonmap.data import load_metro_data
from reasonmap.metrics import print_plus_summary, print_reasonmap_summary, summarize_plus, summarize_reasonmap
from reasonmap.scoring import score_reasonmap_record
from reasonmap.scoring_plus import score_plus_record

DEFAULT_RESULTS_DIRS = {"reasonmap": "results", "plus": "results_plus"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Score ReasonMap / ReasonMap-Plus predictions (no inference).")
    p.add_argument("predictions", nargs="+", help="prediction JSONL files written by infer.py")
    p.add_argument("--results-dir", help="where per-sample results go (default: results / results_plus by task)")
    p.add_argument("--stations-dir", help="station tables (default: ./stations in the repo)")
    p.add_argument("--no-write", action="store_true", help="only print the summary, do not write per-sample files")
    p.add_argument("--summary-json", help="also write the summary to this JSON file")
    return p.parse_args(argv)


def load_records(path):
    """Records of one prediction file, de-duplicated by sample index (last one wins)."""
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[score] {path}:{lineno}: skipping malformed line")
                continue
            records[record["idx"]] = record
    return [records[i] for i in sorted(records)]


def score_record(record, stations_dir=None):
    if record["task"] == "plus":
        return score_plus_record(record)
    meta = record["meta"]
    metro_data = load_metro_data(meta["country"], meta["city"], stations_dir)
    return score_reasonmap_record(record, metro_data)


def write_result(results_dir, record, result):
    meta = record["meta"]
    out_dir = Path(results_dir) / meta["country"] / meta["city"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{record['idx']}_{record['model_tag']}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def main(argv=None):
    args = parse_args(argv)
    scored = defaultdict(list)  # task -> [(model_tag, result)]
    for path in args.predictions:
        records = load_records(path)
        if not records:
            print(f"[score] {path}: no records")
            continue
        for record in records:
            task = record["task"]
            result = score_record(record, args.stations_dir)
            scored[task].append((record["model_tag"], result))
            if not args.no_write:
                write_result(args.results_dir or DEFAULT_RESULTS_DIRS[task], record, result)
        print(f"[score] {path}: scored {len(records)} samples")

    summary = {}
    if scored.get("reasonmap"):
        summary["reasonmap"] = summarize_reasonmap(scored["reasonmap"])
        print("\n===== ReasonMap =====")
        print_reasonmap_summary(summary["reasonmap"])
    if scored.get("plus"):
        summary["plus"] = summarize_plus(result for _, result in scored["plus"])
        print("\n===== ReasonMap-Plus =====")
        print_plus_summary(summary["plus"])
    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return 0 if scored else 1


if __name__ == "__main__":
    sys.exit(main())
