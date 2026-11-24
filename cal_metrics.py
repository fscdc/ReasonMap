"""Aggregate every per-sample result file under results/ and results_plus/ (all models)."""

import argparse

from reasonmap.metrics import (
    REASONMAP_EVAL_CITIES,
    iter_result_files,
    print_plus_summary,
    print_reasonmap_summary,
    summarize_plus,
    summarize_reasonmap,
)


def analyze_results(root_dir):
    """ReasonMap metrics per model (model name taken from the result file name)."""
    print_reasonmap_summary(summarize_reasonmap(iter_result_files(root_dir, cities=REASONMAP_EVAL_CITIES)))


def analyze_results_v2_weighted(root_dir):
    """ReasonMap-Plus accuracy (plain and difficulty-weighted) per model and question type."""
    print_plus_summary(summarize_plus(data for _, data in iter_result_files(root_dir)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--results-plus-dir", default="./results_plus")
    args = parser.parse_args()
    analyze_results(args.results_dir)
    analyze_results_v2_weighted(args.results_plus_dir)
