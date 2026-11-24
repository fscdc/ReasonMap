"""ReasonMap evaluation toolkit.

The package is split into two independent halves:

* inference (`reasonmap.backends`, driven by `infer.py`): turns benchmark samples into raw
  model generations and writes them to a JSONL file;
* scoring (`reasonmap.scoring`, `reasonmap.scoring_plus`, `reasonmap.metrics`, driven by
  `score.py` / `cal_metrics.py`): turns those generations into per-sample scores and
  aggregated metrics.

The scoring half is pure Python (no torch / transformers), so it can be re-run anywhere.
"""

TASKS = ("reasonmap", "plus")
