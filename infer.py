#!/usr/bin/env python
"""Inference only: run a model on ReasonMap / ReasonMap-Plus and save its raw generations.

Each sample becomes one JSON line in ``outputs/{task}/{model_tag}.jsonl`` holding the
generations plus the sample metadata needed for scoring, so ``score.py`` never has to
touch the model or the images. Re-running the same command resumes where it stopped.

Examples:
    # transformers, in-process
    python infer.py --task reasonmap --backend hf --model Qwen/Qwen2.5-VL-7B-Instruct
    # a running vLLM / SGLang server
    python infer.py --task reasonmap --backend vllm --model Qwen/Qwen2.5-VL-7B-Instruct --num-workers 16
    python infer.py --task plus --backend sglang --model Qwen/Qwen2.5-VL-7B-Instruct
    # closed models used in the paper
    python infer.py --task reasonmap --preset Doubao-115
"""

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from reasonmap import TASKS
from reasonmap.backends import (
    BACKENDS,
    DEFAULT_SERVER_MAX_IMAGE_PIXELS,
    ONLYTEXT_SUFFIX,
    PRESETS,
    build_backend,
    resolve_preset,
)
from reasonmap.backends.openai_api import DEFAULT_MAX_TOKENS
from reasonmap.data import QUESTION_FIELDS, get_map_image_path, load_benchmark, sample_metadata
from reasonmap.models import FAMILIES


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ReasonMap inference (no scoring).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task", choices=TASKS, default="reasonmap")
    p.add_argument("--backend", choices=BACKENDS, default="hf")
    p.add_argument("--preset", help=f"closed-model preset, one of {sorted(PRESETS)} (append {ONLYTEXT_SUFFIX} for text-only)")
    p.add_argument("--model", "--model_path", "--model-path", dest="model",
                   help="HF repo id / local path (hf), served model name (vllm/sglang), or API model id. "
                        f"A trailing '{ONLYTEXT_SUFFIX}' enables --text-only.")
    p.add_argument("--served-model-name", help="model name to send to the server, if different from --model")
    p.add_argument("--model-tag", help="name used for output / result files (default: last component of --model)")
    p.add_argument("--model-family", choices=sorted(FAMILIES),
                   help="prompting / answer-extraction conventions (default: detected from the model name)")
    p.add_argument("--text-only", action="store_true", help="ablation: do not show the map")

    g = p.add_argument_group("generation")
    g.add_argument("--max-tokens", type=int, help=f"max generated tokens (default {DEFAULT_MAX_TOKENS}; presets: provider default; 0 = provider default)")
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-p", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--repetition-penalty", type=float)

    g = p.add_argument_group("server / API backends")
    g.add_argument("--base-url", help="e.g. http://localhost:8000/v1 (vllm) or http://localhost:30000/v1 (sglang)")
    g.add_argument("--api-key")
    g.add_argument("--num-workers", type=int, default=8, help="concurrent requests")
    g.add_argument("--max-image-pixels", type=int,
                   help="downscale maps above this many pixels before sending them to a server / API "
                        f"(default: send the original map; e.g. {DEFAULT_SERVER_MAX_IMAGE_PIXELS} = Qwen2.5-VL's input limit). "
                        "0 disables a preset's limit")
    g.add_argument("--timeout", type=float, default=3600)

    g = p.add_argument_group("transformers backend")
    g.add_argument("--attn-implementation", help="default: flash_attention_2 if installed, else sdpa")

    g = p.add_argument_group("data / output")
    g.add_argument("--cache-dir", "--cache_path", dest="cache_dir", help="HF cache dir for datasets and models")
    g.add_argument("--split", help="dataset split (default: validation for reasonmap, test for plus)")
    g.add_argument("--data-dir", help="load the dataset from a local copy instead of the Hub")
    g.add_argument("--maps-dir", help="where map PNGs are cached (default: ./maps in the repo)")
    g.add_argument("--output-dir", default="outputs")
    g.add_argument("--output", help="explicit output JSONL path (overrides --output-dir)")
    g.add_argument("--start", type=int, default=0, help="first sample index")
    g.add_argument("--limit", type=int, help="number of samples to run (for quick tests)")
    g.add_argument("--overwrite", action="store_true", help="discard existing outputs instead of resuming")
    args = p.parse_args(argv)
    if not args.preset and not args.model:
        p.error("--model or --preset is required")
    return args


def resolve_config(args):
    """Merge preset / backend defaults with CLI flags into one config dict."""
    if args.preset:
        cfg, text_only = resolve_preset(args.preset)
        text_only = text_only or args.text_only
        if args.model:
            cfg["model"] = args.model
        display_name = args.preset
        if text_only and not display_name.endswith(ONLYTEXT_SUFFIX):
            display_name += ONLYTEXT_SUFFIX
        model_tag = args.model_tag or display_name
    else:
        model, text_only = args.model, False
        if model.endswith(ONLYTEXT_SUFFIX):
            model, text_only = model[: -len(ONLYTEXT_SUFFIX)], True
        cfg = {"backend": args.backend, "model": model, "max_tokens": DEFAULT_MAX_TOKENS}
        text_only = text_only or args.text_only
        suffix = ONLYTEXT_SUFFIX if text_only else ""
        model_tag = args.model_tag or (Path(model.rstrip("/")).name + suffix)
        display_name = model + suffix

    for key in ("base_url", "api_key", "model_family", "cache_dir", "attn_implementation", "timeout"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.served_model_name:
        cfg["served_model"] = args.served_model_name
    if args.max_tokens is not None:
        cfg["max_tokens"] = args.max_tokens or None
    if args.max_image_pixels is not None:
        cfg["max_image_pixels"] = args.max_image_pixels or None
    cfg.setdefault("max_tokens", None)
    cfg["sampling"] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }
    cfg.update(task=args.task, text_only=text_only, model_tag=model_tag, display_name=display_name)
    return cfg


def load_done(path):
    """Indices that already have a complete record in ``path``."""
    done = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["idx"])
            except (json.JSONDecodeError, KeyError):
                continue  # a partially written last line from an interrupted run
    return done


def main(argv=None):
    args = parse_args(argv)
    cfg = resolve_config(args)
    task = cfg["task"]

    output = Path(args.output) if args.output else Path(args.output_dir) / task / f"{cfg['model_tag']}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output.exists():
        output.unlink()

    dataset = load_benchmark(task, split=args.split, cache_dir=args.cache_dir, data_dir=args.data_dir)
    stop = len(dataset) if args.limit is None else min(len(dataset), args.start + args.limit)
    done = load_done(output)
    todo = [i for i in range(args.start, stop) if i not in done]
    print(f"[infer] task={task} model={cfg['display_name']} backend={cfg['backend']} -> {output}")
    print(f"[infer] {len(todo)} samples to run ({len(done)} already done)")
    if not todo:
        return 0

    backend = build_backend(cfg)
    print(f"[infer] model family: {backend.family.name}")
    public_cfg = {k: v for k, v in cfg.items() if k != "api_key"}
    with open(output.with_suffix(".config.json"), "w", encoding="utf-8") as f:
        json.dump({**public_cfg, "family": backend.family.name, "argv": sys.argv[1:]}, f, ensure_ascii=False, indent=2)

    questions = QUESTION_FIELDS[task]

    def run_one(idx):
        sample = dataset[idx]
        image_path = None if cfg["text_only"] else get_map_image_path(sample, task, args.maps_dir)
        generations = {key: backend.generate(sample[field], image_path).to_dict() for key, field in questions.items()}
        return {
            "idx": idx,
            "task": task,
            "model": cfg["display_name"],
            "model_tag": cfg["model_tag"],
            "model_id": cfg.get("served_model") or cfg["model"],
            "backend": cfg["backend"],
            "family": backend.family.name,
            "text_only": cfg["text_only"],
            "meta": sample_metadata(sample),
            "generations": generations,
        }

    failures = 0
    start_time = time.time()
    with open(output, "a", encoding="utf-8") as out, tqdm(total=len(todo), desc=f"{task}:{cfg['model_tag']}") as bar:
        def write(record):
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

        if backend.concurrent and args.num_workers > 1:
            with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
                futures = {pool.submit(run_one, idx): idx for idx in todo}
                for future in as_completed(futures):
                    try:
                        write(future.result())
                    except Exception as e:  # noqa: BLE001
                        failures += 1
                        print(f"[infer] sample {futures[future]} failed: {type(e).__name__}: {e}")
                    bar.update(1)
        else:
            for idx in todo:
                try:
                    write(run_one(idx))
                except KeyboardInterrupt:
                    raise
                except Exception as e:  # noqa: BLE001
                    failures += 1
                    print(f"[infer] sample {idx} failed: {type(e).__name__}: {e}")
                    traceback.print_exc()
                bar.update(1)
    backend.close()

    elapsed = time.time() - start_time
    print(f"[infer] finished {len(todo) - failures}/{len(todo)} samples in {elapsed:.0f}s -> {output}")
    if failures:
        print(f"[infer] {failures} samples failed; re-run the same command to retry them.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
