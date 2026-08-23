#!/usr/bin/env python3
"""Build a portable list of MEMORA-Planning evaluation runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from memora.evaluation.settings import PUBLIC_EVALUATION_CONDITIONS
from memora_bench.paths import (
    memora_planning_root,
    normalize_pid,
    validate_suite_name,
)

DEFAULT_CONDITIONS = list(PUBLIC_EVALUATION_CONDITIONS)

# Public CLI: --suite replay | generalize. The path resolver maps these to the
# released on-disk suites.
SUITE_CHOICES = ("replay", "generalize")


def _planning_memory_root() -> Path:
    explicit = os.environ.get("PLANNING_MEMORY_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    data_root = Path(os.environ.get("MEMORA_DATA_ROOT", "data")).expanduser().resolve()
    return data_root / "participant_memory" / "memora_paper"


def _planning_suite_file(suite: str, pid: str) -> Path:
    return (
        memora_planning_root()
        / "suites"
        / validate_suite_name(suite)
        / f"{normalize_pid(pid)}.json"
    )

def _spec(
    *,
    pid: str,
    cond: str,
    suite: str,
    output_root: Path,
    memory_root: Path,
    max_iterations: int,
    num_samples: Optional[int],
) -> Dict[str, Any]:
    pid_lc = pid.lower()
    bench = _planning_suite_file(suite, pid)
    out_pid = output_root / pid
    base: Dict[str, Any] = {
        "pid": pid,
        "benchmark_file": str(bench),
        "output_dir": str(out_pid),
        "max_iterations": max_iterations,
        "tag": f"{pid}/{cond}",
    }
    if num_samples is not None:
        base["num_samples"] = num_samples

    if cond == "no_memory":
        return {**base, "condition": "no_memory"}
    if cond == "flat_1d_raw":
        return {
            **base,
            "condition": cond,
            "memory_file": str(
                memory_root / f"flat_1d_raw/flat_1d_raw_{pid_lc}.json"
            ),
            "tag": f"{pid}/flat_1d_raw",
        }
    if cond == "flat_1d_edited":
        return {
            **base,
            "condition": cond,
            "memory_file": str(memory_root / f"flat_1d_edited/flat_1d_edited_{pid_lc}.json"),
            "tag": f"{pid}/flat_1d_edited",
        }
    if cond == "graph_2d_raw":
        return {
            **base,
            "condition": cond,
            "memory_file": str(
                memory_root / f"graph_2d_raw/graph_2d_raw_{pid_lc}.json"
            ),
            "tag": f"{pid}/graph_2d_raw",
        }
    if cond == "graph_2d_edited":
        return {
            **base,
            "condition": cond,
            "memory_file": str(memory_root / f"graph_2d_edited/graph_2d_edited_{pid_lc}.json"),
            "tag": f"{pid}/graph_2d_edited",
        }
    if cond == "memora_episodic":
        return {
            **base,
            "condition": cond,
            "memory_file": str(
                memory_root / f"memora_episodic/participant_memory_{pid_lc}.json"
            ),
            "tag": f"{pid}/memora_episodic",
        }
    if cond == "memora_full":
        return {
            **base,
            "condition": cond,
            "memory_file": str(
                memory_root / f"memora_full/participant_memory_{pid_lc}.json"
            ),
            "tag": f"{pid}/memora_full",
        }
    raise ValueError(f"Unknown condition: {cond!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=SUITE_CHOICES, default="replay")
    ap.add_argument("--pids", nargs="+", default=["P01"])
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--memory-root",
        dest="memory_root",
        default=None,
        help="Root containing the released participant-memory settings",
    )
    ap.add_argument("--max-iterations", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    memory_root = Path(args.memory_root or _planning_memory_root())
    output_root = Path(args.output_root)
    specs: List[Dict[str, Any]] = []
    missing: List[str] = []
    for pid in args.pids:
        for cond in args.conditions:
            spec = _spec(
                pid=pid,
                cond=cond,
                suite=args.suite,
                output_root=output_root,
                memory_root=memory_root,
                max_iterations=args.max_iterations,
                num_samples=args.num_samples,
            )
            for key in ("benchmark_file", "memory_file"):
                p = spec.get(key)
                if p and not Path(p).exists():
                    missing.append(f"  {pid}/{cond}: {key} -> {p}")
            specs.append(spec)

    if missing:
        msg = "Missing inputs:\n" + "\n".join(missing)
        if args.strict:
            raise SystemExit(msg)
        print("WARNING:", msg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(specs, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(specs)} specs -> {out_path}")


if __name__ == "__main__":
    main()
