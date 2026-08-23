#!/usr/bin/env python3
"""Reproduce a single reported paper number.

Usage:
  python3 scripts/paper_results/reproduce_results.py --list
  python3 scripts/paper_results/reproduce_results.py eam_qa_memora_gemma26b
  python3 scripts/paper_results/reproduce_results.py planning_generalize_qwen35b_memora_full --refresh
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Any, Optional

from eam_qa_metrics import compute_filtered_results
from planning_metrics import aggregate_planning_rgp

_REPO = Path(__file__).resolve().parents[2]

# Curated headline and full-MEMORA results (see this directory's README).
REGISTRY: dict[str, dict[str, Any]] = {
    # --- EAM-QA Table 3: F_no_priors ∧ F_M_commit, MEMORA row (per backbone) ---
    "eam_qa_memora_gemma26b": {
        "paper": "EAM-QA (MEMORA-Embodied Memory Assessment) — Table 3",
        "row": "MEMORA",
        "column": "Gemma-4-26B-A4B-it",
        "metric": "Filtered accuracy (%)",
        "expected": 54.1,
        "tolerance": 0.15,
        "kind": "eam_qa_filtered",
        "out_subdir": "eam_qa/gemma4_26b",
        "condition": "memora_full",
    },
    "eam_qa_memora_qwen27b": {
        "paper": "EAM-QA — Table 3",
        "row": "MEMORA",
        "column": "Qwen3.6-27B",
        "metric": "Filtered accuracy (%)",
        "expected": 69.1,
        "tolerance": 0.15,
        "kind": "eam_qa_filtered",
        "out_subdir": "eam_qa/qwen3p6_27b",
        "condition": "memora_full",
    },
    "eam_qa_memora_gemma31b": {
        "paper": "EAM-QA appendix / Gemma-4-31B results",
        "row": "MEMORA",
        "column": "Gemma-4-31B-it",
        "metric": "Filtered accuracy (%)",
        "expected": 74.5,
        "tolerance": 0.15,
        "kind": "eam_qa_filtered",
        "out_subdir": "eam_qa/gemma4_31b",
        "condition": "memora_full",
    },
    # --- MEMORA-Planning: Robot-Grounded Plan score (RGP), memora_full ---
    "planning_replay_gemma26b_memora_full": {
        "paper": "Planning Replay, 18 participants",
        "row": "memora_full",
        "column": "Gemma-4-26B-A4B-it",
        "metric": "RGP",
        "expected": 0.377,
        "tolerance": 0.002,
        "kind": "planning_rgp",
        "out_subdir": "planning/replay/gemma4_26b",
        "condition": "memora_full",
    },
    "planning_replay_gemma31b_memora_full": {
        "paper": "Planning Replay, 18 participants",
        "row": "memora_full",
        "column": "Gemma-4-31B-it",
        "metric": "RGP",
        "expected": 0.375,
        "tolerance": 0.002,
        "kind": "planning_rgp",
        "out_subdir": "planning/replay/gemma4_31b",
        "condition": "memora_full",
    },
    "planning_replay_qwen27b_memora_full": {
        "paper": "Planning Replay, 18 participants",
        "row": "memora_full",
        "column": "Qwen3.6-27B",
        "metric": "RGP",
        "expected": 0.334,
        "tolerance": 0.002,
        "kind": "planning_rgp",
        "out_subdir": "planning/replay/qwen3p6_27b",
        "condition": "memora_full",
    },
    "planning_replay_qwen35b_memora_full": {
        "paper": "Planning Replay, 18 participants",
        "row": "memora_full",
        "column": "Qwen3.6-35B-A3B",
        "metric": "RGP",
        "expected": 0.338,
        "tolerance": 0.002,
        "kind": "planning_rgp",
        "out_subdir": "planning/replay/qwen3p6_35b",
        "condition": "memora_full",
    },
    "planning_generalize_gemma26b_memora_full": {
        "paper": "Planning Generalize, 18 participants",
        "row": "memora_full",
        "column": "Gemma-4-26B-A4B-it",
        "metric": "RGP",
        "expected": 0.504,
        "tolerance": 0.002,
        "kind": "planning_rgp",
        "out_subdir": "planning/generalize/gemma4_26b",
        "condition": "memora_full",
    },
    "planning_generalize_gemma31b_memora_full": {
        "paper": "Planning Generalize, 18 participants",
        "row": "memora_full",
        "column": "Gemma-4-31B-it",
        "metric": "RGP",
        "expected": 0.532,
        "tolerance": 0.002,
        "kind": "planning_rgp",
        "out_subdir": "planning/generalize/gemma4_31b",
        "condition": "memora_full",
    },
    "planning_generalize_qwen27b_memora_full": {
        "paper": "Planning Generalize, 18 participants",
        "row": "memora_full",
        "column": "Qwen3.6-27B",
        "metric": "RGP",
        "expected": 0.393,
        "tolerance": 0.002,
        "kind": "planning_rgp",
        "out_subdir": "planning/generalize/qwen3p6_27b",
        "condition": "memora_full",
    },
    "planning_generalize_qwen35b_memora_full": {
        "paper": "Planning Generalize, 18 participants",
        "row": "memora_full",
        "column": "Qwen3.6-35B-A3B",
        "metric": "RGP",
        "expected": 0.450,
        "tolerance": 0.002,
        "kind": "planning_rgp",
        "out_subdir": "planning/generalize/qwen3p6_35b",
        "condition": "memora_full",
    },
}

def _data_root() -> Path:
    return Path(
        os.environ.get("MEMORA_DATA_ROOT")
        or (_REPO / "data")
    ).expanduser()


def _resolve_output_dir(data_root: Path, subdir: str) -> Path:
    subdir = subdir.strip("/")
    if subdir.startswith("outputs/"):
        return (data_root / subdir).resolve()
    return (data_root / "outputs" / subdir).resolve()


def _entry_data_available(meta: dict[str, Any]) -> bool:
    data_root = _data_root()
    out_root = _resolve_output_dir(data_root, meta["out_subdir"])
    if not out_root.exists():
        return False
    if meta["kind"] == "eam_qa_filtered":
        return any(out_root.rglob("results_eam_qa.json"))
    if meta["kind"] == "planning_rgp":
        return (
            any(out_root.glob("P*/planning_results_*.json"))
            or any(out_root.glob("P*/results_*.json"))
            or (out_root / "planning_metrics.json").exists()
        )
    return True


def _print_header(entry_id: str, meta: dict[str, Any]) -> None:
    print("=" * 72)
    print(f"Paper number: {entry_id}")
    print(f"  Table : {meta.get('paper')}")
    print(f"  Row   : {meta.get('row')}")
    print(f"  Col   : {meta.get('column')}")
    print(f"  Metric: {meta.get('metric')}")
    if meta.get("expected") is not None:
        print(f"  Expected: {meta['expected']}")
    print("=" * 72)


def _check_expected(got: float, expected: Optional[float], tol: float) -> bool:
    if expected is None:
        return True
    ok = abs(got - expected) <= tol
    mark = "OK" if ok else "MISMATCH"
    print(f"\n[{mark}] got={got:.4f}  expected={expected}  (±{tol})")
    return ok


def run_entry(entry_id: str, *, refresh: bool = False, strict: bool = False) -> int:
    if entry_id not in REGISTRY:
        print(f"Unknown entry: {entry_id}", file=sys.stderr)
        return 2
    meta = REGISTRY[entry_id]
    _print_header(entry_id, meta)
    data_root = _data_root()
    out_root = _resolve_output_dir(data_root, meta["out_subdir"])
    if not data_root.exists():
        print(f"ERROR: data root missing: {data_root}", file=sys.stderr)
        return 1

    kind = meta["kind"]
    ok = True

    if kind == "eam_qa_filtered":
        results = compute_filtered_results(out_root)
        cond = meta["condition"]
        row = results["by_condition"].get(cond)
        if not row:
            print(f"ERROR: no results for condition {cond}", file=sys.stderr)
            return 1
        acc = float(row["accuracy_pct"])
        n = int(row["total"])
        print(f"OUT_ROOT = {out_root}")
        print(f"Filter subset n = {n}")
        print(f"EAM-QA MEMORA (memora_full) = {acc:.1f}%")
        if results.get("best_baseline"):
            bb = results["best_baseline"]
            print(
                f"Best baseline ({bb['condition']}) = {bb['accuracy_pct']:.1f}%  "
                f"(Δ = {results['delta_vs_best_baseline_pp']:+.1f}pp)"
            )
        ok = _check_expected(acc, meta.get("expected"), meta.get("tolerance", 0.2))

    elif kind == "planning_rgp":
        # Component scripts print full diagnostic tables.  A single-number
        # reproduction command keeps those details quiet and reports only the
        # paper-facing aggregate below.
        with contextlib.redirect_stdout(io.StringIO()):
            agg = aggregate_planning_rgp(
                out_root,
                condition=meta.get("condition"),
                refresh=refresh,
            )
        print(f"OUT_ROOT = {out_root}")
        cond = meta["condition"]
        val = agg.get("rgp") or agg.get("panel_rgp", {}).get(cond)
        if val is None:
            print(f"ERROR: no RGP for {cond}", file=sys.stderr)
            return 1
        print(f"{cond} RGP = {val:.3f}")
        if agg.get("best_baseline"):
            bb = agg["best_baseline"]
            print(f"Best baseline ({bb['condition']}) = {bb['rgp']:.3f}")
        ok = _check_expected(float(val), meta.get("expected"), meta.get("tolerance", 0.02))

    else:
        print(f"ERROR: unknown kind {kind}", file=sys.stderr)
        return 1

    return 0 if ok or not strict else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("entry_id", nargs="?", help="Registry id (see --list)")
    ap.add_argument("--list", action="store_true", help="Print all paper-number entries (human-readable)")
    ap.add_argument("--ids", action="store_true",
                    help="Print only paper-number ids, one per line (for scripting)")
    ap.add_argument("--available", action="store_true",
                    help="With --ids or --list, show only entries whose output directory is present")
    ap.add_argument(
        "--refresh", action="store_true", help="Recompute planning_metrics.json"
    )
    ap.add_argument("--strict", action="store_true", help="Exit 1 if value ≠ expected")
    args = ap.parse_args()

    if args.ids:
        for eid in REGISTRY:
            if args.available and not _entry_data_available(REGISTRY[eid]):
                continue
            print(eid)
        return

    if args.list:
        for eid, meta in REGISTRY.items():
            if args.available and not _entry_data_available(meta):
                continue
            exp = meta.get("expected")
            exp_s = f" → {exp}" if exp is not None else ""
            print(f"  {eid:<42} {meta.get('paper')} / {meta.get('row')}{exp_s}")
        return

    if not args.entry_id:
        ap.error("entry_id required (or use --list)")

    sys.exit(run_entry(args.entry_id, refresh=args.refresh, strict=args.strict))


if __name__ == "__main__":
    main()
