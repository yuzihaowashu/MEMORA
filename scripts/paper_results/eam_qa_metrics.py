"""Metrics for EAM-QA (MEMORA-Embodied Memory Assessment)."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from memora.evaluation.settings import PUBLIC_EVALUATION_CONDITIONS

PIDS = [
    "p01", "p02", "p03", "p04", "p06", "p07", "p09", "p11", "p12",
    "p22", "p23", "p25", "p26", "p27", "p28", "p30", "p35", "p37",
]
COND_ORDER = list(PUBLIC_EVALUATION_CONDITIONS)


def load_results(out_root: Path, pid: str, cond_label: str):
    fp = out_root / pid / cond_label / "results_eam_qa.json"
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.load(f).get("results", [])


def qmap(items):
    return {it.get("question_id"): it for it in (items or [])}


def is_committed(item) -> bool:
    p = (item.get("predicted") or item.get("predicted_answer") or "").strip().upper()
    return p in {"A", "B", "C", "D"}


def compute_filtered_results(out_root: Path) -> dict[str, Any]:
    """Pooled F_no_priors ∧ F_M_commit accuracy per condition."""
    out_root = out_root.expanduser().resolve()
    by_pid: dict[str, dict[str, list]] = {pid: {} for pid in PIDS}
    missing: list[tuple[str, str]] = []

    for pid in PIDS:
        for cond in COND_ORDER:
            r = load_results(out_root, pid, cond)
            if r is None:
                missing.append((pid, cond))
            else:
                by_pid[pid][cond] = r

    filter_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for pid in PIDS:
        none_qm = qmap(by_pid[pid].get("no_memory"))
        m_qm = qmap(by_pid[pid].get("memora_full"))
        if not none_qm or not m_qm:
            continue
        elig = {
            qid
            for qid in set(none_qm) & set(m_qm)
            if not none_qm[qid].get("is_correct") and is_committed(m_qm[qid])
        }
        for c in COND_ORDER:
            qm = qmap(by_pid[pid].get(c))
            if not qm:
                continue
            n = sum(1 for q in elig if q in qm)
            k = sum(1 for q in elig if q in qm and qm[q].get("is_correct"))
            filter_counts[c][0] += k
            filter_counts[c][1] += n

    rows: dict[str, dict[str, float | int]] = {}
    for c in COND_ORDER:
        k, n = filter_counts[c]
        if n:
            rows[c] = {"correct": k, "total": n, "accuracy_pct": k / n * 100.0}

    mem = rows.get("memora_full", {})
    mem_acc = float(mem.get("accuracy_pct", 0.0))
    best_base = None
    best_acc = -1.0
    for c in COND_ORDER:
        if c == "memora_full" or c not in rows:
            continue
        acc = float(rows[c]["accuracy_pct"])
        if acc > best_acc:
            best_acc = acc
            best_base = c

    return {
        "out_root": str(out_root),
        "missing_cells": len(missing),
        "by_condition": rows,
        "memora": rows.get("memora_full"),
        "best_baseline": (
            {"condition": best_base, **rows[best_base]}
            if best_base and best_base in rows
            else None
        ),
        "delta_vs_best_baseline_pp": (
            mem_acc - best_acc if best_base else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate EAM-QA results across released participants."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=(
            Path(os.environ["EAM_QA_OUT_ROOT"])
            if os.environ.get("EAM_QA_OUT_ROOT")
            else None
        ),
        help="Backbone output directory, for example outputs/eam_qa/gemma4_26b",
    )
    parser.add_argument("--title", default="EAM-QA across 18 participants")
    args = parser.parse_args()
    if not args.out_root:
        parser.error("--out-root or EAM_QA_OUT_ROOT required")

    out_root = args.out_root.expanduser().resolve()
    print("=" * 110)
    print(args.title)
    print(f"OUT_ROOT={out_root}")
    print("=" * 110)

    by_pid = {pid: {} for pid in PIDS}
    missing = []
    for pid in PIDS:
        for condition in COND_ORDER:
            result = load_results(out_root, pid, condition)
            if result is None:
                missing.append((pid, condition))
            else:
                by_pid[pid][condition] = result
    if missing:
        print(f"\n[!] Missing ({len(missing)} cells):")
        for pid, condition in missing[:20]:
            print(f"    {pid} {condition}")
        if len(missing) > 20:
            print(f"    ... (+ {len(missing) - 20} more)")

    print("\nOverall accuracy by participant")
    print(f"{'PID':<6}  " + "  ".join(f"{c[:14]:>14s}" for c in COND_ORDER))
    for pid in PIDS:
        cells = []
        for condition in COND_ORDER:
            result = by_pid[pid].get(condition)
            if result is None:
                cells.append("    -    ")
                continue
            total = len(result)
            correct = sum(1 for item in result if item.get("is_correct"))
            cells.append(f"{correct / total * 100:>5.1f}% n={total:<3d}")
        print(f"{pid.upper():<6}  " + "  ".join(f"{cell:>14s}" for cell in cells))

    print("\nPaper filtered subset (pooled)")
    filtered = compute_filtered_results(out_root)["by_condition"]
    memora_accuracy = float(filtered.get("memora_full", {}).get("accuracy_pct", 0.0))
    for condition in COND_ORDER:
        row = filtered.get(condition)
        if not row:
            continue
        total = int(row["total"])
        accuracy = float(row["accuracy_pct"])
        marker = "  <- MEMORA" if condition == "memora_full" else ""
        print(
            f"  {condition:<22}  {accuracy:>13.1f}%  {total:>6d}  "
            f"{accuracy - memora_accuracy:>+12.1f}pp{marker}"
        )


if __name__ == "__main__":
    main()
