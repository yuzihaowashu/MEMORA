"""Path helpers for the released MEMORA-Planning suites."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

PLANNING_SUITES = frozenset({"replay", "generalize"})


def memora_bench_dir(source_root: os.PathLike[str] | str | None = None) -> Path:
    """Return the root of the installed or source-tree MEMORA-Bench package."""
    explicit = os.environ.get("MEMORA_BENCH_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if source_root is not None:
        return (Path(source_root).expanduser().resolve() / "memora_bench").resolve()
    return Path(__file__).resolve().parent


def normalize_pid(pid: str) -> str:
    """P01, p01, 01 → p01 (lowercase with leading p)."""
    p = pid.strip().upper()
    if not p.startswith("P"):
        p = "P" + p
    return p.lower()


def memora_planning_root(source_root: os.PathLike[str] | str | None = None) -> Path:
    """Return the released MEMORA-Planning data root."""
    explicit = os.environ.get("MEMORA_PLANNING_BENCH_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return memora_bench_dir(source_root) / "planning"


def validate_suite_name(suite: str) -> str:
    """Return a normalized public suite name or raise a clear error."""
    key = suite.strip().lower()
    if key not in PLANNING_SUITES:
        raise ValueError(
            f"Unknown planning suite {suite!r}; choose from {sorted(PLANNING_SUITES)}"
        )
    return key


def glob_planning_suite_patterns(
    source_root: os.PathLike[str] | str | None = None,
) -> List[str]:
    root = memora_planning_root(source_root)
    return [
        str(root / "suites/replay/p*.json"),
        str(root / "suites/generalize/p*.json"),
    ]
