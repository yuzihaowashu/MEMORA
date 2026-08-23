# Paper results

These scripts read saved evaluation JSON under `MEMORA_DATA_ROOT` and
recompute paper-reported numbers. They do not run model inference.

| Script | Purpose |
|--------|---------|
| `reproduce_results.py` | Reproduce 11 curated reported results from saved outputs |
| `eam_qa_metrics.py` | EAM-QA accuracy by participant and on the paper's filtered subset |
| `planning_metrics.py` | OrderExec, KeyObj, PrefAdh, and RGP from saved planning outputs |
| `commands/` | One-command wrappers grouped by benchmark protocol and backbone |

Planning aggregation imports the shared executability predicates from
`src/memora/evaluation/planning/step_checks.py`. All paper-facing planning axes and
their RGP aggregation are defined in `planning_metrics.py`. The released RGP
protocol evaluates successfully parsed, non-empty plans; the output reports
both `n_plans_scored` and `n_empty_plans` for every memory setting.

Example:

```bash
export MEMORA_DATA_ROOT=/path/to/memora_data
source scripts/configure_paths.sh
bash scripts/paper_results/commands/eam_qa/gemma4_26b.sh
```

Each command recomputes its metric and exits nonzero when the result differs
from the registered paper value. Additional arguments are forwarded to
`reproduce_results.py`.

List every registered result and expected value with:

```bash
python3 scripts/paper_results/reproduce_results.py --list
```

Run new inference through `src/memora/evaluation/`. See
[`docs/REPRODUCE.md`](../../docs/REPRODUCE.md) and
[`docs/SETUP.md`](../../docs/SETUP.md).
