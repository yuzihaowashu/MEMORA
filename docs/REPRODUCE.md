# Reproduce MEMORA paper numbers

**Prerequisites:** downloaded data under `MEMORA_DATA_ROOT` ([DATA.md](DATA.md)) and an active analysis environment ([SETUP.md](SETUP.md)).

For **running new inference** (GPU or venv), see [SETUP.md](SETUP.md).

```bash
cd MEMORA
export MEMORA_DATA_ROOT=/path/to/memora_data
source scripts/configure_paths.sh
```

## Minimum sanity run

After downloading the Hugging Face data package, run:

```bash
bash scripts/data/verify.sh
bash scripts/paper_results/commands/eam_qa/gemma4_26b.sh
bash scripts/paper_results/commands/planning_generalize/qwen3_6_35b.sh --refresh
```

This checks the downloaded paths, recomputes one EAM-QA paper row, and
recomputes one MEMORA-Planning Generalize row from saved JSON outputs.

## A. Re-aggregate existing runs (no GPU)

### One script per registered paper result (recommended)

See [the paper-results index](../scripts/paper_results/README.md). Example:

```bash
source scripts/configure_paths.sh
bash scripts/paper_results/commands/eam_qa/qwen3_6_27b.sh
```

### EAM-QA memory assessment across models

```bash
python3 scripts/paper_results/eam_qa_metrics.py \
  --out-root "$MEMORA_DATA_ROOT/outputs/eam_qa/gemma4_26b" \
  --title "Gemma-4-26B EAM-QA"

python3 scripts/paper_results/eam_qa_metrics.py \
  --out-root "$MEMORA_DATA_ROOT/outputs/eam_qa/qwen3p6_27b" \
  --title "Qwen3.6-27B EAM-QA"

python3 scripts/paper_results/eam_qa_metrics.py \
  --out-root "$MEMORA_DATA_ROOT/outputs/eam_qa/gemma4_31b" \
  --title "Gemma-4-31B EAM-QA"

```

The Hugging Face package includes saved EAM-QA outputs for Gemma-26B,
Qwen3.6-27B, and Gemma-31B.

### Planning metrics

```bash
python3 scripts/paper_results/planning_metrics.py \
  "$MEMORA_DATA_ROOT/outputs/planning/replay/qwen3p6_35b"
```

This reports OrderExec, KeyObj, PrefAdh, and their RGP aggregate from saved
planning outputs. The released protocol scores successfully parsed, non-empty
plans and reports empty-plan counts separately. It does not call an LLM.

## B. Re-run inference (GPU)

Use the CLIs in [SETUP.md](SETUP.md) §3–4, or a task list produced by:

| What you need | How to get it |
|---------------|---------------|
| EAM-QA task list | Write a JSON array with `benchmark_file`, `condition`, `memory_file`, and `output` fields |
| MEMORA-Planning task list | Build with `scripts/build_evaluation_runs.py` |
| Released paper memory files | Download from Hugging Face per [DATA.md](DATA.md) |
| New memory files from your own videos | Follow [PIPELINE.md](PIPELINE.md) |

Example:

```bash
memora-eam-qa run-multi \
  --model google/gemma-4-26B-A4B-it \
  --task-list /path/to/tasks.json \
  --tensor-parallel-size 2 \
  --require-e5 \
  --skip-existing
```

## C. Regenerate benchmarks

See [scripts/benchmark_construction/README.md](../scripts/benchmark_construction/README.md):

- **EAM-QA:** curated benchmark JSON is released directly under
  `src/memora_bench/eam_qa/questions/`; construction is documented in
  the paper appendix.
- **Planning:** `extract_replay_candidates.py` → `build_replay_suite.py`; `generate_generalize_candidates.py` → `build_generalize_suite.py`

Evaluation task lists are independent of benchmark construction and are built
with `scripts/build_evaluation_runs.py`.
