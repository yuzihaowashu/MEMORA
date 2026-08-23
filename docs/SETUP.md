# Environment and running experiments

MEMORA is **runnable from this repo alone**: inference and analysis are driven by Python entry points under `src/memora/`. You choose where data and weights live.

## 1. Environment (venv)

| What you need | Command |
|---------------|---------|
| Analyze saved outputs (no GPU) | `bash scripts/setup_environment.sh analysis` |
| Local GPU + vLLM 0.19.1 | `bash scripts/setup_environment.sh gpu` |

The setup script requires Python 3.10-3.12 and checks the interpreter before
creating `.venv`. When the system `python3` is newer, choose one explicitly:

```bash
PYTHON_BIN=python3.12 bash scripts/setup_environment.sh analysis
```

### Dependencies

**Python:** 3.10–3.12 for analysis; the paper GPU container uses **Python 3.12.13**.

| Mode | Install | Notes |
|------|---------|-------|
| Saved-output analysis | `bash scripts/setup_environment.sh analysis` | Base package + `analysis` extra |
| Local GPU inference | `bash scripts/setup_environment.sh gpu` | Base package + `analysis,gpu` extras |

Canonical dependency groups and version pins live in **`pyproject.toml`**.

#### Base and `analysis` — no GPU

| Package | Version |
|---------|---------|
| `numpy` | **2.2.6** |
| `openai` | **2.32.0** (optional API scripts) |
| `pandas` | >= 2.0 |
| `huggingface_hub` | **1.11.0** |

#### `[gpu]` — paper container stack

Built from **`docker://vllm/vllm-openai:v0.19.1`**. Verified inside the paper
Apptainer image `vllm_v0.19.1.sif` (Python 3.12.13).

| Package | Version |
|---------|---------|
| Python (in container) | **3.12.13** |
| `vllm` | **0.19.1** |
| `torch` | **2.10.0+cu129** (bundled in container; installed via `vllm` on local GPU) |
| `transformers` | **5.5.4** |
| `accelerate` | **1.13.0** |
| `tokenizers` | **0.22.2** |
| `safetensors` | **0.7.0** |
| `pillow` | **12.2.0** |
| `tqdm` | **4.67.3** |
| `sentence-transformers` | >= 3.0 (bootstrapped at job time on cluster; in venv for local GPU) |
| `scikit-learn` | >= 1.3 |
| `qwen-omni-utils` | Current release; required by the paper Segment Encoder |
| `av`, `decord` | Video decoding fallbacks used by the Segment Encoder |

The Segment Encoder also requires the `ffmpeg` and `ffprobe` executables on
`PATH`. These are system packages rather than Python dependencies.

If GPU install fails, match PyTorch to your CUDA driver first, then follow the
[vLLM 0.19.1 install guide](https://docs.vllm.ai/en/v0.19.1/getting_started/installation.html).

Manual installs from the repository root:

```bash
pip install -e ".[analysis]"       # metrics and benchmark analysis
pip install -e ".[analysis,gpu]"   # full local paper stack on a CUDA machine
```

### Repo and data paths (after venv is active)

Set the data root for **your** machine before sourcing `scripts/configure_paths.sh`
(defaults to `./data` if unset):

```bash
export MEMORA_DATA_ROOT=/path/to/memora_data   # participant memory, saved outputs, model cache
export HF_HOME=/path/to/hf_cache
export HF_TOKEN=<your_huggingface_token>   # gated models (Gemma, etc.)
source scripts/configure_paths.sh
```

The configuration script derives source and benchmark paths from the repository
location and defaults participant memory to
`$MEMORA_DATA_ROOT/participant_memory/memora_paper`. Override individual
variables before sourcing only when using a different layout.

## 2. Data layout

Download participant memory files and saved evaluation outputs from Hugging Face ([DATA.md](DATA.md)) or use existing JSON under:

```text
$MEMORA_DATA_ROOT/
  participant_memory/memora_paper/...
  outputs/eam_qa/.../results_eam_qa.json
  outputs/planning/.../P01/results_memora_full.json
```

Download the Hugging Face package directly into this layout with:

```bash
python3 scripts/data/download.py
```

Benchmark definitions ship as the sibling `memora_bench` package:

```text
$MEMORA_BENCH_DIR/
  eam_qa/questions/p01.json
  planning/suites/replay/p01.json
  planning/suites/generalize/p01.json
```

The default planning-memory root from `scripts/configure_paths.sh` is
`$MEMORA_DATA_ROOT/participant_memory/memora_paper`.

## 3. Run EAM-QA (GPU)

### One EAM-QA run

```bash
memora-eam-qa run \
  --model google/gemma-4-26B-A4B-it \
  --benchmark-file "$EAM_QA_BENCH_DIR/p01.json" \
  --condition no_memory \
  --output "$MEMORA_DATA_ROOT/outputs/quickstart/no_memory_p01/results_eam_qa.json" \
  --tensor-parallel-size 2 \
  --max-model-len 65536
```

### Multiple EAM-QA runs with one model load

Task list format: JSON array of runs with `benchmark_file`, `condition`,
`memory_file`, and `output`, plus optional `tag` and `participant_id`. Here
`condition` uses the same paper condition names accepted by the single-run CLI.
Paths in task-list JSON may use environment variables such as
`${MEMORA_BENCH_DIR}` and `${MEMORA_DATA_ROOT}`.

```bash
cat > /tmp/eam_qa_task_list.json <<'JSON'
[
  {
    "benchmark_file": "${MEMORA_BENCH_DIR}/eam_qa/questions/p01.json",
    "condition": "no_memory",
    "memory_file": "",
    "output": "${MEMORA_DATA_ROOT}/outputs/quickstart/no_memory_p01/results_eam_qa.json",
    "tag": "quickstart_no_memory_p01",
    "participant_id": "P01"
  }
]
JSON

memora-eam-qa run-multi \
  --model google/gemma-4-26B-A4B-it \
  --task-list /tmp/eam_qa_task_list.json \
  --tensor-parallel-size 2 \
  --skip-existing
```

Generate planning task lists with `scripts/build_evaluation_runs.py`; for EAM-QA, create JSON task lists following the schema above.

## 4. Run Planning (GPU)

For one released MEMORA-Planning task, use a participant-specific memory file:

```bash
memora-plan \
  --model Qwen/Qwen3.6-35B-A3B \
  --benchmark-file "$MEMORA_PLANNING_BENCH_DIR/suites/replay/p01.json" \
  --memory-file "$PLANNING_MEMORY_ROOT/memora_full/participant_memory_p01.json" \
  --condition memora_full \
  --require-e5 \
  --output-dir "$MEMORA_DATA_ROOT/outputs/planning_quickstart/P01"
```

Here, `memora_full` is the paper/data setting for consolidated MEMORA memory.
The same public setting name is used by the CLI, memory directory, and saved
result filename.

Paper-protocol inference uses E5 retrieval. Keep `--require-e5` enabled for
strict reproduction; without it, the runtime may use keyword retrieval when
the embedding model cannot be loaded.

Multi-task: `--task-list path/to/specs.json` (same idea as EAM-QA). Build lists with `scripts/build_evaluation_runs.py`, or write JSON by hand using the same fields:

```bash
python3 scripts/build_evaluation_runs.py \
  --suite replay \
  --pids P01 P02 \
  --conditions memora_full \
  --output-root "$MEMORA_DATA_ROOT/outputs/planning_quickstart" \
  --out /tmp/planning_quickstart.json \
  --strict
```

## 5. Analysis (no GPU)

After EAM-QA and planning JSON exist:

```bash
source scripts/configure_paths.sh
bash scripts/data/verify.sh

python3 scripts/paper_results/eam_qa_metrics.py \
  --out-root "$MEMORA_DATA_ROOT/outputs/eam_qa/gemma4_26b"

python3 scripts/paper_results/planning_metrics.py \
  "$MEMORA_DATA_ROOT/outputs/planning/replay/qwen3p6_35b"
```

The analysis scripts accept both public saved planning outputs
(`P*/results_<condition>.json`) and newly generated timestamped outputs
(`P*/planning_results_<condition>_<timestamp>.json`).

Details: [REPRODUCE.md](REPRODUCE.md).

### Paper-reported numbers (one script per registered result)

```bash
bash scripts/paper_results/commands/eam_qa/gemma4_26b.sh
```

Index: [paper-results README](../scripts/paper_results/README.md).

Optional planning-suite construction: [scripts/benchmark_construction/README.md](../scripts/benchmark_construction/README.md).

To build participant memory from raw video before running these evaluations,
follow [PIPELINE.md](PIPELINE.md).
