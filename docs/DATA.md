# Data (not in Git)

The Git repository contains the released benchmark definitions, ground truth,
and evaluation code. Large participant-memory files and saved model outputs
used to reproduce the paper tables are distributed separately through the
companion Hugging Face dataset.

MEMORA does **not** redistribute raw EPIC-KITCHENS videos or the full source
annotation tables. Obtain them through the official EPIC-KITCHENS release
channels and follow their license terms:

- EPIC-KITCHENS official site: https://epic-kitchens.github.io/2025
- EPIC-KITCHENS-100 annotations and downloader: https://github.com/epic-kitchens/epic-kitchens-100-annotations

MEMORA benchmark tasks, participant memory, and saved evaluation outputs are
derived from EPIC-KITCHENS-100. They remain subject to the EPIC-KITCHENS
CC BY-NC 4.0 license and terms, including attribution and non-commercial-use
requirements. The repository's MIT license applies to MEMORA source code, not
to these EPIC-KITCHENS-derived files.

## What to Download

| File group | Local path after download | Needed for |
|------------|------------------------------|------------|
| Participant memory files | `$MEMORA_DATA_ROOT/participant_memory/memora_paper/` | EAM-QA and MEMORA-Planning with memory |
| Saved evaluation outputs | `$MEMORA_DATA_ROOT/outputs/` | Reproducing paper tables without GPU inference |
| Benchmark JSON | Shipped in Git under `src/memora_bench/` | Running new evaluations |
| Model weights / cache | `$HF_HOME` or `$MEMORA_DATA_ROOT/hf_models/` | Local vLLM inference |

## Downloaded Layout

After `scripts/data/download.py`, the downloaded package is arranged as:

```text
$MEMORA_DATA_ROOT/
  participant_memory/memora_paper/
    memora_full/
      participant_memory_p01.json
      ...
    memora_episodic/                 # Planning view
    graph_2d_edited/
    graph_2d_raw/
    eam_qa_memora_episodic/          # EAM-QA view
    flat_1d_edited/
    flat_1d_raw/
  outputs/
    eam_qa/
    planning/replay/
    planning/generalize/
```

The evaluation CLI uses the seven paper condition names below. Directory names
usually match those conditions; the one exception is called out explicitly.

| Name | Meaning |
|------|---------|
| `memora_full` | Full MEMORA: four typed stores, online editing, and offline consolidation |
| `memora_episodic` | MEMORA without offline Inferred Knowledge consolidation; used by MEMORA-Planning |
| `eam_qa_memora_episodic` | Download directory containing the EAM-QA memory files used for the `memora_episodic` condition; this is a directory name, not an additional condition |
| `graph_2d_edited`, `graph_2d_raw` | Graph-2D baseline rebuilt after editing / built from raw observations |
| `flat_1d_edited`, `flat_1d_raw` | Flat-1D baseline rebuilt after editing / built from raw observations |
| `no_memory` | No participant-specific memory |

## Download the Data

Set a data root for your machine:

```bash
export MEMORA_DATA_ROOT=/path/to/memora_data
python3 scripts/data/download.py
```

The downloader defaults to the immutable Hugging Face revision used for this
release. Pass `--revision main` only when you intentionally want a newer data
snapshot.

After downloading:

```bash
export MEMORA_DATA_ROOT=/path/to/memora_data
source scripts/configure_paths.sh
bash scripts/data/verify.sh
```

If `scripts/data/verify.sh` reports missing paths, verify that these directories exist:

```text
$MEMORA_DATA_ROOT/participant_memory/memora_paper/
$MEMORA_DATA_ROOT/outputs/
```

The downloadable package includes saved EAM-QA outputs for Gemma-26B,
Qwen-27B, and Gemma-31B, plus saved MEMORA-Planning outputs for all four
released planning backbones.

## Benchmark JSON in Git

These small benchmark files ship with the repository and do not need to be
downloaded from Hugging Face:

| Git path | Description |
|----------|-------------|
| `src/memora_bench/eam_qa/questions/{pid}.json` | EAM-QA questions with abstain choices for each released participant |
| `src/memora_bench/planning/suites/replay/{pid}.json` | Replay planning tasks for each released participant |
| `src/memora_bench/planning/suites/generalize/{pid}.json` | Generalize planning tasks for each released participant |

Downloaded evaluation outputs use the
public benchmark layout, such as
`outputs/planning/generalize/qwen3p6_35b/` and `memora_full`.
Here `qwen3p6` is the path-safe form of `Qwen3.6`.
Saved public planning outputs use clear per-condition filenames such as
`results_memora_full.json`; newly generated runs may instead use timestamped
filenames such as `planning_results_memora_YYYYMMDD_HHMMSS.json`.

Runtime model weights and optional containers are not included in the Hugging
Face data package; see [SETUP.md](SETUP.md).
