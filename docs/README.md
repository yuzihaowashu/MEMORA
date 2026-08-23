# MEMORA documentation

Start with the [repository README](../README.md) for setup and a quick
reproduction path.

## Documents

| Document | When to read |
|----------|--------------|
| [SETUP.md](SETUP.md) | First-time environment setup, data paths, and running evaluation CLIs |
| [PIPELINE.md](PIPELINE.md) | Form participant memory from video and use it for planning |
| [REPRODUCE.md](REPRODUCE.md) | Recomputing paper tables from saved evaluation JSON |
| [DATA.md](DATA.md) | Downloading the Hugging Face data package |
| [MEMORA-Bench README](../src/memora_bench/README.md) | Benchmark protocols, files, and sizes |

## Project website

The project website is kept separately from these researcher guides:

- entry point: `website/index.html`
- benchmark explorer: `website/benchmark-explorer.html`
- site assets: `website/assets/`
- publishing source: `website/`, deployed by GitHub Actions

The explorer data is generated from the released benchmark JSON. Rebuild it
after changing benchmark files:

```bash
python3 scripts/website/build_benchmark_explorer.py
```

For a local preview from the repository root:

```bash
python3 -m http.server 8765 --directory website
```

## Related indexes

| Path | Contents |
|------|----------|
| [Paper-results scripts](../scripts/paper_results/README.md) | Metrics and commands for reported headline results |
| [../scripts/benchmark_construction/README.md](../scripts/benchmark_construction/README.md) | Optional MEMORA-Planning suite construction |
| [../src/memora/README.md](../src/memora/README.md) | Core source-tree layout |

## Typical paths

**Reproduce saved paper numbers (no GPU):**

1. [DATA.md](DATA.md) — download the Hugging Face data
2. [SETUP.md](SETUP.md) — create the analysis venv
3. [REPRODUCE.md](REPRODUCE.md) — run the paper-results commands

**Run new evaluations (GPU):**

1. [SETUP.md](SETUP.md) — `bash scripts/setup_environment.sh gpu`
2. [MEMORA-Bench README](../src/memora_bench/README.md) — benchmark JSON locations in Git
3. [SETUP.md](SETUP.md) §3–4 — EAM-QA and planning CLIs

**Build memory from new video (GPU):**

1. [PIPELINE.md](PIPELINE.md) — run the Segment Encoder
2. [PIPELINE.md](PIPELINE.md) — run the Memory Editor and offline consolidation
3. [PIPELINE.md](PIPELINE.md) — use the resulting memory for planning
