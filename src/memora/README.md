# MEMORA (`src/memora`)

The source tree follows the lifecycle in the paper:

```text
pipeline/segment_encoder/
  video -> typed segment observations

pipeline/memory_editor/
  observations -> edited Environment, Entity, and Activity Memory

pipeline/consolidation/
  repeated episodes -> Inferred Knowledge

memory_agent/
  formed memory -> representation-specific tools -> agent answers or plans

evaluation/eam_qa/ + evaluation/planning/
  evidence tools -> EAM-QA answers or grounded plans
```

The write-time implementation is documented in
[`pipeline/README.md`](pipeline/README.md). The full runnable path is in
[`docs/PIPELINE.md`](../../docs/PIPELINE.md).

## Benchmark interfaces

Both benchmarks ship in the sibling **[MEMORA-Bench](../memora_bench/README.md)** package:

| Arm | CLI | Benchmark JSON |
|-----|-----|----------------|
| EAM-QA | `evaluation/eam_qa/runner.py` | `../memora_bench/eam_qa/questions/` |
| MEMORA-Planning | `evaluation/planning/runner.py` | `../memora_bench/planning/suites/` (`replay`, `generalize`) |

The packaged JSON defines what is evaluated; `evaluation/` defines how models
are run on it. Optional author-side utilities for reconstructing the planning
suites live separately in
[`scripts/benchmark_construction/`](../../scripts/benchmark_construction/README.md).
Use [`scripts/build_evaluation_runs.py`](../../scripts/build_evaluation_runs.py)
to assemble multi-participant planning runs from the released suites.

The three memory-representation interfaces are organized under
[`memory_agent/memory_representations/`](memory_agent/README.md). MEMORA's
four-store implementation lives under
[`memory_agent/tools/`](memory_agent/tools/README.md).

**Paper table aggregation** lives outside the importable package under
[`scripts/paper_results/`](../../scripts/paper_results/README.md).

Install and run: [repository README](../../README.md), [docs/SETUP.md](../../docs/SETUP.md).

Paper numbers: [scripts/paper_results/](../../scripts/paper_results/README.md).
