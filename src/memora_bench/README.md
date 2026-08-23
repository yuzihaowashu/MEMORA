# MEMORA-Bench

MEMORA-Bench evaluates two properties of the same participant-specific memory:
faithfulness to prior experience and utility for future action.

[Explore real benchmark items in the interactive viewer](https://yuzihaowashu.github.io/MEMORA/benchmark-explorer.html).

| Protocol | Released files | Ground truth | Size |
|----------|-------------------|--------------|------|
| **EAM-QA** | Question, choices A-E, participant/video scope, and EAM type | Correct answer letter and answer text | 2,212 answerable items + 551 unanswerable controls |
| **Planning Replay** | Goal grounded in an observed participant workflow | Ordered EPIC-KITCHENS narration steps | 207 tasks |
| **Planning Generalize** | Transfer, composition, or fully novel goal | Ordered grounded reference plan | 153 tasks |

## Files

```text
eam_qa/questions/pXX.json
planning/suites/replay/pXX.json
planning/suites/generalize/pXX.json
```

EAM-QA and both MEMORA-Planning splits cover the same 18 participants.
Questions and references are static benchmark files; they are not produced
from a tested memory system's outputs.

Evaluation entry points:

- EAM-QA: [`src/memora/evaluation/eam_qa/runner.py`](../memora/evaluation/eam_qa/runner.py)
- MEMORA-Planning: [`src/memora/evaluation/planning/runner.py`](../memora/evaluation/planning/runner.py)

The paper's Robot-Grounded Plan score (RGP) is the unweighted mean of the
condition-level OrderExec, KeyObj, and PrefAdh axis scores. Saved-output
aggregation is implemented in `scripts/paper_results/planning_metrics.py`. These axes are
computed over successfully parsed, non-empty plans; empty-plan counts are
reported separately by the aggregation script.

Supporting documentation:

- [Optional planning-suite construction](../../scripts/benchmark_construction/README.md)
- [Path helpers](paths.py)
