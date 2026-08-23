# Memory formation pipeline

This package implements the write-time lifecycle described in the paper.

```text
egocentric video
  -> Segment Encoder
  -> Environment, Entity, and Activity observations
  -> Memory Editor
  -> persistent participant memory
  -> offline consolidation
  -> Inferred Knowledge
```

## Components

| Paper component | Source | Input | Output |
|---|---|---|---|
| Segment Encoder | `segment_encoder/` | Non-overlapping video segments | `segment_observations.jsonl` |
| Memory Editor | `memory_editor/` | Each video's segment observations in timestamp order | `participant_memory.json` and edit history |
| Offline consolidation | `consolidation/` | All edited episodes for one participant | Root-level `inferred_knowledge` |
| Graph-2D builder | `representation_builders/graph_2d.py` | Raw observations or edited participant memory | Controlled graph representation |

## Reading the implementation

Read the formation path in this order:

1. `segment_encoder/pipeline.py` coordinates videos, sequential context,
   retries, and durable output. Its supporting modules expose the observation
   contract (`core.py`, `observations.py`), video inputs (`video.py`), persisted
   records (`records.py`), stage prompt (`prompts.py`), and inference backends
   (`backends/`).
2. `memory_editor/typed_memory.py`: defines the persistent episodic state and
   deterministic Environment and Activity update primitives.
3. `memory_editor/segment_processing.py`: coordinates one observation update
   across the three online stores. `memory_editor/object_operations.py` applies
   validated Add, Update, Delete, or Noop decisions to Entity Memory while
   preserving state history.
4. `memory_editor/cli.py`: processes segments in timestamp order and writes the
   participant memory plus the edit trace.
5. `consolidation/runner.py`: gathers evidence across the participant's edited
   episodes and writes the Inferred Knowledge store.
6. `representation_builders/graph_2d.py`: deterministically projects raw or
   edited evidence into the Graph-2D comparison representation. Flat-1D's
   write-time editor is implemented in `memory_editor/flat_1d.py`.

The important data contracts are:

```text
segment_observations.jsonl
  environment + object_registry + activity_narrative

participant_memory.json
  memories_by_video
    environment_log + object_registry + activity_log
  inferred_knowledge
```

This separation mirrors the method: the Segment Encoder observes, the Memory
Editor maintains persistent episodic state, and offline consolidation derives
cross-episode regularities.

Stage-specific prompts live beside the stage that uses them:

| Directory | Role |
|---|---|
| `segment_encoder/prompts.py` | Environment, Entity, and Activity observation prompts |
| `memory_editor/prompts.py` | Entity Memory Add, Update, Delete, and Noop prompt |
| `consolidation/prompts.py` | Participant preference and reusable procedure prompts |
| `formation_config.py` | Segmentation, experience description, and Environment Memory place IDs shared across formation |

The Segment Encoder backends share one observation contract; selecting a local
or API backend does not change the downstream memory format.

The read-time path is separate from memory formation:

- `../memory_agent/` contains MEMORA's type-aware four-store interface and
  the Flat-1D and Graph-2D comparison interfaces.
- `../evaluation/eam_qa/runner.py` evaluates retrospective memory use.
- `../evaluation/planning/runner.py` evaluates memory-grounded planning.

Commands and expected files are documented in
[`docs/PIPELINE.md`](../../../docs/PIPELINE.md).
