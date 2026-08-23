# Run the MEMORA pipeline

This guide follows the method lifecycle from raw egocentric video to a memory
file that can be queried by the planning agent. It is different from
[`REPRODUCE.md`](REPRODUCE.md), which recomputes paper numbers from released
outputs without rerunning memory formation.

## Data and environment

Install the GPU environment and make FFmpeg available on `PATH`:

```bash
bash scripts/setup_environment.sh gpu
source .venv/bin/activate

ffmpeg -version
ffprobe -version
```

Set the repository and output paths:

```bash
export MEMORA_ROOT="$PWD"
export VIDEO_ROOT=/path/to/EPIC-KITCHENS
export RUN_ROOT=/path/to/memora_run
mkdir -p "$RUN_ROOT"
```

`VIDEO_ROOT` may use the EPIC-KITCHENS layout
`P01/videos/P01_101.MP4`, or contain videos directly. Create a text file with
one video ID per line. Videos grouped into one participant memory should share
the same participant prefix. The Memory Editor rejects mixed-participant input.

```text
P01_101
P01_102
```

## 1. Segment Encoder

The Segment Encoder splits each video into non-overlapping 10-second segments
and directly emits Environment, Entity, and Activity observations. The paper
uses `Qwen/Qwen2.5-Omni-7B`.

```bash
memora-segment-encode \
  --video-dir "$VIDEO_ROOT" \
  --video-ids-file /path/to/p01_video_ids.txt \
  --output-dir "$RUN_ROOT/segment_observations" \
  --model-name Qwen/Qwen2.5-Omni-7B \
  --backend huggingface \
  --observation-format memora
```

Primary output:

```text
$RUN_ROOT/segment_observations/segment_observations.jsonl
```

Each line records one segment, its time window, and the three observation types.

## 2. Memory Editor and offline consolidation

The Memory Editor processes observations in timestamp order. It deterministically
merges Environment Memory, appends Activity Memory, and selects Add, Update,
Delete, or Noop for Entity Memory. With `--run-offline-consolidation`, the same
loaded model then consolidates participant-level Inferred Knowledge across all
episodes. The paper uses `Qwen/Qwen3-30B-A3B-Instruct-2507`.

```bash
memora-memory-edit \
  --input "$RUN_ROOT/segment_observations/segment_observations.jsonl" \
  --output-dir "$RUN_ROOT/participant_memory" \
  --memory-format memora \
  --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --backend vllm \
  --use-e5-retrieval \
  --require-e5 \
  --run-offline-consolidation
```

Primary outputs:

```text
$RUN_ROOT/participant_memory/participant_memory.json
$RUN_ROOT/participant_memory/memory_edit_history.jsonl
```

The final JSON groups edited per-video episodic records with one
participant-level Inferred Knowledge store:

```text
participant_memory.json
  memories_by_video
    <video_id>
      environment_log
      object_registry
      activity_log
  inferred_knowledge
```

## 3. Build Graph-2D comparison memories (optional)

Graph-2D uses the same deterministic builder for both controlled conditions.
Build the raw condition directly from Segment Encoder observations:

```bash
memora-graph-build \
  --input "$RUN_ROOT/segment_observations/segment_observations.jsonl" \
  --output "$RUN_ROOT/graph_2d_raw.json" \
  --evaluation-setting graph_2d_raw
```

Build the edited condition from participant memory:

```bash
memora-graph-build \
  --input "$RUN_ROOT/participant_memory/participant_memory.json" \
  --output "$RUN_ROOT/graph_2d_edited.json" \
  --evaluation-setting graph_2d_edited
```

The builder creates activity, object, environment, and inferred-knowledge
nodes with the rule-based edges used by the paper baseline. Flat-1D editing is
available through `memora-memory-edit --memory-format flat_1d`.

## 4. Use the memory for planning

Run a released MEMORA-Planning task with the formed memory:

```bash
memora-plan \
  --model Qwen/Qwen3.6-35B-A3B \
  --benchmark-file src/memora_bench/planning/suites/replay/p01.json \
  --memory-file "$RUN_ROOT/participant_memory/participant_memory.json" \
  --condition memora_full \
  --require-e5 \
  --output-dir "$RUN_ROOT/planning"
```

At read time, the agent receives the same type-aware tool interface used in the
paper. Tool implementations are organized under
`src/memora/memory_agent/tools/` by Environment, Entity, Activity, and
Inferred Knowledge.

The paper protocol uses E5 for both write-time record selection and read-time
retrieval. `--require-e5` prevents a missing model or dependency from silently
changing either operation to keyword retrieval.

## Use the released memories instead

Rerunning the full formation pipeline is expensive and is not required to
verify the paper tables. Download the released participant memories and saved
outputs from Hugging Face as described in [`DATA.md`](DATA.md), then follow
[`REPRODUCE.md`](REPRODUCE.md).

## API backends

Both formation CLIs support OpenAI-compatible APIs through `--backend api`,
`--model-name`, `--api-base`, and `--api-key`. API mode requires an explicit
model name because model aliases differ across endpoints. Set
`DASHSCOPE_API_KEY` for the default DashScope-compatible formation endpoint,
or pass an explicit endpoint and key.
The EAM-QA and planning CLIs also support API inference through `--use-api` and
recognize both `DASHSCOPE_API_KEY` and `OPENAI_API_KEY`. API execution is useful
for development or new videos, but it reproduces the paper configuration only
when the endpoint serves the same paper models and generation settings.
