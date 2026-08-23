# MEMORA-Planning suite construction

These optional author-side utilities reconstruct the released
**MEMORA-Planning** suites from EPIC-KITCHENS-100 annotations and participant
memory. Running MEMORA or reproducing reported results does not require them;
the curated Replay and Generalize suites are already released under
`src/memora_bench/planning/suites/`.

EAM-QA construction is documented in the paper appendix. This directory does
not claim to reproduce that separate curation process.

## Construction stages

| Stage | Script | Output |
|-------|--------|--------|
| Extract Replay candidates | `extract_replay_candidates.py` | Candidate action sequences from EPIC-KITCHENS-100 annotations |
| Build Replay suite | `build_replay_suite.py` | Participant-grounded Replay tasks |
| Generate Generalize candidates | `generate_generalize_candidates.py` | Transfer, composition, and fully novel candidate goals |
| Build Generalize suite | `build_generalize_suite.py` | Participant-grounded goals and reference plans |

`build_replay_suite.py` can optionally use a local language model to rewrite
task instructions. This is a construction-time operation, not an evaluation
judge. Reported planning metrics remain rule-based.

## Example

```bash
source scripts/configure_paths.sh

B="$MEMORA_BENCH_DIR/planning"
W="$MEMORA_DATA_ROOT/benchmark_build"

python3 scripts/benchmark_construction/extract_replay_candidates.py \
  --csv-path /path/to/EPIC_100_train.csv \
  --memory-jsonl /path/to/segment_observations.jsonl \
  --output "$W/replay_candidates_p01.json"

python3 scripts/benchmark_construction/build_replay_suite.py \
  --input "$W/replay_candidates_p01.json" \
  --memory-file "$PLANNING_MEMORY_ROOT/memora_full/participant_memory_p01.json" \
  --output "$B/suites/replay/p01.json"
```

The official annotation source is linked in [`docs/DATA.md`](../../docs/DATA.md). By
default, extraction uses the released participant-video selection in
`planning/data/participant_video_ids.jsonl`.
