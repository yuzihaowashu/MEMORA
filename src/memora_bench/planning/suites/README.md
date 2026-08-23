# MEMORA-Planning task suites

| Directory | Suite | CLI |
|-----------|-------|-----|
| `replay/` | Replay (observed workflows) | `--suite replay` |
| `generalize/` | Generalize (transfer, composition, and fully novel goals) | `--suite generalize` |

Replay reference steps retain their source-video narration records, including
timestamps and grounded narrations when available. Generalize reference steps
are grounded instruction strings because they need not correspond to one
observed episode.

Each suite contains one JSON file per released participant (18 EPIC P-IDs,
for example `p01.json`, `p22.json`).
Every file is a top-level JSON task array with the same core fields:
`task_id`, `participant_id`, `task_type`, `task_query`, supporting video IDs,
and `ground_truth_steps`.

Parent bundle: [../../README.md](../../README.md).
