# EAM-QA — MEMORA-Embodied Memory Assessment

Multiple-choice benchmark over 18 participants with an explicit **abstain
option (E)** for questions not supported by available memory.

| Path | Description |
|------|-------------|
| `questions/p{pid}.json` | 18 released participants (EPIC P-IDs, e.g. `p01`, `p22`, `p37`) |

**Counts (public release):** 2,212 answerable items + 551 E-correct abstain probes
(2,763 total). Paper headline metrics use the experience-dependent filtered subset
over these MC items (see Appendix B-E).

Every item includes a stable `question_id` for matching questions to saved
predictions.

Parent bundle: [../README.md](../README.md).
