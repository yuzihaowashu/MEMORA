# Memory Tools

`TypedMemoryTools` implements the four-store tool surface wrapped by
[`MEMORATools`](../memory_representations/memora/tools.py).

| Component | What It Handles |
|-----------|-----------------|
| `tool_interface.py` | Public facade, unified search, and tool dispatch |
| `context.py` | Memory loading, temporal reconstruction, and participant scope |
| `evidence/cross_store.py` | Cross-store episodic and semantic evidence views |
| `stores/environment.py` | Environment Memory: places, zones, and spatial relations |
| `stores/entity.py` | Entity Memory: object identity, attributes, state, and history |
| `stores/entity_normalization.py` | Participant-level Entity Memory normalization and deduplication |
| `stores/activity.py` | Activity Memory: timestamped action evidence |
| `stores/inferred.py` | Inferred Knowledge: routines, habits, preferences, and patterns |
| `evidence/episode.py` | Episode-level evidence for recall-style queries |
| `evidence/semantic.py` | Semantic evidence assembled across objects, activities, and patterns |
| `evidence/planning.py` | Planning-oriented evidence and object grounding |
| `embedding.py` | Shared optional E5 model used by semantic retrieval |
| `schemas.py` | Tool definitions exposed to the agent |
| `ranking.py` | Type-aware ranking configuration for Inferred Knowledge retrieval |
| `lexicon.py` | Query expansion terms for objects and locations |
| `formatting.py` | Compact text formatting for tool results |

The layout mirrors the paper's four-store memory design. New retrieval logic
should usually go into the relevant store or evidence module; `tool_interface.py`
stays focused on routing and composing tool outputs.
