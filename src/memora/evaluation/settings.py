"""Paper evaluation conditions shared by both MEMORA-Bench protocols."""

PUBLIC_EVALUATION_CONDITIONS = (
    "no_memory",
    "flat_1d_raw",
    "flat_1d_edited",
    "graph_2d_raw",
    "graph_2d_edited",
    "memora_episodic",
    "memora_full",
)

CONDITION_MEMORY_TYPE = {
    "no_memory": "no_memory",
    "flat_1d_raw": "flat_1d",
    "flat_1d_edited": "flat_1d",
    "graph_2d_raw": "graph_2d",
    "graph_2d_edited": "graph_2d",
    "memora_episodic": "memora",
    "memora_full": "memora",
}
