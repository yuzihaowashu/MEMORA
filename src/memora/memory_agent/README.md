# Memory Agent

This directory implements MEMORA's read-time agent: it queries formed
participant memory through tools and produces answers or memory-grounded plans.

| File or folder | Role |
|----------------|------|
| `agent.py` | ReAct loop and model backend wrappers |
| `tool_call_parsing.py` | Normalizes Qwen, Gemma, and Hermes tool-call text |
| `agent_environment.py` | Manages ReAct state, tool calls, and final task answers |
| `memory_representations/` | Parallel MEMORA, Flat-1D, and Graph-2D representation interfaces |
| `tools/` | Four-store retrieval components used by MEMORA |

The main entry point for MEMORA's typed memory tools is
`tools/tool_interface.py`. It is intentionally an orchestration layer: store-specific
retrieval lives under `tools/stores/`, context lifecycle under `tools/context.py`,
and cross-store evidence assembly under `tools/evidence/` and
`tools/evidence/cross_store.py`.

The three memory representations are organized symmetrically:

- `memory_representations/memora/`: type-aware tools over the four MEMORA stores;
- `memory_representations/flat/`: one `search` interface over chronological text;
- `memory_representations/graph/`: graph-compatible search and temporal tools.

Each representation contains its read-time tool adapter and system prompt. The
evaluation settings map the seven paper conditions onto these three interfaces;
No-Memory is handled separately by the evaluation runner.

For benchmark CLIs, see `src/memora/evaluation/`.
