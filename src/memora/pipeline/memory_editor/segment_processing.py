"""Coordinate one online update across MEMORA's typed memory stores."""

import json
import logging
from typing import Any, Dict

from memora.pipeline.memory_editor.retriever import get_e5_retriever
from memora.pipeline.memory_editor.object_operations import (
    apply_object_operations,
    maintain_unmentioned_observations,
)
from memora.pipeline.memory_editor.typed_memory import (
    EmbodiedMemoryState,
    TYPED_MEMORY_EDITOR_PROMPT,
    _compact_registry_for_prompt,
    apply_rule_based_operations,
)
from memora.pipeline.memory_editor.model_response import parse_json_response
from memora.pipeline.memory_editor.prompts import build_memory_editor_prompt

logger = logging.getLogger(__name__)

def process_typed_memory_segment(
    current_memory: EmbodiedMemoryState,
    new_segment: Dict[str, Any],
    llm,
    sampling_params,
    tokenizer,
    max_objects: int = 50,  # Max objects to show in prompt
    use_e5_retrieval: bool = False,  # Enable E5 retrieval for large registries
    config=None,  # Optional formation prompt configuration.
) -> tuple:
    """
    Process a single MEMORA typed-memory segment.

    Args:
        current_memory: Current MEMORA memory state
        new_segment: New observation from the Segment Encoder
        llm: vLLM model
        sampling_params: Sampling parameters
        tokenizer: Tokenizer
        max_objects: Maximum objects to include in prompt (to prevent too long prompts)
        use_e5_retrieval: Use E5 to select relevant objects when registry is large
        config: Optional FormationConfig controlling prompts and place IDs.

    Returns:
        (updated_memory, rule_based_ops, llm_ops)
    """
    # Activity and Environment Memory use deterministic update rules.
    rule_based_ops = apply_rule_based_operations(
        current_memory,
        new_segment,
        config=config,
    )

    # Prepare the Entity Memory input for semantic editing.
    new_object_registry = new_segment.get("object_registry", {}) or {}

    # Some Segment Encoder backends can emit malformed
    # registry entries: a bare string (the name) or None instead of the
    # expected dict. Normalize once so downstream `.get(...)` calls are safe.
    if isinstance(new_object_registry, dict):
        _normalized = {}
        for _oid, _odata in new_object_registry.items():
            if _odata is None:
                continue
            if isinstance(_odata, str):
                _odata = {"name": _odata}
            elif not isinstance(_odata, dict):
                continue
            _normalized[_oid] = _odata
        new_object_registry = _normalized
    else:
        new_object_registry = {}

    # ============================================================
    # Object Registry Filtering (when registry is too large)
    # ============================================================
    # Select the most relevant records when the registry exceeds the prompt budget.
    # Priority: 1) Objects in new_object_registry (always include)
    #           2) E5 retrieval based on new object names (if enabled)
    #           3) Most recently updated objects (fallback)

    current_registry_for_prompt = current_memory.object_registry
    filtered_object_ids = None  # Track which objects were shown to LLM

    if len(current_memory.object_registry) > max_objects:
        # Get object_ids from new observations
        new_object_ids = set(new_object_registry.keys())

        if use_e5_retrieval and new_object_ids:
            # Use E5 to find similar objects from current registry
            e5_retriever = get_e5_retriever()

            # Build query from new object descriptions.
            # Some Segment Encoder backends emit `"spatial_info": null`. Using
            # `.get("spatial_info", {})` only returns `{}` when the key is
            # missing; an explicit None remains None and the next
            # `.get(...)` crashes. The `or {}` guard covers both cases.
            query_parts = []
            for obj_id, obj_data in new_object_registry.items():
                if not isinstance(obj_data, dict):
                    continue
                name = obj_data.get("name", obj_id)
                spatial = obj_data.get("spatial_info") or {}
                location = spatial.get("location", "") if isinstance(spatial, dict) else ""
                query_parts.append(f"{name} {location}")
            query = " ".join(query_parts)

            # Convert current registry to list for retrieval
            def _obj_text(k, v):
                if not isinstance(v, dict):
                    return {"object_id": k, "text": str(k)}
                spatial = v.get("spatial_info") or {}
                loc = spatial.get("location", "unknown") if isinstance(spatial, dict) else "unknown"
                return {"object_id": k, "text": f"{v.get('name', k)} at {loc}"}

            current_objects = [
                _obj_text(k, v)
                for k, v in current_memory.object_registry.items()
                if k not in new_object_ids  # Don't include new objects (they'll be added back)
            ]

            # Retrieve relevant objects
            remaining_slots = max_objects - len(new_object_ids)
            if current_objects and remaining_slots > 0:
                relevant_objects = e5_retriever.retrieve(query, current_objects, top_k=remaining_slots)
                relevant_ids = {obj["object_id"] for obj in relevant_objects}
            else:
                relevant_ids = set()

            # Combine: new objects + relevant existing objects
            filtered_object_ids = new_object_ids | relevant_ids
            current_registry_for_prompt = {
                k: v for k, v in current_memory.object_registry.items()
                if k in filtered_object_ids
            }

            logger.debug(f"    E5 filtered {len(current_memory.object_registry)} objects → {len(current_registry_for_prompt)} for prompt")
        else:
            # Fallback: Include new objects + take first max_objects from current registry
            # (Assumes objects are somewhat ordered by recency in dict)
            filtered_object_ids = set(new_object_ids)
            remaining_slots = max_objects - len(new_object_ids)

            for obj_id in current_memory.object_registry.keys():
                if obj_id not in filtered_object_ids:
                    filtered_object_ids.add(obj_id)
                    if len(filtered_object_ids) >= max_objects:
                        break

            current_registry_for_prompt = {
                k: v for k, v in current_memory.object_registry.items()
                if k in filtered_object_ids
            }

            logger.debug(f"    Truncated {len(current_memory.object_registry)} objects → {len(current_registry_for_prompt)} for prompt")

    # Build context for LLM (object_registry only — environment is rule-based,
    # inferred_knowledge is populated offline)
    turn_id = new_segment.get("turn_id", 0)
    time_window = new_segment.get("time_window", {})

    compact_reg = _compact_registry_for_prompt(current_registry_for_prompt)
    n_total = len(current_memory.object_registry)
    n_shown = len(current_registry_for_prompt)
    filter_note = f" [showing {n_shown}/{n_total}]" if filtered_object_ids else ""

    user_message = f"""## Current Object Registry ({n_shown} objects{filter_note}):
{compact_reg}

## New Object States (from current segment -- full JSON):
{json.dumps(new_object_registry, ensure_ascii=False, indent=2)}

For each object in New Object States, decide: ADD / UPDATE / DELETE / NOOP."""

    # Use the configured experience description in the editor system prompt.
    if config is not None:
        system_prompt = build_memory_editor_prompt(config)
    else:
        system_prompt = TYPED_MEMORY_EDITOR_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # Format for vLLM
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    llm_ops = []

    try:
        outputs = llm.generate([prompt], sampling_params)
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned empty output for memory editor prompt")
        response_text = outputs[0].outputs[0].text.strip()

        result = parse_json_response(response_text)
        if not isinstance(result, dict):
            raise ValueError("Memory Editor response is not a JSON object")

        object_operations = result.get("object_operations")
        if not isinstance(object_operations, list):
            raise ValueError("Memory Editor response has no object_operations list")
        allowed_events = {"ADD", "UPDATE", "DELETE", "NOOP"}
        seen_operation_ids = set()
        for index, operation in enumerate(object_operations):
            if not isinstance(operation, dict):
                raise ValueError(f"Object operation {index} is not a JSON object")
            event = operation.get("event")
            object_id = operation.get("object_id")
            if event not in allowed_events:
                raise ValueError(f"Object operation {index} has invalid event: {event!r}")
            if not isinstance(object_id, str) or not object_id.strip():
                raise ValueError(f"Object operation {index} has no object_id")
            if object_id in seen_operation_ids:
                raise ValueError(f"Memory Editor returned duplicate operations for {object_id}")
            seen_operation_ids.add(object_id)
            if event == "ADD" and not isinstance(operation.get("data"), dict):
                raise ValueError(f"ADD operation for {object_id} has no data object")
            if event == "UPDATE" and not isinstance(operation.get("changes"), dict):
                raise ValueError(f"UPDATE operation for {object_id} has no changes object")

        if result:
            # ============================================================
            # Apply object operations (with state_history tracking)
            # (environment is rule-based; inferred_knowledge is offline)
            # ============================================================

            llm_ops = apply_object_operations(
                current_memory,
                object_operations,
                new_object_registry,
                turn_id,
                time_window,
            )
            llm_ops = maintain_unmentioned_observations(
                current_memory,
                new_object_registry,
                llm_ops,
                turn_id,
                time_window,
            )

    except Exception as e:
        raise RuntimeError(
            f"Memory Editor failed for turn {turn_id}: {type(e).__name__}: {e}"
        ) from e

    return current_memory, rule_based_ops, llm_ops
