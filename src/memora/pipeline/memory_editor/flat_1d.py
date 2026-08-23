"""
Write-time Memory Editor for the controlled Flat-1D baseline.

Contains:
- MEMORY_EDITOR_PROMPT / MEMORY_EDITOR_PROMPT_SHORT
- process_fact_group  – processes a single video/participant group
- run_memory_editor    – orchestrates all groups
- _print_group_visualization  – optional pretty-print
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from memora.pipeline.memory_editor.model_response import parse_json_response
from memora.pipeline.memory_editor.records import (
    MemoryEntry,
    group_observations_by_scope,
)
from memora.pipeline.memory_editor.retriever import get_e5_retriever

logger = logging.getLogger(__name__)


# Flat-1D Memory Editor prompt used in the controlled comparison.
MEMORY_EDITOR_PROMPT = """You are a smart Memory Manager which controls the memory of a system.

You maintain a list of memory elements. Each element has:
- "id": a string identifier (e.g. "0", "1", "2")
- "text": a concise fact, preference, state, or activity description

You receive:
1) The current memory (a list of {id, text})
2) A list of newly retrieved facts (strings)

Your task is to compare each newly retrieved fact with the existing memory and decide how the memory should change using FOUR operations:

- ADD:     Add a new element to memory (for genuinely new information)
- UPDATE:  Update an existing memory element (same underlying fact, but new state or better wording)
- DELETE:  Delete an existing memory element (when the new information contradicts it)
- NOOP:    No operation - make no change (memory element remains as is, or the new fact is irrelevant or already covered)

You must output a JSON object with a "memory" list. Each item in "memory" is a memory edit:

- For ADD:
  - You must generate a NEW id not already used in memory.
  - Set "event" to "ADD".
- For UPDATE:
  - Keep the SAME id as the element you update.
  - Provide the new "text".
  - Include an "old_memory" field containing the previous text.
  - Set "event" to "UPDATE".
- For DELETE:
  - Keep the SAME id as the element you delete.
  - Keep the same "text".
  - Set "event" to "DELETE".
- For NOOP:
  - Keep the SAME id and text.
  - Set "event" to "NOOP".

IMPORTANT RULES:

1. ADD:
   - Use ADD when the retrieved fact contains new information that is not already present in the memory.
   - For each ADD, generate a new unique id as a string (e.g. "3", "4", ...).
   - Do NOT add duplicates if an equivalent fact already exists.

2. UPDATE:
   - Use UPDATE when the retrieved fact:
     - Refines, extends, or corrects an existing memory about the same underlying fact, OR
     - Describes the same fact but with more relevant detail or a more accurate state, OR
     - Adds new information that should be MERGED with existing memory (e.g., adding new preferences).
   - Example (a): if memory says "User likes to play cricket" and new fact is "Loves to play cricket with friends",
     then UPDATE with the more detailed version.
   - Example (b): if memory says "Likes cheese pizza" and new fact is "Loves cheese pizza",
     do NOT update because they convey the same information (only wording differs).
   - Example (c): if memory says "Likes cheese pizza" and new fact is "Loves chicken pizza",
     then UPDATE by MERGING: "Likes cheese and chicken pizza".
   - When updating, KEEP the same id and include "old_memory".

3. DELETE:
   - Use DELETE when the retrieved fact clearly CONTRADICTS an existing memory.
   - Example: memory says "The dishwasher is closed" and new fact says "The person opens the dishwasher".
   - When deleting, keep the same id and text, but set "event": "DELETE".
   - Do NOT generate new ids for deletion.

4. NOOP (No Operation):
   - Use NOOP when:
     - The memory element is still correct and not modified by any new fact, OR
     - The retrieved fact is effectively the same information already stored, OR
     - The retrieved fact is irrelevant and should not be stored.
   - For NOOP, you keep id and text unchanged and set "event": "NOOP".

--- EXAMPLES ---

IMPORTANT: Only output operations for NEW FACTS, NOT for existing memory items!

1. **ADD Example** (new information):
Memory: [{ "id": "0", "text": "The plate is on the drying rack." }]
New facts: ["[activity] [10-20s] The person discards an orange peel into the trash can."]
Output:
    {"memory": [
        { "id": "1", "text": "[10-20s] The person discards an orange peel into the trash can.", "event": "ADD" }
    ]}
Note: We only output ADD for the new fact. The existing memory item is NOT repeated.

2. **UPDATE Example** (state changed):
Memory: [{ "id": "0", "text": "The spatula is in the sink." }]
New facts: ["[state] The spatula is on the drying rack."]
Output:
    {"memory": [
        { "id": "0", "text": "The spatula is on the drying rack.", "event": "UPDATE", "old_memory": "The spatula is in the sink." }
    ]}
Note: Same object (spatula), different location → UPDATE.

3. **NOOP Example** (duplicate fact):
Memory: [{ "id": "0", "text": "The plate is on the counter." }]
New facts: ["[state] The plate is on the counter."]
Output:
    {"memory": [
        { "text": "The plate is on the counter.", "event": "NOOP" }
    ]}
Note: Exact same text exists in memory → NOOP (no id needed).

4. **DELETE Example** (rarely used - only for clear contradictions):
Memory: [{ "id": "0", "text": "The fridge door is open." }]
New facts: ["[state] The fridge door is closed."]
Output:
    {"memory": [
        { "id": "0", "text": "The fridge door is open.", "event": "DELETE" },
        { "id": "1", "text": "The fridge door is closed.", "event": "ADD" }
    ]}
Note: DELETE old state AND ADD new state. Or better, just use UPDATE for state changes.

5. **Mixed Example** (multiple new facts):
Memory: [{ "id": "0", "text": "[0-10s] Wash plate" }, { "id": "1", "text": "Plate on counter" }]
New facts: ["[activity] [10-20s] Wash plate", "[state] Plate in sink", "[environment] Kitchen has sink"]
Output:
    {"memory": [
        { "id": "2", "text": "[10-20s] Wash plate", "event": "ADD" },
        { "id": "1", "text": "Plate in sink", "event": "UPDATE", "old_memory": "Plate on counter" },
        { "id": "3", "text": "Kitchen has sink", "event": "ADD" }
    ]}
Note: Activity with different time → ADD, State change → UPDATE, New environment → ADD.

6. **Empty Memory Example** (all ADD):
Memory: []
New facts: ["[activity] [0-10s] Wash plate", "[state] Plate on counter", "[environment] Kitchen has sink"]
Output:
    {"memory": [
        { "id": "0", "text": "[0-10s] Wash plate", "event": "ADD" },
        { "id": "1", "text": "Plate on counter", "event": "ADD" },
        { "id": "2", "text": "Kitchen has sink", "event": "ADD" }
    ]}
Note: Empty memory → ALL facts are ADD!

---

## FACT TYPE GUIDELINES (CRITICAL!)

Each retrieved fact is labeled with its type: [state], [activity], or [environment].
Use these type hints to guide your operation choice:

**[activity] facts** with time prefix like "[0-10s] Person washes plate"
  - ALWAYS ADD. Different time = different event
  - "[0-10s] wash plate" and "[10-20s] wash plate" are TWO different events
  - Only NOOP if EXACT same text+time already exists in memory

**[state] facts** like "Plate is on counter"
  - UPDATE ONLY if state CHANGED (e.g., "on counter" → "in sink")
  - NOOP if SAME TEXT exists! (e.g., "Plate on rack" → "Plate on rack" = NOOP)
  - New object not in memory → ADD

**[environment] facts** like "Kitchen has stainless steel sink"
  - Not in memory → ADD
  - Already exists → NOOP
  - Rarely UPDATE (only to enrich/correct)

---

You may output multiple edits in "memory" to reflect the effects of all retrieved facts.
If a retrieved fact is used to UPDATE or DELETE, you should not also ADD it as a new element.
Focus on producing a final consistent memory state.
Return ONLY the JSON object. Do not add explanations.
/no_think"""


# ============================================================================
# Alternative: Short/Concise Prompt (for smaller models or faster inference)
# ============================================================================
MEMORY_EDITOR_PROMPT_SHORT = """You are a Memory Manager. For each NEW fact, decide what operation to perform.

## YOUR TASK
You receive:
1) Current memory bank (list of {id, text})
2) New facts to process (list of strings with type labels)

For EACH new fact, output ONE operation:
- ADD: New information not in memory → add with new id
- UPDATE: Same object but DIFFERENT state → update existing id
- DELETE: Contradicts existing memory → delete by id
- NOOP: Already in memory (same text) or irrelevant → ignore

## OUTPUT FORMAT
{"memory": [
  {"id": "3", "text": "new fact text", "event": "ADD"},
  {"id": "1", "text": "updated state", "event": "UPDATE", "old_memory": "old state"},
  {"text": "duplicate fact", "event": "NOOP"}
]}

## FACT TYPE RULES

**[activity] facts** with time prefix like "[0-10s] Person washes plate"
  - ALWAYS ADD. Different time = different event
  - Only NOOP if EXACT same text+time exists

**[state] facts** like "Plate is on counter"
  - UPDATE ONLY if state CHANGED
  - NOOP if SAME TEXT exists!
  - New object → ADD

**[environment] facts** like "Kitchen has stainless steel sink"
  - Not in memory → ADD
  - Already exists → NOOP

Return ONLY the JSON object. No explanations.
/no_think"""


# ============================================================================
# Flat-memory processing functions
# ============================================================================

def _validate_editor_operations(result: Any, has_facts: bool) -> List[Dict[str, Any]]:
    """Validate the Flat-1D editor contract before mutating memory."""
    if not isinstance(result, dict):
        raise ValueError("Flat-1D Memory Editor response is not a JSON object")
    operations = result.get("memory")
    if not isinstance(operations, list):
        raise ValueError("Flat-1D Memory Editor response has no memory operation list")
    if has_facts and not operations:
        raise ValueError("Flat-1D Memory Editor returned no operations for new facts")

    allowed_events = {"ADD", "UPDATE", "DELETE", "NOOP"}
    mutation_targets = set()
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise ValueError(f"Memory operation {index} is not a JSON object")
        event = operation.get("event")
        if event not in allowed_events:
            raise ValueError(f"Memory operation {index} has invalid event: {event!r}")
        if not isinstance(operation.get("text"), str):
            raise ValueError(f"Memory operation {index} has no text")
        if event in {"UPDATE", "DELETE"}:
            target = operation.get("id")
            if not isinstance(target, str) or not target:
                raise ValueError(f"{event} operation {index} has no target id")
            if target in mutation_targets:
                raise ValueError(f"Duplicate mutations for memory id {target}")
            mutation_targets.add(target)
    return operations

def process_fact_group(
    facts: List[Dict[str, Any]],
    initial_memory: List[MemoryEntry],
    llm,
    sampling_params,
    tokenizer,
    max_memories: int,
    group_id: str,
    history_file_handle=None,  # For incremental writing
    use_e5_retrieval: bool = False  # Use E5 semantic retrieval
) -> tuple:
    """Process a group of facts (single video/participant)."""
    from tqdm import tqdm

    memory = initial_memory.copy()
    # Safely compute next_id: handle non-numeric IDs gracefully
    numeric_ids = []
    for m in memory:
        try:
            numeric_ids.append(int(m.id))
        except (ValueError, TypeError):
            pass
    next_id = max(numeric_ids + [-1]) + 1

    history = []
    total_ops = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0}
    turn_id = 0  # Turn counter for this video/group

    # Get E5 retriever if enabled
    e5_retriever = get_e5_retriever() if use_e5_retrieval else None

    for fact_entry in tqdm(facts, desc=f"Processing {group_id}", leave=False):
        episode_id = fact_entry.get("episode_id")
        segment = fact_entry.get("segment", {})
        narrations = fact_entry.get("narrations", [])

        if not narrations:
            continue

        # Extract fact texts and types
        fact_texts = [n.get("text", "") for n in narrations]
        fact_types = [n.get("type", "unknown") for n in narrations]

        time_window = {
            "start": segment.get("start_time", 0),
            "end": segment.get("end_time", 0)
        }

        # Build current memory state for prompt (LIMITED to max_memories)
        # CRITICAL: Use list() to create a COPY, not a reference!
        # Copy before appending so memory_before remains an immutable snapshot.

        if len(memory) > max_memories:
            if e5_retriever is not None:
                # E5 Semantic Retrieval: Select most relevant memories
                query = " ".join(fact_texts[:5])  # Use new facts as query
                memory_for_prompt = e5_retriever.retrieve(query, memory, top_k=max_memories)
                logger.debug(f"    E5 Retrieved {len(memory_for_prompt)}/{len(memory)} relevant memories")
            else:
                # Fallback: Take the most recent memories
                memory_for_prompt = list(memory[-max_memories:])
        else:
            memory_for_prompt = list(memory)

        # ============================================================
        # Preserve stable record IDs across editor responses.
        # ============================================================
        # Remap IDs to 0, 1, 2, ... for cleaner model input
        # Store mapping for reverse lookup after model output
        id_remap = {}  # remapped_id -> original_id
        id_remap_reverse = {}  # original_id -> remapped_id

        remapped_memory = []
        for new_idx, m in enumerate(memory_for_prompt):
            original_id = m.id
            remapped_id = str(new_idx)

            id_remap[remapped_id] = original_id
            id_remap_reverse[original_id] = remapped_id

            remapped_memory.append({
                "id": remapped_id,
                "text": m.text
            })

        # Next available ID for ADD (in remapped space)
        next_remapped_id = len(memory_for_prompt)

        current_memory = remapped_memory  # Use remapped version for prompt

        # Preserve both the complete state and the bounded subset shown to the model.
        memory_before_snapshot = [m.to_dict() for m in memory]
        prompt_memory_snapshot = [m.to_dict() for m in memory_for_prompt]

        # ============================================================
        # FORMAT FACTS WITH TYPE LABELS: Consistent with env.py
        # ============================================================
        formatted_facts = []
        for i, text in enumerate(fact_texts):
            fact_type = fact_types[i] if i < len(fact_types) else "unknown"
            formatted_facts.append(f"[{fact_type}] {text}")

        user_message = f"""Current memory:
{json.dumps(current_memory, ensure_ascii=False, indent=2)}

Next available ID for ADD: "{next_remapped_id}"

New facts to process:
{json.dumps(formatted_facts, ensure_ascii=False, indent=2)}

Please determine how to update the memory based on fact types:
- [state] facts: Prefer UPDATE for same objects (states change over time)
- [activity] facts: Prefer ADD (activities are sequential events)
- [environment] facts: Prefer NOOP or UPDATE (environment is stable)"""

        messages = [
            {"role": "system", "content": MEMORY_EDITOR_PROMPT},
            {"role": "user", "content": user_message}
        ]

        # Format for vLLM
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        try:
            outputs = llm.generate([prompt], sampling_params)
            if not outputs or not outputs[0].outputs:
                raise RuntimeError("LLM returned empty output for Flat-1D Memory Editor prompt")
            response_text = outputs[0].outputs[0].text.strip()
            result = parse_json_response(response_text)
            editor_operations = _validate_editor_operations(result, has_facts=bool(fact_texts))

            # Apply operations
            # Build lookup for existing memory texts (for NOOP validation)
            existing_memory_texts = {m.text for m in memory}

            operations = []
            for op in editor_operations:
                event = op["event"]
                remapped_id = op.get("id", "")  # Model outputs remapped ID
                raw_text = op.get("text", "")

                # ============================================================
                # Strip [type] prefix from model output
                # ============================================================
                # Model may output "[activity] [0-10s] Wash plate" instead of "[0-10s] Wash plate"
                # Strip the [type] prefix to get clean text
                text = re.sub(r'^\[(activity|state|environment)\]\s*', '', raw_text)

                # ============================================================
                # REVERSE ID MAPPING: Convert remapped ID back to original
                # ============================================================
                if remapped_id in id_remap:
                    original_id = id_remap[remapped_id]
                else:
                    # New ID (for ADD) or invalid - use as-is or generate new
                    original_id = str(next_id) if event == "ADD" else remapped_id

                # ============================================================
                # Validate DELETE/UPDATE IDs exist
                # ============================================================
                if event in ["DELETE", "UPDATE"] and remapped_id not in id_remap:
                    logger.warning(f"    {event} with invalid ID '{remapped_id}' - skipping")
                    # Record as failed operation for diagnostics
                    op_record = {"event": f"{event}_INVALID", "id": remapped_id, "text": text}
                    operations.append(op_record)
                    continue

                # ============================================================
                # Convert NOOP to ADD if text doesn't exist in memory
                # ============================================================
                if event == "NOOP" and text not in existing_memory_texts:
                    # Model wrongly NOOP'd a fact that doesn't exist in memory
                    # This often happens with states/environments when memory is empty
                    logger.debug(f"    Converting NOOP to ADD: '{text[:50]}...' (not in memory)")
                    event = "ADD"

                # ============================================================
                # Find fact type from original fact_texts (for ALL events)
                # ============================================================
                fact_type = None
                if text in fact_texts:
                    idx = fact_texts.index(text)
                    if idx < len(fact_types):
                        fact_type = fact_types[idx]

                if event == "ADD":
                    new_id = str(next_id)
                    next_id += 1

                    new_entry = MemoryEntry(
                        id=new_id,
                        text=text,  # Use cleaned text (without [type] prefix)
                        episode_id=episode_id,
                        time_window=time_window,
                        fact_type=fact_type
                    )
                    memory.append(new_entry)
                    existing_memory_texts.add(text)
                    # Include fact_type in operation history
                    op_record = {"event": "ADD", "id": new_id, "text": text}
                    if fact_type:
                        op_record["fact_type"] = fact_type
                    operations.append(op_record)
                    total_ops["ADD"] += 1

                elif event == "UPDATE":
                    for entry in memory:
                        if entry.id == original_id:  # Use original_id (reverse mapped)
                            old_text = entry.text
                            entry.text = text  # Use cleaned text
                            entry.time_window = time_window
                            # Update entry's fact_type if found
                            if fact_type:
                                entry.fact_type = fact_type
                            op_record = {
                                "event": "UPDATE",
                                "id": original_id,  # Use original_id in output
                                "text": text,
                                "old_text": old_text
                            }
                            if fact_type:
                                op_record["fact_type"] = fact_type
                            operations.append(op_record)
                            total_ops["UPDATE"] += 1
                            existing_memory_texts.discard(old_text)
                            existing_memory_texts.add(text)
                            break

                elif event == "DELETE":
                    # Find the entry to get its fact_type before deleting
                    # Use original_id (reverse mapped from model's remapped ID)
                    deleted_entry = next((m for m in memory if m.id == original_id), None)
                    if deleted_entry:
                        fact_type = deleted_entry.fact_type
                        memory = [m for m in memory if m.id != original_id]
                        existing_memory_texts.discard(deleted_entry.text)
                        op_record = {"event": "DELETE", "id": original_id, "text": text}
                        if fact_type:
                            op_record["fact_type"] = fact_type
                        operations.append(op_record)
                        total_ops["DELETE"] += 1
                    else:
                        logger.warning(f"    DELETE target not found: id={original_id}")

                else:  # NOOP
                    # Also record NOOP operations for training completeness
                    op_record = {"event": "NOOP", "text": text}
                    if fact_type:
                        op_record["fact_type"] = fact_type
                    operations.append(op_record)
                    total_ops["NOOP"] += 1

            # Always record time window, even if all operations are NOOP
            # This ensures complete training data coverage
            # Record before/after states for auditing each edit operation.
            history_entry = {
                "episode_id": episode_id,
                "turn_id": turn_id,  # Turn number within this video (0, 1, 2, ...)
                "time_window": time_window,
                "memory_before": memory_before_snapshot,
                "prompt_memory": prompt_memory_snapshot,
                "new_facts": fact_texts,  # Original facts presented to model
                "new_fact_types": fact_types,  # Fact types for each new fact
                "operations": operations,  # Model's operations (ADD/UPDATE/DELETE/NOOP)
                "memory_after": [{"id": m.id, "text": m.text} for m in memory],  # Memory state AFTER operations (for benchmark evaluation!)
                "memory_size_before": len(memory_before_snapshot),
                "prompt_memory_size": len(prompt_memory_snapshot),
                "memory_size_after": len(memory),
            }
            history.append(history_entry)
            turn_id += 1  # Increment turn counter

            # Incremental write: save immediately to file
            if history_file_handle:
                history_file_handle.write(json.dumps(history_entry, ensure_ascii=False) + '\n')
                history_file_handle.flush()  # Ensure data is written to disk

        except Exception as e:
            raise RuntimeError(
                f"Flat-1D Memory Editor failed for {episode_id}, turn {turn_id}: "
                f"{type(e).__name__}: {e}"
            ) from e

    return memory, history, total_ops, next_id


def run_memory_editor(
    extracted_facts: List[Dict[str, Any]],
    initial_memory: List[MemoryEntry],
    llm,
    sampling_params,
    tokenizer,
    memory_scope: str = "per_participant",
    max_memories: int = 60,
    enable_visualization: bool = False,
    output_dir: Optional[Path] = None,  # For incremental writing
    use_e5_retrieval: bool = False  # Use E5 semantic retrieval
) -> tuple:
    """
    Run memory editor on all extracted facts.

    Args:
        extracted_facts: List of fact entries
        initial_memory: Initial memory state (global)
        llm: vLLM model
        sampling_params: Sampling parameters
        tokenizer: Tokenizer
        memory_scope: "per_video", "per_participant", or "global"
        max_memories: Maximum memories to include in prompt
        output_dir: Output directory for incremental saving (optional)
        use_e5_retrieval: Use E5 semantic retrieval to select prompt records

    Returns:
        (all_memories, all_history, total_ops, all_group_memories)
    """
    from tqdm import tqdm

    # Group facts by scope
    grouped_facts = group_observations_by_scope(extracted_facts, memory_scope)

    logger.info(f" Memory scope: {memory_scope}")
    logger.info(f"   Groups: {len(grouped_facts)}")
    logger.info(f"   Max memories per prompt: {max_memories}")
    logger.info(f"   E5 Retrieval: {' Enabled' if use_e5_retrieval else ' Disabled (using recency)'}")
    if output_dir:
        logger.info(f"Incremental saving enabled: {output_dir}")

    all_memories = []
    all_history = []
    total_ops = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0}
    all_group_memories = {}

    # Open files for incremental writing
    history_file_handle = None
    memory_per_group_file = None
    memory_per_group_handle = None  # Must be initialized to avoid NameError in finally block

    if output_dir:
        history_file = output_dir / "operation_history.jsonl"
        history_file_handle = open(history_file, 'w', encoding='utf-8')
        logger.info(f"Incremental history file: {history_file}")

        if memory_scope in ["per_video", "per_participant"]:
            memory_per_group_file = output_dir / "participant_memory_per_group.jsonl"
            memory_per_group_handle = open(memory_per_group_file, 'w', encoding='utf-8')
            logger.info(f"Incremental memory file: {memory_per_group_file}")

    try:
        # Process each group
        for group_id in tqdm(sorted(grouped_facts.keys()), desc="Processing groups"):
            group_facts = grouped_facts[group_id]

            # For per_video/per_participant: start with empty memory
            # For global: use accumulated memory
            if memory_scope in ["per_video", "per_participant"]:
                group_memory = []
            else:  # global
                group_memory = all_memories.copy()

            # Add initial memory if provided
            if initial_memory and memory_scope == "global":
                group_memory = initial_memory.copy()

            logger.info(f"\n Processing {group_id}: {len(group_facts)} facts")

            memory, history, ops, next_id = process_fact_group(
                facts=group_facts,
                initial_memory=group_memory,
                llm=llm,
                sampling_params=sampling_params,
                tokenizer=tokenizer,
                max_memories=max_memories,
                group_id=group_id,
                history_file_handle=history_file_handle,
                use_e5_retrieval=use_e5_retrieval
            )

            # Store results
            group_memory_record = {
                "group_id": group_id,
                "memory": [m.to_dict() for m in memory],
                "operations": ops
            }
            all_group_memories[group_id] = group_memory_record

            # Incremental save: per-group memory file
            if output_dir and memory_scope in ["per_video", "per_participant"]:
                memory_per_group_handle.write(json.dumps(group_memory_record, ensure_ascii=False) + '\n')
                memory_per_group_handle.flush()

            # Accumulate for global scope
            if memory_scope == "global":
                all_memories = memory
            else:
                all_memories.extend(memory)

            all_history.extend(history)

            for op_type in total_ops:
                total_ops[op_type] += ops[op_type]

            logger.info(f"    {group_id}: {len(memory)} memories, ADD={ops['ADD']}, UPDATE={ops['UPDATE']}, DELETE={ops['DELETE']}")

            # Detailed visualization (if enabled)
            if enable_visualization:
                _print_group_visualization(
                    group_id=group_id,
                    memory_before=group_memory,
                    memory_after=memory,
                    history=history,
                    ops=ops
                )

    finally:
        # Close file handles safely
        if history_file_handle:
            history_file_handle.close()
            logger.info(f"    History file saved: {len(all_history)} entries")
        if memory_per_group_handle:
            memory_per_group_handle.close()
            logger.info(f"    Per-group memory file saved: {len(all_group_memories)} groups")

    return all_memories, all_history, total_ops, all_group_memories


def _print_group_visualization(
    group_id: str,
    memory_before: List[MemoryEntry],
    memory_after: List[MemoryEntry],
    history: List[Dict],
    ops: Dict[str, int]
):
    """
     Print detailed visualization for a processing group.
    """
    op_symbols = {"ADD": "", "UPDATE": "", "DELETE": "", "NOOP": "⏸️"}

    print(f"\n{'='*70}")
    print(f" MEMORY OPERATIONS - {group_id}")
    print(f"{'='*70}")

    print(f"\n BEFORE: {len(memory_before)} entries")
    if memory_before:
        for m in memory_before[:3]:
            text = m.text if hasattr(m, 'text') else m.get('text', str(m))
            print(f"   [{m.id if hasattr(m, 'id') else m.get('id', '?')}] {text[:50]}...")
        if len(memory_before) > 3:
            print(f"   ... and {len(memory_before) - 3} more")

    print("\n OPERATIONS:")
    for h in history[-5:]:  # Show last 5 operations
        time_window = h.get("time_window", {})
        time_str = f"{time_window.get('start', 0)}-{time_window.get('end', 0)}s" if time_window else "N/A"

        for op in h.get("operations", []):
            event = op.get("event", "NOOP").upper()
            op_id = op.get("id", "?")
            text = op.get("text", "")[:40]
            old_text = op.get("old_text", "")[:20]
            symbol = op_symbols.get(event, "❓")

            if event == "UPDATE":
                print(f"   {symbol} [{time_str}] id={op_id}: '{old_text}...' → '{text}...'")
            else:
                print(f"   {symbol} [{time_str}] id={op_id}: {text}...")

    if len(history) > 5:
        print(f"   ... and {len(history) - 5} more time windows")

    print(f"\n AFTER: {len(memory_after)} entries")
    if memory_after:
        for m in memory_after[:3]:
            text = m.text if hasattr(m, 'text') else m.get('text', str(m))
            print(f"   [{m.id if hasattr(m, 'id') else m.get('id', '?')}] {text[:50]}...")
        if len(memory_after) > 3:
            print(f"   ... and {len(memory_after) - 3} more")

    print("\n SUMMARY:")
    print(f"    ADD: {ops['ADD']} |  UPDATE: {ops['UPDATE']} | "
          f" DELETE: {ops['DELETE']} | ⏸️ NOOP: {ops['NOOP']}")
    print(f"   Memory change: {len(memory_before)} → {len(memory_after)} "
          f"(Δ = {len(memory_after) - len(memory_before):+d})")
    print(f"{'='*70}\n")
