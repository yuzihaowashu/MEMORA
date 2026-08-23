"""Task-level instructions for EAM-QA answering and abstention."""

from typing import Any, Dict


NO_MEMORY_SYSTEM_PROMPT = """You are answering a multiple-choice question about a participant's prior embodied experience. You have no participant memory, retrieved evidence, or external context. Select option E when the participant-specific evidence needed to distinguish A-D is unavailable. Return only `**Answer: X**`, where X is A, B, C, D, or E."""


def format_multiple_choice_question(task: Dict[str, Any]) -> str:
    """Format one released EAM-QA item for a direct model call."""
    lines = [str(task.get("question") or "").strip(), "", "Options:"]
    lines.extend(
        f"{chr(65 + index)}) {choice}"
        for index, choice in enumerate(task.get("choices", []))
    )
    return "\n".join(lines)


FORCED_ANSWER_PROMPT_EAM_QA = """You have reached the maximum number of search iterations. Based on ALL the information you have gathered so far, you MUST now provide your final answer.

## Requirements:
1. **YOU MUST PROVIDE AN ANSWER** - Choose A, B, C, D, or E.
2. **Use evidence, not guessing** - Choose A-D when the gathered memory evidence supports one option better than the others. Exact wording is not required.
3. **Option E means no usable memory evidence** - Choose E only if the relevant narrative, pattern, or aggregation tools are empty, unrelated, or too contradictory to support any A-D option.
4. **Do not over-abstain** - If one A-D option is supported by a video summary, narrative evidence, pattern, count, or transition aggregate, choose it rather than E.
5. **Do not search again** - No more tool calls allowed.

## Output Format:
First, provide a brief reasoning process grounded in the gathered information.
Then, output your final answer in this exact format:
**Answer: X** (where X is A, B, C, D, or E)

Now, analyze your gathered information and provide your final answer:"""

EAM_QA_ABSTENTION_GUIDANCE = (
    '\n\n## EAM-QA OPTION-E MODE\n'
    'This question includes option E for insufficient information.\n'
    '- Output A-D when narrative, pattern, object-history, count, or transition evidence supports one option better than the others.\n'
    '- Do not require exact lexical match between evidence and choice; map paraphrases to the closest supported option.\n'
    '- Output `**Answer: E**` only when the relevant evidence is missing, empty, unrelated, or too contradictory to support any A-D option.\n'
    '- For ERecall, do not select E after just one empty generic search; check video summary or focused narrative evidence first.\n'
    '- For semantic memory, do not select E after just one empty pattern search; use aggregation when choices compare actions or objects.\n'
    '- If a specialized tool returns partial but relevant evidence for A-D, prefer the supported A-D option over E.\n'
    '- Final answer format: `**Answer: X**`, where X is A, B, C, D, or E.\n'
)

_ERECALL_DEFAULT_TO_AD_HINT = (
    "- Paraphrase tolerance — treat each pair as the same activity unless context contradicts: wash≈rinse≈scrub≈clean, cut≈slice≈chop≈dice≈julienne, pour≈ladle≈scoop, fry≈sauté≈cook, mix≈stir≈whisk, pan≈wok≈pot≈skillet, mitts/oven gloves≈gloves, pills/medication≈tablets, container/box/tub≈tupperware, jar≈jug≈bottle (for liquids). When a narrative phrase matches an option via a paraphrase, count it as supported.\n"
    "- Most-specific wins — when two or more options have keyword overlap with the narrative, pick the one whose *combination* of verb + object is most concretely supported (e.g. 'cleaned counter near washing machine' is a worse match for 'washed clothes in washing machine' than 'opened the washing machine door and loaded clothes' is, even though both touch the noun 'washing machine'). Prefer options that match BOTH the action and the object.\n"
    "- Default to A-D: if at least one option's verb OR distinctive object class is supported by the activity evidence (including paraphrases above), pick that letter. Choose E only when no option's action and no option's object — even loosely — appears anywhere in the activity stream.\n"
)

_QUESTION_TYPE_GUIDANCE = {
    # Ordered-procedure questions use MEMORA's typed evidence interface.
    # because search_patterns / get_semantic_evidence are typed-memory tools.
    ("SRoutine", "memora"): (
        "\n\n## SRoutine QUESTION RULES\n"
        "- This is a routine/strategy question. Use `search_patterns` or `get_semantic_evidence` to find consolidated cross-episode evidence.\n"
        "- Prefer reusable procedure evidence with concrete object transitions or ordered steps when available.\n"
        "- For object-specific questions, answer from concrete object handling, micro-sequence, ordered steps, or choice phrases.\n"
        "- If search_patterns returns specific routine evidence matching the object/action, choose the supported A-D option instead of E.\n"
        "- Choose E only after consolidated regularities, semantic evidence, and narrative evidence are missing or unrelated.\n"
    ),

    # Event recall prioritizes evidence tied to the named video and only uses
    # participant-level regularities as context.
    ("ERecall", "memora"): (
        "\n\n## ERecall QUESTION RULES\n"
        "- This is an episodic event-recall question about a named video. Use `get_video_activities(video_id)` first for the full chronological activity stream; do NOT use `get_video_summary` (it truncates).\n"
        "- Each activity returned has TWO fields: a generic `summary` (e.g. 'cleaning the kitchen after a meal') and a concrete `narrative` field that contains the literal verbs and objects (e.g. 'picks up a plate and a knife ... washes the plate and knife in the sink'). The `narrative` field is where the verb+object evidence for A-D lives — scan every `narrative` for nouns from each option before deciding.\n"
        "- If after scanning every `narrative` no option is supported, make ONE focused follow-up: `get_narrative_evidence(video_id + distinctive nouns from A-D)`.\n"
        "- Do not answer from generalised preference/routine cards for event-specific questions; this is the only question type where event evidence outranks pattern evidence.\n"
        "- If `search_patterns` returns participant-level regularities or reusable procedures, treat them as cross-episode context rather than evidence of what happened in this specific video. Only episode evidence tied to the named video should determine the answer.\n"
        "- **Object-specific sub-pattern (\"In video Vxxx, P01 handled <object>. What did they do with it?\")**: the choices are typically 2-3 consecutive actions on the named object (e.g. \"turn on, then set\" vs. \"set, then open fridge\"). For this sub-pattern, the primary evidence is the `action_breakdown` list inside each activity — a chronological array of `{timestamp, action, object}` steps. Scan it for the FIRST 2 (or 3) consecutive steps whose `object` is the named object, then pick the choice that lists the same verb pair in the same order. Reject choices whose action sequence never touches the named object.\n"
        "- **Object-specific anti-abstention**: for the sub-pattern above, if the activity stream alone is too coarse to resolve the verb pair, do ONE follow-up `get_narrative_evidence(video_id + object)` and inspect its `action_breakdown`. Choose E only after both the stream and the narrative breakdown fail to surface a consecutive action pair involving the object.\n"
        + _ERECALL_DEFAULT_TO_AD_HINT
    ),
    ("ERecall", "graph_2d"): (
        "\n\n## ERecall QUESTION RULES\n"
        "- Event-recall question about a named video. Use `search(video_id)` first to retrieve activity nodes (with `previous_action` / `next_action` / `objects_involved`).\n"
        "- If the activity nodes do not clearly support exactly one of A-D, make ONE focused follow-up `search(video_id + distinctive nouns from A-D)`; if the question names a specific object, also call `get_object_history(<object>)`.\n"
        + _ERECALL_DEFAULT_TO_AD_HINT
    ),
    ("ERecall", "flat_1d"): (
        "\n\n## ERecall QUESTION RULES\n"
        "- Event-recall question about a named video. Use `search(video_id)` first to retrieve the chronological segment dump.\n"
        "- If the segment dump does not clearly support exactly one of A-D, make ONE focused follow-up `search(video_id + distinctive nouns from A-D)`.\n"
        + _ERECALL_DEFAULT_TO_AD_HINT
    ),
}

def select_question_type_guidance(question_type: str, memory_type: str) -> str:
    """Return guidance for one paper-defined question and memory type."""
    return _QUESTION_TYPE_GUIDANCE.get((question_type, memory_type), "")
