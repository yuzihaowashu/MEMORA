"""Prompt profile used by the released MEMORA-Planning evaluation."""

PLANNING_FORCED_ANSWER_PROMPT = """You have reached the maximum number of search iterations. Based on ALL the information you have gathered, generate your final plan NOW.

## ABSOLUTE RULES — DO NOT VIOLATE
- The system has **DISABLED all tools**. Any function call will be silently DISCARDED.
- DO NOT emit `<tool_call>`, `<function=`, or `<parameter=` XML tags — ANY such output is treated as a failure.
- DO NOT emit `<think>` or `<reasoning>` blocks. Start your reply directly with `Plan:`.
- Your reply MUST start with the literal string `Plan:` followed by a newline.

## Plan requirements
1. The plan MUST accomplish the original TASK. Do NOT follow a retrieved procedure that is about a different task.
2. Ground EVERY step in retrieved information — use EXACT color, material, and location from memory for every object.
3. If memory shows this person has specific habits, reflect those in your plan.

## Output Format — EVERY object MUST include color + material + location:
Plan:
1. [VERB] the [COLOR] [MATERIAL] [OBJECT_NAME] [from/on/at EXACT_LOCATION]
2. [VERB] the [COLOR] [MATERIAL] [OBJECT_NAME] [from/on/at EXACT_LOCATION]
...

Example: "Grasp the red ceramic kettle from the stove"
Do NOT write bare nouns like "the pan" — always describe objects with characteristics.

Generate the plan now, starting with `Plan:`:"""
