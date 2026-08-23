"""Memory-grounded task environment for MEMORA-Planning."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from memora.evaluation.planning.parser import extract_plan_from_response
from memora.evaluation.planning.context import PlanningContextMixin
from memora.evaluation.planning.tasks import resolve_task_instruction
from memora.evaluation.planning.prompts import (
    DEFAULT_PLANNER_PROFILE,
    PLANNER_PROFILES,
    PLANNING_FORCED_ANSWER_PROMPT,
    PLANNING_SYSTEM_PROMPT_FLAT_1D,
)
from memora.memory_agent.agent_environment import AgentEnvironment

logger = logging.getLogger(__name__)

class PlanningEnvironment(PlanningContextMixin):
    """Thin wrapper around :class:`AgentEnvironment` that swaps the QA prompt
    for a planning prompt and replaces answer-evaluation with plan extraction.

    The underlying tool-call / step loop is unchanged — only ``reset``,
    ``_evaluate_answer``, and ``get_conversation_for_model`` are overridden.
    """

    def __init__(
        self,
        memory_file: str,
        max_iterations: int = 8,
        memory_type: str = "memora",
        include_tips: bool = False,
        planner_profile: Optional[str] = DEFAULT_PLANNER_PROFILE,
    ):
        self._env = AgentEnvironment(
            memory_file=memory_file,
            max_iterations=max_iterations,
            use_category_search=False,
            include_tips=include_tips,
            memory_type=memory_type,
        )
        self.max_iterations = max_iterations
        self.memory_type = memory_type
        self.planner_profile = planner_profile
        if memory_type == "flat_1d":
            self._system_prompt = PLANNING_SYSTEM_PROMPT_FLAT_1D
        else:
            if planner_profile is None:
                raise ValueError("MEMORA and Graph-2D require a planner profile")
            if planner_profile not in PLANNER_PROFILES:
                raise ValueError(
                    f"Unknown planner profile {planner_profile!r}; choose from "
                    f"{sorted(PLANNER_PROFILES)}"
                )
            self._system_prompt = PLANNER_PROFILES[planner_profile]

        self._extracted_plan: List[str] = []
        self._raw_final_response: str = ""

    # -- delegated attributes -----------------------------------------------

    @property
    def task_data(self):
        return self._env.task_data

    @property
    def conversation_history(self):
        return self._env.conversation_history

    @property
    def iteration_count(self):
        return self._env.iteration_count

    # -- public API (same signature as AgentEnvironment) -----------------

    def reset(self, task: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment for a planning task.

        Transforms the planning-task dict into the format
        ``AgentEnvironment.reset`` expects, then replaces the system prompt.
        Injects pre-searched memory context into the user message.
        """
        self._extracted_plan = []
        self._raw_final_response = ""
        self._current_task_id = task.get("task_id", "")

        participant_id = task.get("participant_id", "")
        video_id = task.get("video_id", "")
        task_query = resolve_task_instruction(task)

        env_task = {
            "question": task_query,
            "video_id": video_id,
            "participant_id": participant_id,
            "cognitive_level": "semantic",
            "video_ids": task.get("video_ids", [video_id] if video_id else None),
            "choices": [],
            "is_multiple_choice": False,
            "ground_truth": "",
            "question_id": task.get("task_id", ""),
            "qa_type": "planning",
        }

        observation, info = self._env.reset(env_task)

        self._env._current_system_prompt = self._system_prompt
        observation["system_prompt"] = self._system_prompt

        if self.memory_type != "no_memory":
            memory_ctx, retrieval_confidence = self._build_memory_context(task_query)
            if memory_ctx:
                # The memory-grounded prompt uses a dedicated routine-skill-first
                # strategy and its own
                # explicit tool repertoire (get_routine_skill / get_preferences
                # / search_objects). Override the generic per-confidence
                # guidance so the user message does not collide with the
                # system prompt.
                if self.planner_profile == "memora_full":
                    guidance = (
                        f"## Strategy for \"{task_query}\" (memory-grounded routine-first)\n"
                        f"The auto-retrieved context above may already contain a routine_skill "
                        f"candidate, raw episodes, or scene objects. Treat it as a HINT.\n\n"
                        f"1. **FIRST tool call** — `get_routine_skill(goal_query=\"<the task's main goal>\")`. "
                        f"If the top hit has `similarity >= 0.6`, adopt its `canonical_steps` as your plan skeleton. "
                        f"If `0.5 <= similarity < 0.6` adapt it loosely. Below 0.5, fall back to `search_activities` once.\n"
                        f"2. **Optional second call** — `get_preferences(query=\"<what to personalise>\")` to fetch "
                        f"this person's stable habits relevant to the task (storage, hand preference, cleanup style).\n"
                        f"3. **THEN ground objects** — `search_objects(name=...)` for at most 2 key objects you will mention. "
                        f"Every object in your final plan must include color + material + location.\n\n"
                        f"Stop after at most 4 tool calls and emit your final plan as `Plan: 1. ... 2. ...`."
                    )
                elif retrieval_confidence == "high":
                    guidance = (
                        f"GOOD MATCH: The procedure sequences above appear relevant to "
                        f"\"{task_query}\". Use them as a template — adapt the steps for the "
                        f"current task (substitute the target object if needed).\n\n"
                        f"Search for each KEY OBJECT in the task to get its exact appearance "
                        f"(color, material) and current location."
                    )
                elif retrieval_confidence == "medium":
                    guidance = (
                        f"PARTIAL MATCH: The activities above are related but don't closely "
                        f"match \"{task_query}\".\n"
                        f"**Strategy**: Use your GENERAL KNOWLEDGE for the plan structure. "
                        f"Use memory to ground objects — search for each key object to find "
                        f"its exact color, material, and location in this kitchen.\n\n"
                        f"Start by searching for each KEY OBJECT mentioned in the task."
                    )
                else:
                    guidance = (
                        f"NOVEL TASK: Very little relevant memory was found for "
                        f"\"{task_query}\".\n"
                        f"**Strategy**: Plan using your GENERAL KNOWLEDGE. "
                        f"Search for each key OBJECT to ground it with appearance and location "
                        f"from this specific kitchen.\n\n"
                        f"Start by searching for each KEY OBJECT mentioned in the task."
                    )
                ctx_header = (
                    f"## Memory Context (auto-retrieved, confidence: "
                    f"{retrieval_confidence})"
                )
                enriched_msg = (
                    f"Task: {task_query}\n"
                    f"Participant: {participant_id}\n\n"
                    f"{ctx_header}\n"
                    f"{memory_ctx}\n\n"
                    f"{guidance}"
                )
                if self._env.conversation_history:
                    self._env.conversation_history[0]["content"] = enriched_msg
                    observation["conversation"] = self._env.conversation_history

        info.update({
            "task_id": task.get("task_id", ""),
            "task_type": task.get("task_type", ""),
            "ground_truth_steps": task.get("ground_truth_steps", []),
        })

        return observation, info

    def step(self, action: Dict[str, Any]) -> Tuple[Optional[Dict], float, bool, Dict]:
        """Delegate to the inner env, but intercept the terminal state to
        extract a plan instead of evaluating an MC answer.
        """

        response_text = action.get("content", "")
        observation, reward, done, step_info = self._env.step(action)

        if done:
            self._raw_final_response = response_text
            self._extracted_plan = extract_plan_from_response(response_text)
            step_info["generated_plan"] = self._extracted_plan
            step_info["raw_response"] = self._raw_final_response
            reward = 0.0

        if not done and observation is not None:
            observation["system_prompt"] = self._system_prompt

        return observation, reward, done, step_info

    def get_conversation_for_model(self) -> List[Dict[str, str]]:
        """Build the message list with the planning system prompt and, when
        in forced-answer mode, swap in the planning-specific forced prompt.

        Also truncates tool outputs when total context gets too long to
        prevent exceeding max_model_len (32768 tokens ~ 25k chars).
        """
        MAX_CHARS = 24000

        messages = self._env.get_conversation_for_model()

        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = self._system_prompt

        if self._env._force_answer_mode:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user" and "maximum number of search iterations" in messages[i].get("content", ""):
                    messages[i]["content"] = PLANNING_FORCED_ANSWER_PROMPT
                    break

            # ---------------------------------------------------------------
            # Forced-answer hardening: scrub prior assistant turns so that
            # models like Qwen3.x (which mimic the XML <tool_call> pattern
            # they emitted in earlier iterations) don't echo more XML at
            # iter N. We rewrite each assistant message that carried
            # `tool_calls` into a plain text summary of WHICH tools it
            # called. The tool RESPONSES (role=tool) are kept verbatim
            # because they contain the actual retrieved memory content.
            # ---------------------------------------------------------------
            for m in messages:
                if m.get("role") != "assistant":
                    continue
                tool_calls = m.get("tool_calls") or []
                if not tool_calls:
                    continue
                # Preserve a compact record of the requested evidence so the
                # model still sees the search trajectory without the XML form.
                summaries = []
                for tool_call in tool_calls:
                    fn = (tool_call or {}).get("function") or {}
                    name = fn.get("name") or "tool"
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        args_str = args
                    else:
                        try:
                            import json as _json
                            args_str = _json.dumps(args, ensure_ascii=False)
                        except Exception:
                            args_str = str(args)
                    summaries.append(f"  - {name}({args_str})")
                m["content"] = (
                    (m.get("content") or "")
                    + ("\n" if m.get("content") else "")
                    + "Tools I called this turn:\n"
                    + "\n".join(summaries)
                )
                # Drop the structured tool_calls so the chat template
                # cannot regenerate the <tool_call> XML in this turn.
                m.pop("tool_calls", None)

        total = sum(len(m.get("content", "")) for m in messages)
        if total > MAX_CHARS:
            for m in messages:
                if m["role"] == "tool" and len(m.get("content", "")) > 1500:
                    m["content"] = m["content"][:1500] + "\n... [truncated for context length]"
            total = sum(len(m.get("content", "")) for m in messages)
            if total > MAX_CHARS:
                for m in messages:
                    if m["role"] == "tool" and len(m.get("content", "")) > 800:
                        m["content"] = m["content"][:800] + "\n... [truncated]"

        return messages
