#!/usr/bin/env python3
"""Task environment shared by EAM-QA and MEMORA-Planning.

The environment exposes the selected memory condition's tools, records the
agent interaction, and evaluates the final task response.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

from memora.evaluation.eam_qa.prompts import (
    EAM_QA_ABSTENTION_GUIDANCE,
    FORCED_ANSWER_PROMPT_EAM_QA,
    select_question_type_guidance,
)
from memora.memory_agent.memory_representations import (
    FLAT_1D_SYSTEM_PROMPT,
    FORCED_ANSWER_PROMPT,
    FORCED_ANSWER_PROMPT_SHORT_ANSWER,
    GRAPH_2D_SYSTEM_PROMPT,
    SHORT_ANSWER_GUIDANCE,
    TYPED_MEMORY_SYSTEM_PROMPT,
    TYPED_MEMORY_SYSTEM_PROMPT_WITH_CATEGORY,
    create_memory_tools,
)

logger = logging.getLogger(__name__)


class AgentEnvironment:
    """ReAct environment shared by the released memory conditions."""

    def __init__(
        self,
        memory_file: str,
        max_iterations: int = 5,
        use_category_search: bool = False,
        include_tips: bool = False,
        memory_type: str = "memora",
    ):
        """
        Initialize the environment.

        Args:
            memory_file: Path to a participant memory JSON/JSONL file.
            max_iterations: Max tool call iterations
            use_category_search: If True, use category-aware search with an optional
                category argument. If False, search across all memory categories.
            memory_type: ``memora`` for the four typed stores, ``graph_2d`` for
                the graph baseline, or ``flat_1d`` for the chronological-text baseline.
        """
        self.memory_type = memory_type

        self.memory_tools = create_memory_tools(
            memory_type,
            memory_file,
            include_tips=include_tips,
        )
        self.max_iterations = max_iterations
        self.use_category_search = use_category_search
        self._current_system_prompt = self._select_system_prompt()
        self._current_tools = self._select_tools()

        # State
        self.task_data = None
        self.conversation_history = []
        self.iteration_count = 0
        self._tool_calls_log = []
        self._responses_log = []
        self._force_answer_mode = False
        self._previous_search_queries: dict = {}

    def _select_system_prompt(self) -> str:
        """Select the appropriate system prompt based on memory_type."""
        if self.memory_type == "graph_2d":
            return GRAPH_2D_SYSTEM_PROMPT
        if self.memory_type == "flat_1d":
            return FLAT_1D_SYSTEM_PROMPT
        if self.use_category_search:
            return TYPED_MEMORY_SYSTEM_PROMPT_WITH_CATEGORY
        return TYPED_MEMORY_SYSTEM_PROMPT

    def _select_tools(self) -> list:
        """Select the appropriate tool definitions based on memory_type."""
        return self.memory_tools.get_tools_definition(allow_category=self.use_category_search)

    def reset(self, task: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset environment for a new task.

        Args:
            task: Task dict with question, video_id, etc.

        Returns:
            observation: Dict with system_prompt, tools, conversation
            info: Metadata dict
        """
        self.task_data = task
        self.conversation_history = []
        self.iteration_count = 0
        self._tool_calls_log = []
        self._responses_log = []
        self._force_answer_mode = False
        self._previous_search_queries = {}

        # Extract task fields
        question = task.get("question", "")
        video_id = task.get("video_id", "")
        participant_id = task.get("participant_id", "")
        cognitive_level = task.get("cognitive_level", "episodic")
        ask_turn_id = task.get("ask_turn_id")
        timestamp_seconds = task.get("timestamp_seconds")
        choices = task.get("choices", [])
        qa_type = task.get("qa_type", "")
        question_type = str(qa_type or "unknown").strip()
        if hasattr(self.memory_tools, "set_question_context"):
            self.memory_tools.set_question_context(question_type)

        # Determine question type: episodic vs semantic.
        # Episodic questions need temporal context (time filtering)
        # Semantic questions need participant context (cross-video aggregation)
        is_episodic = cognitive_level == "episodic" or question_type == "ERecall"

        # Determine video_ids to use:
        # - Cross-video semantic questions usually provide video_ids.
        # - Single-video event-recall questions usually provide video_id.
        # - Otherwise: use all participant videos
        task_video_ids = task.get("video_ids")
        if not task_video_ids and video_id:
            task_video_ids = [video_id]

        # Set context based on question type
        if is_episodic and video_id and (ask_turn_id is not None or timestamp_seconds is not None):
            # Episodic questions use only evidence available through the
            # question segment (or through the explicit timestamp).
            self.memory_tools.set_temporal_context(
                video_id=video_id,
                ask_turn_id=ask_turn_id,
                time_threshold_seconds=timestamp_seconds
            )
        else:
            # Semantic questions: set participant context for cross-video aggregation.
            # No time filtering needed - these questions aggregate across all videos
            self.memory_tools.set_participant_context(
                participant_id=participant_id,
                video_ids=task_video_ids
            )

        # Build user message with choices
        user_message = question
        if choices:
            user_message += "\n\nOptions:"
            for i, choice in enumerate(choices):
                user_message += f"\n{chr(65+i)}) {choice}"

        # Initialize conversation
        self.conversation_history = [
            {"role": "user", "content": user_message}
        ]

        # Build observation (prompt and tool schema depend on memory type)
        self._current_system_prompt = self._select_system_prompt()
        # Append short-answer instructions when running without MC choices
        if not task.get("is_multiple_choice", True):
            self._current_system_prompt += SHORT_ANSWER_GUIDANCE
        elif len(choices) >= 5:
            self._current_system_prompt += EAM_QA_ABSTENTION_GUIDANCE
        self._current_system_prompt += select_question_type_guidance(
            question_type, self.memory_type
        )
        self._current_tools = self._select_tools()
        observation = {
            "system_prompt": self._current_system_prompt,
            "tools": self._current_tools,
            "conversation": self.conversation_history
        }

        # Build info
        info = {
            "question_id": task.get("question_id", ""),
            "video_id": video_id,
            "participant_id": participant_id,
            "ground_truth": task.get("ground_truth", ""),
            "qa_type": question_type,
            "cognitive_level": cognitive_level,
            "is_multiple_choice": task.get("is_multiple_choice", True),
            "choices": choices
        }

        return observation, info

    def step(self, action: Dict[str, Any]) -> Tuple[Optional[Dict], float, bool, Dict]:
        """
        Execute one step.

        Args:
            action: Model's response with content and tool_calls

        Returns:
            observation: Next observation (None if done)
            reward: Score
            done: Whether episode is complete
            info: Additional info
        """
        self.iteration_count += 1

        response_text = action.get("content", "")
        tool_calls = action.get("tool_calls", [])

        # Log response
        self._responses_log.append(response_text)

        # Add assistant response to conversation
        if tool_calls:
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text,
                "tool_calls": tool_calls
            })
        else:
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })

        # Check for tool calls
        if tool_calls and self.iteration_count < self.max_iterations:
            tool_results = []

            for tc in tool_calls:
                tool_name = tc.get("function", {}).get("name", "")
                raw_arguments = tc.get("function", {}).get("arguments", "{}")
                try:
                    arguments = json.loads(raw_arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must decode to a JSON object")
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    result = {
                        "error": f"Invalid arguments for {tool_name or 'unknown tool'}: {exc}"
                    }
                    self._tool_calls_log.append({
                        "tool": tool_name,
                        "arguments": raw_arguments,
                        "result": result,
                    })
                    tool_results.append({
                        "tool_call_id": tc.get("id", ""),
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    continue

                if tool_name == "search":
                    query_raw = arguments.get("query", "")
                    query_norm = query_raw.strip().lower()
                else:
                    query_raw = ""
                    query_norm = ""

                if tool_name == "search" and query_norm in self._previous_search_queries:
                    previous = self._previous_search_queries[query_norm]
                    result = {
                        "_duplicate": True,
                        "_message": (
                            f'You already searched "{query_raw}"{previous}. '
                            "Try a different query, such as an individual object name."
                        ),
                    }
                    self._tool_calls_log.append({
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": result,
                    })
                    tool_results.append({
                        "tool_call_id": tc.get("id", ""),
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    continue

                # Execute tool
                try:
                    result = self.memory_tools.execute_tool(tool_name, arguments)
                except Exception as exc:
                    logger.exception("Tool execution failed: %s", tool_name)
                    result = {
                        "error": f"{tool_name or 'Tool'} failed: {type(exc).__name__}: {exc}"
                    }
                # --- Track search queries & annotate empty results ---
                if tool_name == "search":
                    total = 0
                    if isinstance(result, dict):
                        summary = result.get("_summary", {})
                        if isinstance(summary, dict) and "total" in summary:
                            total = int(summary.get("total", 0) or 0)
                        else:
                            total = sum(
                                len(result.get(category, []) or [])
                                for category in ("objects", "activities", "environment", "patterns")
                            )

                    status = (
                        f" (returned {total} results)"
                        if total > 0
                        else " (returned 0 results)"
                    )
                    self._previous_search_queries[query_norm] = status

                    if total == 0 and isinstance(result, dict):
                        result["_hint"] = (
                            "No results found. Try searching for specific OBJECT NAMES "
                            "(e.g., \"knife\", \"bowl\", \"cutting board\") instead of "
                            "action phrases or categories."
                        )

                # Log
                self._tool_calls_log.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result
                })

                tool_results.append({
                    "tool_call_id": tc.get("id", ""),
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })

            # Add tool results to conversation
            for tr in tool_results:
                self.conversation_history.append(tr)

            # ================================================================
            # Check if the next iteration will be the last.
            # If so, enable forced-answer mode to request a final response.
            # ================================================================
            if self.iteration_count >= self.max_iterations - 1:
                self._force_answer_mode = True
                logger.info(" Enabling forced answer mode (last iteration)")

            # Return next observation
            observation = {
                "system_prompt": self._current_system_prompt,
                "tools": self._current_tools,
                "conversation": self.conversation_history,
                "tool_results": tool_results,
                "force_answer_mode": self._force_answer_mode
            }

            return observation, 0.0, False, {"tool_results": tool_results}

        # ================================================================
        # Tool calls at max iteration: force an answer.
        # ================================================================
        if tool_calls and self.iteration_count >= self.max_iterations:
            logger.info(" Max iterations reached with tool calls - forcing answer")
            self._force_answer_mode = True

            # Keep the structured tool schema in the observation so the
            # downstream API call still has a valid ``tools`` parameter
            # — required by OpenAI when the conversation history contains
            # any prior tool_calls / role:"tool" messages. The agent will
            # use ``tool_choice="none"`` (or, for vLLM, post-filter the
            # response) to actually prevent the model from calling tools.
            observation = {
                "system_prompt": self._current_system_prompt,
                "tools": self._current_tools,
                "conversation": self.conversation_history,
                "force_answer_mode": True,
                "tool_results": []
            }

            return observation, 0.0, False, {"forced_answer": True, "tool_results": []}

        # ================================================================
        # No tool calls — could be a legitimate final answer OR a
        # *premature exit* (e.g. Gemma sometimes emits just the literal
        # token "thought\n" with no ``call:`` block and no plan). In the
        # latter case, the session should not terminate; forced-answer mode
        # gives the model one more attempt when iteration budget remains.
        # The heuristic considers a response a
        # premature exit when:
        #   1. it is shorter than ``_PREMATURE_EXIT_CHAR_THRESHOLD``
        #      characters after stripping, AND
        #   2. it contains none of the expected answer markers
        #      ("Plan:", "Answer:", numbered list, etc.), AND
        #   3. forced-answer mode has not already been used (avoid
        #      double-forcing if the model still fails after a forced
        #      attempt — fall through to evaluation in that case).
        # ================================================================
        if (
            self.iteration_count < self.max_iterations
            and not self._force_answer_mode
            and self._looks_like_premature_exit(response_text)
        ):
            logger.info(
                " Detected premature exit (short response, no Plan/Answer marker); "
                "enabling forced answer mode for one more attempt"
            )
            self._force_answer_mode = True
            observation = {
                "system_prompt": self._current_system_prompt,
                "tools": self._current_tools,
                "conversation": self.conversation_history,
                "force_answer_mode": True,
                "tool_results": []
            }
            return observation, 0.0, False, {
                "premature_exit_recovered": True,
                "tool_results": []
            }

        # No tool calls or max iterations - evaluate answer
        reward, evaluation_info = self._evaluate_answer(response_text)

        return None, reward, True, {
            "final_answer": response_text,
            "evaluation_info": evaluation_info,
            "iterations": self.iteration_count,
            "tool_calls": self._tool_calls_log
        }

    # ------------------------------------------------------------------
    # Premature-exit detector (used by step())
    # ------------------------------------------------------------------
    # Number of stripped characters below which a no-tool-call
    # response as a *premature exit*. 30 chars comfortably accommodates
    # short legitimate answers like "Answer: A" while filtering single-
    # word emissions like "thought\n" (8 chars including the newline).
    _PREMATURE_EXIT_CHAR_THRESHOLD = 30
    _ANSWER_MARKERS = (
        "plan:", "answer:", "**plan**", "**answer**",
        "step 1", "step1", "1.", "1)", "- step",
        "the plan", "final plan", "final answer",
    )

    def _looks_like_premature_exit(self, response_text: str) -> bool:
        """Heuristic: True if the response is too short *and* contains no
        recognised plan/answer marker — i.e. the model emitted a stub like
        "thought\\n" without actually producing a final answer."""
        if not response_text:
            return True
        text = response_text.strip().lower()
        if len(text) >= self._PREMATURE_EXIT_CHAR_THRESHOLD:
            return False
        return not any(marker in text for marker in self._ANSWER_MARKERS)

    def _evaluate_answer(self, response: str) -> Tuple[float, Dict]:
        """
        Evaluate the answer.

        Returns:
            (score, evaluation_info)
        """
        ground_truth = self.task_data.get("ground_truth", "")
        is_mc = self.task_data.get("is_multiple_choice", True)

        if is_mc:
            # Extract answer letter from response
            # Look for "Answer: X" pattern (handles "Answer: A", "Answer: A)", "Answer: A) Yes", etc.)
            # Support A-E for questions with an abstain option.
            match = re.search(r'Answer:\s*([A-E])[\)\s]', response, re.IGNORECASE)
            if match:
                predicted = match.group(1).upper()
            else:
                # Try "**Answer: X**" format (markdown bold)
                match = re.search(r'\*\*Answer:\s*([A-E])[\)\s]', response, re.IGNORECASE)
                if match:
                    predicted = match.group(1).upper()
                else:
                    # Try to find standalone letter (last resort) - support A-E
                    match = re.search(r'\b([A-E])\b', response)
                    predicted = match.group(1).upper() if match else ""

            # Compare with ground truth
            if isinstance(ground_truth, str):
                gt_letter = ground_truth.strip().upper()
                # Support A-E for questions with an abstain option.
                if len(gt_letter) == 1 and gt_letter in "ABCDE":
                    correct = predicted == gt_letter
                else:
                    # Ground truth is the answer text, need to map to letter
                    choices = self.task_data.get("choices", [])
                    gt_idx = None
                    for i, c in enumerate(choices):
                        # Remove "A) ", "B) " prefix if present for comparison
                        choice_text = c.strip()
                        if choice_text.startswith(chr(65 + i) + ") "):
                            choice_text = choice_text[3:].strip()
                        if choice_text.lower() == ground_truth.strip().lower():
                            gt_idx = i
                            break
                    if gt_idx is not None:
                        gt_letter = chr(65 + gt_idx)
                        correct = predicted == gt_letter
                    else:
                        # Fallback: string matching in response
                        correct = ground_truth.strip().lower() in response.lower()
            elif isinstance(ground_truth, dict):
                # Handle recall questions with dict ground truth containing timestamp.
                timestamp = ground_truth.get("timestamp")
                choices = self.task_data.get("choices", [])

                if timestamp is not None and choices:
                    # Map timestamp to correct choice letter
                    # Choices are like: "Between 90-135 seconds", "Between 135-180 seconds", etc.
                    gt_letter = None
                    for i, choice in enumerate(choices):
                        # Parse choice to extract time range
                        # Match patterns like "Between 90-135 seconds" or "Around 270-272 seconds"
                        range_match = re.search(r'(\d+)-(\d+)', choice)
                        if range_match:
                            start = float(range_match.group(1))
                            end = float(range_match.group(2))
                            if start <= timestamp <= end:
                                gt_letter = chr(65 + i)
                                break
                        # Also check for "Around X-Y" pattern
                        around_match = re.search(r'Around\s+(\d+)-(\d+)', choice, re.IGNORECASE)
                        if around_match:
                            start = float(around_match.group(1))
                            end = float(around_match.group(2))
                            if start <= timestamp <= end:
                                gt_letter = chr(65 + i)
                                break

                    if gt_letter:
                        correct = predicted == gt_letter
                    else:
                        # Timestamp doesn't match any range - check if model's answer is reasonable
                        # This should not happen for well-formed temporal-recall questions.
                        correct = False
                else:
                    # Dict but no timestamp: check for event-verification format.
                    answer_value = ground_truth.get("answer")
                    if answer_value is not None:
                        # Event-verification question: answer is A (Yes) or B (No).
                        # answer=True means "Yes" (A), answer=False means "No" (B)
                        expected = "A" if answer_value else "B"
                        correct = predicted == expected
                    else:
                        # Fallback: check for explicit order metadata.
                        is_correct_order = ground_truth.get("is_correct_order")
                        if is_correct_order is not None:
                            expected = "A" if is_correct_order else "B"
                            correct = predicted == expected
                        else:
                            # If ground_truth is dict but has neither answer nor is_correct_order,
                            # try to use correct_answer from task_data
                            correct_answer = self.task_data.get("correct_answer", "")
                            if correct_answer:
                                correct = predicted == correct_answer.strip().upper()
                            else:
                                correct = False
            else:
                correct = False

            score = 1.0 if correct else 0.0
            return score, {
                "predicted": predicted,
                "ground_truth": ground_truth,
                "correct": correct
            }
        else:
            # Free-form evaluation: Token F1 (SQuAD-style) + substring match
            gt_str = str(ground_truth).strip()

            # Extract the answer portion from the response
            pred_text = self._extract_freeform_answer(response)

            # Token F1 (primary metric for short-answer)
            token_f1, token_precision, token_recall = self._compute_token_f1(
                pred_text, gt_str
            )

            # Substring containment fallback
            contains = gt_str.lower() in response.lower()

            # Score: use Token F1 as the reward signal
            score = token_f1

            return score, {
                "predicted": pred_text[:200],
                "ground_truth": gt_str,
                "token_f1": token_f1,
                "token_precision": token_precision,
                "token_recall": token_recall,
                "contains_match": contains,
                "correct": token_f1 >= 0.5,
            }

    @staticmethod
    def _normalize_tokens(text: str) -> List[str]:
        """Lowercase, strip punctuation, split into word tokens."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if t]

    @staticmethod
    def _compute_token_f1(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
        """SQuAD-style token F1 between prediction and ground truth.

        Returns (f1, precision, recall).
        """
        pred_tokens = AgentEnvironment._normalize_tokens(prediction)
        gt_tokens = AgentEnvironment._normalize_tokens(ground_truth)
        if not gt_tokens and not pred_tokens:
            return 1.0, 1.0, 1.0
        if not gt_tokens or not pred_tokens:
            return 0.0, 0.0, 0.0
        from collections import Counter
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            return 0.0, 0.0, 0.0
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1, precision, recall

    @staticmethod
    def _extract_freeform_answer(response: str) -> str:
        """Pull the final short-answer text from a ReAct response.

        Looks for patterns like 'Answer: ...' or 'Final answer: ...' and
        returns the text after it.  Falls back to the last non-empty line.
        """
        # Try "**Answer: some text**" or "Answer: some text"
        match = re.search(
            r'\*?\*?(?:Final\s+)?Answer:\s*\*?\*?\s*(.+?)(?:\*\*|\n|$)',
            response, re.IGNORECASE
        )
        if match:
            ans = match.group(1).strip().rstrip("*").strip()
            if ans:
                return ans
        # Fallback: last non-empty line
        for line in reversed(response.strip().splitlines()):
            line = line.strip()
            if line:
                return line
        return response.strip()

    def get_conversation_for_model(self) -> List[Dict[str, str]]:
        """Get conversation in format suitable for model input"""
        messages = [
            {"role": "system", "content": self._current_system_prompt}
        ]

        for msg in self.conversation_history:
            if msg["role"] == "tool":
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "name": msg.get("name", ""),
                    "content": msg.get("content", "")
                })
            else:
                messages.append({
                    "role": msg["role"],
                    "content": msg.get("content", "")
                })
                if "tool_calls" in msg:
                    messages[-1]["tool_calls"] = msg["tool_calls"]

        # ================================================================
        # Add the forced-answer prompt when in forced mode.
        # ================================================================
        if self._force_answer_mode:
            # Summarize gathered information
            gathered_info = self._summarize_gathered_info()

            is_mc = self.task_data.get("is_multiple_choice", True)
            choices = self.task_data.get("choices", [])
            if not is_mc:
                forced_prompt = FORCED_ANSWER_PROMPT_SHORT_ANSWER
            elif len(choices) >= 5:
                forced_prompt = FORCED_ANSWER_PROMPT_EAM_QA
            else:
                forced_prompt = FORCED_ANSWER_PROMPT
            if gathered_info:
                forced_prompt += f"\n\n## Information Gathered So Far:\n{gathered_info}"

            messages.append({
                "role": "user",
                "content": forced_prompt
            })
            logger.info(" Added forced answer prompt to conversation")

        return messages

    def _summarize_gathered_info(self) -> str:
        """Summarize all tool call results for forced answer mode."""
        if not self._tool_calls_log:
            return "No search results available."

        summary_parts = []
        for i, call in enumerate(self._tool_calls_log, 1):
            tool = call.get("tool", "unknown")
            args = call.get("arguments", {})
            result = call.get("result", {})

            # Format result preview
            if isinstance(result, list):
                if result:
                    result_preview = f"{len(result)} results found"
                else:
                    result_preview = "No results"
            elif isinstance(result, dict):
                if "error" in result:
                    result_preview = f"Error: {result['error']}"
                else:
                    # Show key fields
                    keys = list(result.keys())[:5]
                    result_preview = f"Found: {', '.join(keys)}"
            else:
                result_preview = str(result)[:100]

            query = args.get("query", args.get("time_seconds", args.get("zone_name", str(args))))
            summary_parts.append(f"{i}. {tool}(\"{query}\") → {result_preview}")

        return "\n".join(summary_parts)
