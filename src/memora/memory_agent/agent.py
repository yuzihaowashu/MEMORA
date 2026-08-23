#!/usr/bin/env python3
"""
memory_agent.agent - ReAct-style memory-guided agent for MEMORA.

Contains:
  - VLLMInference: vLLM-based inference engine with tool call parsing
  - OpenAIInference: OpenAI API-based inference (drop-in alternative)
  - run_agent_loop: Multi-turn agent loop (observe → think → act → observe)

The agent iteratively searches memory through tools and reasons until
it can answer a question, produce a plan, or reaches the max iteration limit.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional

from memora.memory_agent.agent_environment import AgentEnvironment
from memora.memory_agent.tool_call_parsing import parse_tool_calls
from memora.pipeline.api_client import resolve_api_credentials


# ============================================================================
# Chat-template helpers
# ============================================================================
def _sanitize_messages_for_alternating_roles(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collapse ``tool`` role messages and assistant ``tool_calls`` into a
    plain user/assistant alternation.

    Some chat templates (Gemma-3, Gemma-4, Phi-4, ...) reject the OpenAI-style
    ``tool`` role and require strict user/assistant alternation. Qwen and
    OpenAI tolerate ``tool``. This sanitizer is only used as a fallback when
    the template renderer raises ``Conversation roles must alternate`` etc.

    Strategy:
      - Keep ``system`` as-is (one or more leading systems are fine).
      - For an ``assistant`` message that carries ``tool_calls`` but has no
        ``content``, render the tool calls as text inside the assistant turn
        so they survive the chat template.
      - Convert each ``tool`` message into a ``user`` message tagged
        ``[tool_result] <name>: <content>``.
      - Merge consecutive same-role messages so user/assistant strictly
        alternate.
    """

    flat: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        if role == "tool":
            name = msg.get("name", "tool")
            flat.append({
                "role": "user",
                "content": f"[tool_result] {name}: {content}",
            })
            continue
        if role == "assistant" and msg.get("tool_calls"):
            tc_lines = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                tname = fn.get("name", "")
                targs = fn.get("arguments", "")
                tc_lines.append(f"<tool_call>{json.dumps({'name': tname, 'arguments': targs})}</tool_call>")
            tc_text = "\n".join(tc_lines)
            full = (content + ("\n" if content else "") + tc_text).strip()
            flat.append({"role": "assistant", "content": full})
            continue
        flat.append({"role": role, "content": content})

    # Merge consecutive same-role messages (e.g. two user turns from a
    # tool-result followed by a forced-answer prompt).
    merged: List[Dict[str, Any]] = []
    for msg in flat:
        if (
            merged
            and merged[-1]["role"] == msg["role"]
            and merged[-1]["role"] in ("user", "assistant", "system")
        ):
            merged[-1]["content"] = (
                (merged[-1].get("content") or "") + "\n\n" + (msg.get("content") or "")
            ).strip()
        else:
            merged.append(dict(msg))

    # Drop any leading non-user message after system blocks. Most strict
    # templates expect: system* (optional), then user, then alternation.
    if merged:
        # Skip past system messages
        i = 0
        while i < len(merged) and merged[i]["role"] == "system":
            i += 1
        # If the first non-system message is assistant, prepend a stub user
        if i < len(merged) and merged[i]["role"] == "assistant":
            merged.insert(i, {"role": "user", "content": "(continue)"})

    return merged


# ============================================================================
# vLLM Inference Engine
# ============================================================================
class VLLMInference:
    """vLLM-based inference with tool calling support."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.85,
        temperature: float = 0.3,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        disable_custom_all_reduce: bool = True,
    ):
        self.model_name = model_name
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.disable_custom_all_reduce = disable_custom_all_reduce
        self.enforce_eager = os.getenv("MEMORA_VLLM_ENFORCE_EAGER", "").lower() in {
            "1", "true", "yes", "on",
        }
        raw_ct = os.getenv("MEMORA_CHAT_TEMPLATE_KWARGS_JSON", "").strip()
        self._chat_template_kwargs: Optional[Dict[str, Any]] = None
        if raw_ct:
            try:
                self._chat_template_kwargs = json.loads(raw_ct)
            except json.JSONDecodeError as e:
                raise ValueError(
                    "MEMORA_CHAT_TEMPLATE_KWARGS_JSON must be valid JSON "
                    "(e.g. {\"enable_thinking\": false} for Qwen3 thinking-off)"
                ) from e

    def _chat_template_extras(self) -> Dict[str, Any]:
        """Optional kwargs for ``tokenizer.apply_chat_template`` (Qwen3, etc.)."""
        if self._chat_template_kwargs:
            return {"chat_template_kwargs": self._chat_template_kwargs}
        return {}

    def initialize(self):
        """Initialize vLLM (lazy, on first call)."""
        if self.llm is not None:
            return

        print(" Initializing vLLM engine...")
        print(f"   Model: {self.model_name}")
        print(f"   Tensor Parallel Size: {self.tensor_parallel_size}")
        print(f"   Max Model Length: {self.max_model_len}")

        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer

            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True,
                disable_custom_all_reduce=self.disable_custom_all_reduce,
                enforce_eager=self.enforce_eager,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Anti-degeneracy guards. Without these, Qwen3-30B-A3B and Qwen3-32B
            # occasionally enter a repetition loop inside the ReAct "final plan"
            # step and exhaust max_tokens without emitting the actual plan.
            # Values are conservative so small models and GPT-compatible chat
            # templates are unaffected.
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                repetition_penalty=1.05,
                presence_penalty=0.3,
            )

            print(" vLLM engine ready")
        except Exception as e:
            print(f" vLLM init failed: {e}")
            raise

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None,
        force_no_tool_call: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate response from messages.

        Args:
            messages: OpenAI-style chat message list.
            tools: Optional tool schema list (kept available even when the
                caller intends to forbid new tool calls — see
                ``force_no_tool_call``).
            force_no_tool_call: If True, the caller intends this completion
                to be the *final* answer (the env has entered forced-answer
                mode). For vLLM this is a no-op semantically — the agent
                loop already post-filters any tool_calls that come back —
                but the signature is shared with
                :class:`APIInference.chat_completion`.

        Returns:
            Dict with 'content' and optional 'tool_calls'
        """
        del force_no_tool_call  # vLLM relies on post-filtering, not on a flag
        if self.llm is None:
            self.initialize()

        # Format prompt with tools
        _ct = self._chat_template_extras()
        if tools and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **_ct,
                )
            except Exception:
                # Fallback without tools
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        **_ct,
                    )
                except Exception:
                    # Final fallback for chat templates that require strict
                    # user/assistant alternation (e.g. Gemma-3, Gemma-4):
                    # collapse `tool` messages and assistant tool_calls into
                    # the surrounding user/assistant turns.
                    prompt = self.tokenizer.apply_chat_template(
                        _sanitize_messages_for_alternating_roles(messages),
                        add_generation_prompt=True,
                        tokenize=False,
                        **_ct,
                    )
        else:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **_ct,
                )
            except Exception:
                prompt = self.tokenizer.apply_chat_template(
                    _sanitize_messages_for_alternating_roles(messages),
                    add_generation_prompt=True,
                    tokenize=False,
                    **_ct,
                )

        # Generate
        outputs = self.llm.generate([prompt], self.sampling_params)
        response_text = outputs[0].outputs[0].text.strip()

        # Parse tool calls from response
        tool_calls = self._parse_tool_calls(response_text)

        # Extract content (remove tool call syntax)
        content = self._extract_content(response_text)

        return {
            "content": content,
            "tool_calls": tool_calls
        }

    def _parse_tool_calls(self, text: str) -> List[Dict]:
        """Parse tool calls from model response."""
        return parse_tool_calls(text)

    def _extract_content(self, text: str) -> str:
        """Extract content, removing tool call syntax."""
        content = re.sub(
            r'<tool_call>[\s\S]*?</tool_call>',
            '',
            text,
            flags=re.IGNORECASE,
        )
        content = re.sub(
            r'<function\s*=\s*[A-Za-z_][\w\-]*\s*>[\s\S]*?</function\s*>',
            '',
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(
            r'(?:search(?:_objects|_environment|_activities|_patterns)?|'
            r'get_state_at_time|get_object_history)\s*\([^)]+\)',
            '',
            content,
            flags=re.IGNORECASE,
        )
        return content.strip()


# ============================================================================
# OpenAI API Inference Engine (drop-in alternative to VLLMInference)
# ============================================================================
class OpenAIInference:
    """OpenAI-compatible API inference with native function calling support.

    Implements the same ``chat_completion`` interface as ``VLLMInference``
    so it can be used as a drop-in replacement everywhere the agent loop
    expects a model object.

    The caller supplies the model name and generation budget explicitly so the
    same interface can target hosted APIs or a local OpenAI-compatible server.
    """

    _DEFAULT_MAX_TOKENS = 2048

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.3,
        top_p: float = 0.95,
        max_tokens: int = 0,
        **_kwargs,
    ):
        self.model_name = model_name
        self.api_base, self.api_key = resolve_api_credentials(
            api_base=api_base,
            api_key=api_key,
            default_base=None,
        )
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens if max_tokens > 0 else self._DEFAULT_MAX_TOKENS
        self._client = None

        # Pick up chat_template_kwargs the same way VLLMInference does, so
        # the agent can talk to a vLLM OpenAI-compatible server with
        # {"enable_thinking": false} (Qwen3 family) via extra_body. The
        # vLLM 0.19 OpenAI CLI does NOT accept --chat-template-kwargs, so
        # the only practical channel is per-request extra_body.
        raw_ct = os.getenv("MEMORA_CHAT_TEMPLATE_KWARGS_JSON", "").strip()
        self._chat_template_kwargs: Optional[Dict[str, Any]] = None
        if raw_ct:
            try:
                self._chat_template_kwargs = json.loads(raw_ct)
            except json.JSONDecodeError:
                # Mirror VLLMInference: fail loudly on bad JSON.
                raise ValueError(
                    "MEMORA_CHAT_TEMPLATE_KWARGS_JSON must be valid JSON "
                    "(e.g. {\"enable_thinking\": false} for Qwen3 thinking-off)"
                )

    def initialize(self):
        """Initialize the OpenAI-compatible client."""
        self._ensure_client()

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai  — required for API mode")
        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.api_base:
            kwargs["base_url"] = self.api_base
        self._client = OpenAI(**kwargs)
        print(f" OpenAI-compatible API client ready (model={self.model_name})")

    # ------------------------------------------------------------------
    # Convert the MEMORA tool schema to OpenAI function-calling format.
    # ------------------------------------------------------------------
    @staticmethod
    def _convert_tools(tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
        """Convert MEMORA tool schemas to OpenAI function-calling format."""
        if not tools:
            return None
        oai_tools = []
        for t in tools:
            if t.get("type") == "function":
                oai_tools.append(t)
                continue
            fn = t.get("function", t)
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                },
            })
        return oai_tools or None

    # ------------------------------------------------------------------
    # Main entry point — same signature as VLLMInference.chat_completion
    # ------------------------------------------------------------------
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        force_no_tool_call: bool = False,
    ) -> Dict[str, Any]:
        """OpenAI chat completion with two important safety nets.

        1.  **Conversation sanitisation** — before sending we walk the
            messages list and drop any orphan ``role:"tool"`` messages
            (i.e. tool messages NOT immediately preceded by an assistant
            message whose ``tool_calls`` list contains the matching
            ``tool_call_id``). OpenAI rejects orphan tool messages with
            ``400 — messages with role 'tool' must be a response to a
            preceeding message with 'tool_calls'``.

        2.  **Forced-answer mode** — when the agent is in forced-answer
            mode (``force_no_tool_call=True``), we still keep the
            ``tools`` schema in the request — required by OpenAI
            whenever the messages contain any prior tool_calls — but we
            set ``tool_choice="none"`` so the model cannot emit a new
            tool call.
        """
        self._ensure_client()

        messages = self._sanitize_messages_for_openai_tool_calls(messages)

        oai_tools = self._convert_tools(tools)

        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools
            kwargs["tool_choice"] = "none" if force_no_tool_call else "auto"
        # Forward optional chat-template kwargs (e.g. enable_thinking=false)
        # to an OpenAI-compatible vLLM server via extra_body. Leave the
        # environment variable unset for endpoints that do not accept it.
        if self._chat_template_kwargs:
            kwargs["extra_body"] = {
                "chat_template_kwargs": self._chat_template_kwargs,
            }

        resp = self._client.chat.completions.create(**kwargs)
        content, tool_calls = self._extract_response(resp)

        return {"content": content, "tool_calls": tool_calls}

    @staticmethod
    def _extract_response(resp) -> tuple:
        choice = resp.choices[0]
        msg = choice.message

        content = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        return content, tool_calls

    @staticmethod
    def _sanitize_messages_for_openai_tool_calls(
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Strip orphan tool / dangling tool_call references that OpenAI
        rejects with ``400 — messages with role 'tool' must be a response
        to a preceeding message with 'tool_calls'``.

        We do TWO passes:

        1.  **Drop orphan tool messages.** A ``role:"tool"`` message is
            kept only if the most recent assistant message has a
            ``tool_calls`` entry whose ``id`` matches the tool message's
            ``tool_call_id``. Otherwise it is dropped.
        2.  **Drop dangling tool_call ids.** After pass 1, any
            ``tool_calls`` entry on an assistant message that does not
            have a matching subsequent tool response is removed. If the
            assistant message ends up with an empty ``tool_calls`` list
            we delete the key (an assistant message with no content and
            ``tool_calls: []`` is itself invalid).

        The input ``messages`` list is not mutated in place; a new list
        is returned. When sanitisation actually removes anything we log
        a warning so we can spot upstream bugs.
        """
        if not messages:
            return messages

        out: List[Dict[str, Any]] = []
        last_assistant_tc_ids: set = set()
        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                tcid = msg.get("tool_call_id", "")
                if tcid and tcid in last_assistant_tc_ids:
                    out.append(msg)
                # else: orphan tool, drop silently (logged below in pass 2)
            elif role == "assistant":
                # Copy to avoid mutating caller-owned dict.
                assistant_message = dict(msg)
                tool_calls = assistant_message.get("tool_calls") or []
                last_assistant_tc_ids = {
                    tool_call.get("id")
                    for tool_call in tool_calls
                    if tool_call.get("id")
                }
                out.append(assistant_message)
            else:
                last_assistant_tc_ids = set()
                out.append(msg)

        # Pass 2: prune dangling tool_call ids on assistant messages, and
        # log any sanitisation activity exactly once.
        # We walk the sanitised list forward, collecting which tool_call
        # ids are answered by a subsequent ``tool`` message before the
        # next ``assistant`` message.
        n = len(out)
        cleaned: List[Dict[str, Any]] = []
        for i, msg in enumerate(out):
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                cleaned.append(msg)
                continue
            tc_ids = {tc.get("id") for tc in msg["tool_calls"] if tc.get("id")}
            answered: set = set()
            for j in range(i + 1, n):
                nxt = out[j]
                if nxt.get("role") == "assistant":
                    break
                if nxt.get("role") == "tool":
                    tcid = nxt.get("tool_call_id", "")
                    if tcid in tc_ids:
                        answered.add(tcid)
            kept_tcs = [
                tc for tc in msg["tool_calls"] if tc.get("id") in answered
            ]
            if kept_tcs:
                assistant_message = dict(msg)
                assistant_message["tool_calls"] = kept_tcs
                cleaned.append(assistant_message)
            else:
                # No matching tool responses for any of the tool_calls;
                # remove the ``tool_calls`` key. If content is also empty
                # we drop the assistant message entirely (an empty
                # assistant message with no tool_calls is not useful and
                # would itself be rejected by some servers).
                assistant_message = dict(msg)
                assistant_message.pop("tool_calls", None)
                if not (assistant_message.get("content") or "").strip():
                    continue
                cleaned.append(assistant_message)

        if len(cleaned) != len(messages):
            import logging
            logging.getLogger(__name__).warning(
                "[APIInference] sanitised conversation history: "
                "%d -> %d messages (dropped orphan tool / dangling tool_calls)",
                len(messages), len(cleaned),
            )
        return cleaned


# ============================================================================
# Agent Loop (ReAct-style)
# ============================================================================
def run_agent_loop(
    model: VLLMInference,
    env: AgentEnvironment,
    task: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run one ReAct episode: observe → think → act → observe until done.

    Args:
        model: VLLMInference instance
        env: AgentEnvironment instance
        task: Task dict with question, video_id, choices, ground_truth, etc.
        verbose: Whether to print intermediate steps

    Returns:
        Dict with the final response, deterministic task evaluation, iterations,
        and tool calls.
    """
    observation, info = env.reset(task)

    done = False
    reward = 0.0
    final_info = {}
    iteration = 0

    while not done:
        iteration += 1

        is_forced = observation.get("force_answer_mode", False)
        if is_forced and verbose:
            print(f"\n[Iteration {iteration}]  FORCED ANSWER MODE")

        # Get model response.
        #
        # Forced-mode tool handling: keep the ``tools`` schema available even
        # after the environment requests a final answer. OpenAI's chat API
        # available even when in forced-answer mode. OpenAI's chat-completions
        # API rejects any payload whose ``messages`` array contains role:"tool"
        # or assistant tool_calls **without** a ``tools`` parameter (returns
        # ``400 — messages with role 'tool' must be a response to a preceding
        # message with 'tool_calls'``). We instead signal "no new tool calls"
        # to the inference engine via ``force_no_tool_call=True`` so the API
        # path can set ``tool_choice="none"`` (vLLM has no such requirement
        # and post-filters the response anyway).
        messages = env.get_conversation_for_model()
        tools_to_use = observation.get("tools", [])
        response = model.chat_completion(
            messages,
            tools=tools_to_use,
            force_no_tool_call=is_forced,
        )

        # In forced mode, discard any hallucinated tool calls (vLLM may emit
        # them despite the forced-answer prompt).
        if is_forced and response.get("tool_calls"):
            response["tool_calls"] = []

        # Verbose logging
        if verbose and response.get("tool_calls") and not is_forced:
            print(f"\n[Iteration {iteration}] TOOL CALLS:")
            for tc in response["tool_calls"]:
                tool_name = tc['function']['name']
                tool_args = tc['function']['arguments']
                print(f"   {tool_name}({tool_args})")

        # Step environment
        observation, reward, done, step_info = env.step(response)
        final_info.update(step_info)

        # Verbose logging for results
        if verbose and step_info.get("tool_results"):
            print(f"\n[Iteration {iteration}] RETRIEVED RESULTS:")
            for tr in step_info["tool_results"]:
                tool_name = tr.get("name", "unknown")
                result_content = tr.get("content", "{}")
                try:
                    result_json = json.loads(result_content)
                    formatted = json.dumps(result_json, indent=4, ensure_ascii=False)
                    if len(formatted) > 2000:
                        formatted = formatted[:2000] + "\n    ... [truncated]"
                    print(f"\n   {tool_name} returned:")
                    for line in formatted.split('\n'):
                        print(f"    {line}")
                except json.JSONDecodeError:
                    print(f"\n   {tool_name} returned: {result_content[:500]}")

        if verbose and response.get("content"):
            print(f"\n[Iteration {iteration}] MODEL RESPONSE:")
            response_text = response['content']
            if len(response_text) > 3000:
                print(f"  {response_text[:3000]}")
                print("  ... [response truncated]")
            else:
                for line in response_text.split('\n'):
                    print(f"  {line}")

    return {
        "reward": reward,
        "iterations": iteration,
        "final_info": final_info,
        "info": info
    }
