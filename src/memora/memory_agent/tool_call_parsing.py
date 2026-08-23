"""Normalize model-specific textual tool calls into one tool-call schema.

The parser accepts the tagged, fenced-JSON, Hermes, Gemma, and plain-text
formats emitted by the released planning backbones. The agent consumes only
the normalized OpenAI-style records returned here.
"""

import json
import re
from typing import Any, Dict, List

def parse_tool_calls(text: str) -> List[Dict]:
    """Parse tool calls from model response."""
    tool_calls = []

    # ================================================================
    # Pattern 0: <tool_call> JSON </tool_call> format (HIGHEST PRIORITY)
    # This is what Qwen models typically output
    # ================================================================
    tool_call_tag_pattern = r'<tool_call>\s*([\s\S]*?)\s*</tool_call>'
    tool_call_matches = re.findall(tool_call_tag_pattern, text, re.IGNORECASE)

    for json_str in tool_call_matches:
        json_str = json_str.strip()
        try:
            call_data = json.loads(json_str)
            tool_name = call_data.get("name", "")
            arguments = call_data.get("arguments", {})

            # Handle both formats:
            # 1. "arguments": {"query": "..."}  (dict)
            # 2. "arguments": "{\"query\": \"...\"}"  (escaped string)
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass

            args_str = json.dumps(arguments) if isinstance(arguments, dict) else arguments

            if tool_name:
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": args_str
                    }
                })
        except json.JSONDecodeError:
            json_match = re.search(
                r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}',
                json_str
            )
            if json_match:
                try:
                    call_data = json.loads(json_match.group())
                    tool_name = call_data.get("name", "")
                    arguments = call_data.get("arguments", {})
                    if tool_name:
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments)
                            }
                        })
                except json.JSONDecodeError:
                    pass

            # ----- Qwen3.5 / vLLM 0.19+ "corrupted hermes" fallback -----
            # When vLLM injects ``<function=`` as a chat-template prefix
            # for Qwen3.5/3.6 but the generation never emits ``</function>``,
            # the body looks like:
            #   <function=NAME", "arguments": {"k": "v", ...}}
            # (i.e. the model continued the prefix as Python-style kwargs
            # for an OpenAI-tools call object, then closed with mismatched
            # braces).  We rescue it by extracting the function name and
            # the embedded arguments JSON object.
            if not any(tc.get("function", {}).get("name") for tc in tool_calls):
                corrupted = re.search(
                    r'<function\s*=\s*([A-Za-z_][\w\-]*)\s*"\s*,\s*'
                    r'"arguments"\s*:\s*(\{[\s\S]*?\})\s*\}?\s*$',
                    json_str,
                )
                if corrupted:
                    tool_name = corrupted.group(1).strip()
                    args_blob = corrupted.group(2)
                    try:
                        arguments = json.loads(args_blob)
                    except json.JSONDecodeError:
                        arguments = {"_raw": args_blob}
                    if tool_name:
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments),
                            },
                        })

    # ================================================================
    # Pattern 0b: ```tool_call ... ``` markdown code-fence format
    # Gemma-3 / Gemma-4 emit fenced markdown instead of XML tags.
    # Also catches the closely related ```json fence when the body
    # contains a {"name": ..., "arguments": ...} payload.
    # ================================================================
    if not tool_calls:
        fence_patterns = [
            r'```\s*tool_call\s*\n?([\s\S]*?)```',
            r'```\s*json\s*\n?([\s\S]*?)```',
            # Plain fence with no language tag - Gemma sometimes emits
            # ```\n{...}\n``` after copying the example layout from the
            # system prompt. Gated below by the "name"/"arguments" check.
            r'```\s*\n([\s\S]*?)```',
        ]
        for fence_pat in fence_patterns:
            for json_str in re.findall(fence_pat, text, re.IGNORECASE):
                json_str = json_str.strip()
                if not json_str:
                    continue
                # Must look like a tool-call payload to avoid catching
                # generic ```json reasoning blocks.
                if '"name"' not in json_str or '"arguments"' not in json_str:
                    continue
                try:
                    call_data = json.loads(json_str)
                except json.JSONDecodeError:
                    json_match = re.search(
                        r'\{[\s\S]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[\s\S]*?\}\s*\}',
                        json_str,
                    )
                    if not json_match:
                        continue
                    try:
                        call_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        continue
                tool_name = call_data.get("name", "")
                arguments = call_data.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                args_str = json.dumps(arguments) if isinstance(arguments, dict) else arguments
                if tool_name:
                    tool_calls.append({
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": args_str,
                        },
                    })
            if tool_calls:
                break

    # If we found tool_call tags, deduplicate and return early
    if tool_calls:
        seen = set()
        unique_tool_calls = []
        for tc in tool_calls:
            func = tc.get("function", {})
            key = (func.get("name", ""), func.get("arguments", ""))
            if key not in seen:
                seen.add(key)
                unique_tool_calls.append(tc)

        MAX_TOOL_CALLS_PER_RESPONSE = 3
        if len(unique_tool_calls) > MAX_TOOL_CALLS_PER_RESPONSE:
            unique_tool_calls = unique_tool_calls[:MAX_TOOL_CALLS_PER_RESPONSE]

        return unique_tool_calls

    # ================================================================
    # Pattern 0c: Hermes / Qwen3.5 XML form
    #   <function=name>
    #     <parameter=k1>value1</parameter>
    #     <parameter=k2>value2</parameter>
    #   </function>
    # New Qwen3.5/3.6 chat templates emit this when ``tools`` is passed
    # to ``apply_chat_template``. Optionally wrapped in ``<tool_call>``,
    # which Pattern 0 already strips, so scan the raw text.
    # ================================================================
    hermes_func_pattern = re.compile(
        r'<function\s*=\s*([A-Za-z_][\w\-]*)\s*>([\s\S]*?)</function\s*>',
        re.IGNORECASE,
    )
    hermes_param_pattern = re.compile(
        r'<parameter\s*=\s*([A-Za-z_][\w\-]*)\s*>([\s\S]*?)</parameter\s*>',
        re.IGNORECASE,
    )
    for func_match in hermes_func_pattern.finditer(text):
        tool_name = func_match.group(1).strip()
        body = func_match.group(2)
        if not tool_name:
            continue
        args: Dict[str, Any] = {}
        for pname, pval in hermes_param_pattern.findall(body):
            pval = pval.strip()
            # Try to JSON-parse first (so dict/list args survive),
            # but fall back to raw string for plain values.
            try:
                args[pname.strip()] = json.loads(pval)
            except (json.JSONDecodeError, ValueError):
                args[pname.strip()] = pval
        tool_calls.append({
            "id": f"call_{len(tool_calls)}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(args),
            },
        })

    if tool_calls:
        seen = set()
        unique_tool_calls = []
        for tc in tool_calls:
            func = tc.get("function", {})
            key = (func.get("name", ""), func.get("arguments", ""))
            if key not in seen:
                seen.add(key)
                unique_tool_calls.append(tc)
        return unique_tool_calls[:3]

    # ================================================================
    # Pattern 0d: Gemma 4 native form
    #   call:func_name{key:value,key2:{nested:val},key3:val with spaces}
    # Keys and string values are unquoted (Python-repr-ish), and string
    # values can contain spaces / colons / question marks (i.e. natural
    # English). We do a brace-aware top-level split, then recursively
    # rebuild each segment into JSON.
    # ================================================================
    def _gemma_parse_arglist(blob: str) -> Dict[str, Any]:
        """Parse Gemma-4 ``key:value,key:value`` arglists. Values can be
        either nested ``{...}`` objects or bare strings (potentially with
        spaces, commas-free)."""
        out: Dict[str, Any] = {}
        i, n, depth = 0, len(blob), 0
        seg_start = 0
        segments: List[str] = []
        while i < n:
            ch = blob[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth = max(0, depth - 1)
            elif ch == ',' and depth == 0:
                segments.append(blob[seg_start:i])
                seg_start = i + 1
            i += 1
        if seg_start < n:
            segments.append(blob[seg_start:])
        for seg in segments:
            seg = seg.strip()
            if not seg or ':' not in seg:
                continue
            key, _, val = seg.partition(':')
            key = key.strip().strip('"').strip("'")
            val = val.strip()
            if val.startswith('{') and val.endswith('}'):
                out[key] = _gemma_parse_arglist(val[1:-1])
            else:
                # Strip surrounding quotes if any, else keep raw string.
                if (val.startswith('"') and val.endswith('"')) or \
                   (val.startswith("'") and val.endswith("'")):
                    out[key] = val[1:-1]
                else:
                    out[key] = val
        return out

    gemma_call_pattern = re.compile(
        r'\bcall\s*:\s*([A-Za-z_][\w\-]*)\s*\{([\s\S]+?)\}\s*(?=\bcall\s*:|\Z|\n\s*\n)',
        re.IGNORECASE,
    )
    for func_match in gemma_call_pattern.finditer(text):
        tool_name = func_match.group(1).strip()
        blob = func_match.group(2).strip()
        if not tool_name:
            continue
        try:
            args = _gemma_parse_arglist(blob)
        except Exception:
            args = {"query": blob}
        if not args:
            args = {"query": blob}
        tool_calls.append({
            "id": f"call_{len(tool_calls)}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(args),
            },
        })

    if tool_calls:
        seen = set()
        unique_tool_calls = []
        for tc in tool_calls:
            func = tc.get("function", {})
            key = (func.get("name", ""), func.get("arguments", ""))
            if key not in seen:
                seen.add(key)
                unique_tool_calls.append(tc)
        return unique_tool_calls[:3]

    # ================================================================
    # Pattern 1: Unified search - search("query")
    # ================================================================
    search_pattern = r'\bsearch\s*\(\s*["\']([^"\']+)["\']\s*\)'
    search_matches = re.findall(search_pattern, text, re.IGNORECASE)
    for query in search_matches:
        tool_calls.append({
            "id": f"call_{len(tool_calls)}",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": json.dumps({"query": query})
            }
        })

    # ================================================================
    # Pattern 2: get_state_at_time(time)
    # ================================================================
    time_pattern = r'get_state_at_time\s*\(\s*([0-9.]+)\s*\)'
    time_matches = re.findall(time_pattern, text, re.IGNORECASE)
    for time_val in time_matches:
        tool_calls.append({
            "id": f"call_{len(tool_calls)}",
            "type": "function",
            "function": {
                "name": "get_state_at_time",
                "arguments": json.dumps({"time_seconds": float(time_val)})
            }
        })

    # ================================================================
    # Pattern 2b: get_object_history("object")
    # ================================================================
    history_pattern = r'get_object_history\s*\(\s*["\']([^"\']+)["\']\s*\)'
    history_matches = re.findall(history_pattern, text, re.IGNORECASE)
    for obj_query in history_matches:
        tool_calls.append({
            "id": f"call_{len(tool_calls)}",
            "type": "function",
            "function": {
                "name": "get_object_history",
                "arguments": json.dumps({"object_query": obj_query})
            }
        })

    # ================================================================
    # Pattern 3: JSON format without <tool_call> tags
    # ================================================================
    json_pattern = (
        r'\{[^{}]*"name"\s*:\s*"(search(?:_\w+)?|get_state_at_time|get_object_history)"'
        r'[^{}]*"arguments"\s*:\s*(\{[^{}]+\})[^{}]*\}'
    )
    json_matches = re.findall(json_pattern, text, re.DOTALL)
    for match in json_matches:
        tool_name, args_str = match
        try:
            args = json.loads(args_str)
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args)
                }
            })
        except json.JSONDecodeError:
            pass

    # ================================================================
    # Pattern 4: additional typed search tool calls
    # ================================================================
    fallback_patterns = [
        r'(search_objects|search_environment|search_activities|search_patterns)\s*\(\s*["\']([^"\']+)["\']\s*\)',
    ]

    for pattern in fallback_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            tool_name, query = match
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": tool_name.lower(),
                    "arguments": json.dumps({"query": query})
                }
            })

    # ================================================================
    # Final deduplication and limiting
    # ================================================================
    if tool_calls:
        seen = set()
        unique_tool_calls = []
        for tc in tool_calls:
            func = tc.get("function", {})
            key = (func.get("name", ""), func.get("arguments", ""))
            if key not in seen:
                seen.add(key)
                unique_tool_calls.append(tc)

        MAX_TOOL_CALLS = 3
        tool_calls = unique_tool_calls[:MAX_TOOL_CALLS]

    return tool_calls
