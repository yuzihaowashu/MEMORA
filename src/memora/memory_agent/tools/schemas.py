"""OpenAI-compatible tool schemas for MEMORA retrieval."""

from typing import Any, Dict, List


def build_tools_definition(allow_category: bool = False) -> List[Dict[str, Any]]:
    """Get OpenAI-compatible tool definitions."""
    search_properties: Dict[str, Any] = {
        "query": {
            "type": "string",
            "description": "Full natural-language semantic query. Prefer the original question wording over isolated keywords."
        }
    }
    if allow_category:
        search_properties["category"] = {
            "type": "string",
            "description": "Optional. Search only this category to reduce noise. One of: 'objects', 'activities', 'environment', 'patterns'."
        }
    search_tool = {
        "type": "function",
        "function": {
            "name": "search",
            "description": """Search across all memory categories. Returns results from:
- objects: Items and their locations/states (e.g., "plate on counter", "dirty cup")
- activities: Timestamped actions (e.g., "picks up plate at 1:30", "washes dish")
- environment: Locations and spatial info (e.g., "sink area", "left of counter")
- patterns: Behavioral habits (e.g., "prefers left hand", "coffee routine")

Use this as a fallback or broad semantic lookup. Query with the full question first; only then simplify to keywords if needed.""",
            "parameters": {
                "type": "object",
                "properties": search_properties,
                "required": ["query"]
            }
        }
    }
    return [
        search_tool,
        {
            "type": "function",
            "function": {
                "name": "get_state_at_time",
            "description": """Get a complete state snapshot at a specific time point.
Returns: visible objects (with HISTORICAL state/location), environment, and current activity.

Object states are reconstructed from state_history when available.

Use for questions like:
- "At 0.0s, was X visible?"
- "What was happening at time X?"
- "Around turn 30, when X was on counter, was Y visible?" (uses historical state)

For general "when did X happen" questions, use search() instead.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time_seconds": {
                            "type": "number",
                            "description": "Time point in seconds (e.g., 0.0, 30.5, 120.0)"
                        }
                    },
                    "required": ["time_seconds"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_object_history",
                "description": """Get the history of an object's states and locations.

Returns:
- all_states_observed: ALL states the object has been in (for "Was X ever dirty?" questions)
- all_locations_observed: ALL locations the object has been in
- state_history: Full timeline of state/location changes

Use for:
- "Was the plate ever dirty?" → Check if 'dirty' in all_states_observed
- "Did the cup move during the video?" → Check all_locations_observed
- "What states has the fork been in?" → Return all_states_observed

This is the best tool for "Was X ever Y?" and "Did X move/change?" questions.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_query": {
                            "type": "string",
                            "description": "Object name to track (e.g., 'plate', 'cup', 'fork', 'knife')"
                        }
                    },
                    "required": ["object_query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_video_summary",
                "description": """Return a chronological narrative summary for a video.

Use for broad episodic event-recall questions like:
- "What was P01 doing in video P01_103?"
- "In video P01_109, what happened?"

This is better than object search when the question asks for the overall event/story.
If the question names a target object ("handled a tea cup", "what did they do with the cup?"),
use get_narrative_evidence with the video id and object instead.
Do not use this as the first tool for habit/preference questions; use count_object_uses or compare_objects.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Optional video id, e.g. P01_103. If omitted, uses current video context."
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_video_activities",
                "description": """Return the chronological activity stream for a single video.

Unlike get_video_summary (which keeps only ~6 evenly-spaced snippets), this tool
returns EVERY activity for the named video. This is the right tool for
event-recall questions of the form:
- "What was P0X doing in video P0X_YYY?"
- "In video P0X_YYY, what did P0X do with Z?"

By default (compact=true) it returns one summary line per activity. Set compact=false only if you need the
detailed narrative + action_breakdown for every activity.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video id, e.g. P01_103. If omitted, uses current video context."
                        },
                        "compact": {
                            "type": "boolean",
                            "description": "If true (default), returns one summary line per activity. If false, returns full compact-activity records."
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_narrative_evidence",
                "description": """Search raw activity summaries and detailed narratives.

Use first for object-specific event recall in a named video, e.g.
"P01_104 tea cup handled what did they do".

Also use when get_object_history misses an object, or for open-vocabulary entities
that may not be tracked in object_registry (e.g., trash bag, rolling pin,
cling film).""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Object, event, or phrase to search in activity narratives."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_local_narrative",
                "description": """Return local activity narrative and nearby objects around a timestamp.

Use when an object is not tracked by name, or for questions asking what happened
around a time/turn. This exposes candidates_in_window without hallucinating an
entity match.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time_seconds": {
                            "type": "number",
                            "description": "Center timestamp in seconds. If omitted, uses the current question time when available."
                        },
                        "window": {
                            "type": "number",
                            "description": "Window size in seconds on each side. Defaults to 30."
                        },
                        "video_id": {
                            "type": "string",
                            "description": "Optional video id, e.g. P01_103."
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_semantic_evidence",
                "description": """Return compact typed evidence for semantic memory questions.

Use first for habit/preference/routine questions asking "typically", "usually",
"prefer", "after X", "approach", "strategy", or "routine". This combines
pattern evidence, transition aggregation, and candidate comparison without
returning broad episodic activity logs. If the question has answer-choice
alternatives, pass all A-D choices in candidates when possible so the tool can
score each option. If it asks "after X", pass X as trigger_query.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Full semantic question, including important answer-choice words when useful."
                        },
                        "candidate_a": {
                            "type": "string",
                            "description": "Optional first answer-choice alternative."
                        },
                        "candidate_b": {
                            "type": "string",
                            "description": "Optional second answer-choice alternative."
                        },
                        "candidates": {
                            "type": "object",
                            "description": "Optional full answer-choice map, e.g. {\"A\":\"spatula\", \"B\":\"fork\", \"C\":\"whisk\", \"D\":\"no pattern\"}. Prefer this for MC semantic questions."
                        },
                        "choice_a": {
                            "type": "string",
                            "description": "Optional answer choice A text."
                        },
                        "choice_b": {
                            "type": "string",
                            "description": "Optional answer choice B text."
                        },
                        "choice_c": {
                            "type": "string",
                            "description": "Optional answer choice C text."
                        },
                        "choice_d": {
                            "type": "string",
                            "description": "Optional answer choice D text."
                        },
                        "context_query": {
                            "type": "string",
                            "description": "Optional task context, e.g. preparing food, cleaning, ingredient preparation."
                        },
                        "trigger_query": {
                            "type": "string",
                            "description": "Optional trigger action for after/next questions, e.g. picking up sesame oil."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_patterns",
                "description": """Search MEMORA's inferred pattern layer.

Use first for semantic memory questions about preferences, habits, routines, and
action patterns, especially questions asking "typically", "usually",
"approach", "strategy", or "routine". Returns compact evidence from storage
preferences, organizational habits, workflow patterns, and action sequences.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Full natural-language semantic question or focused pattern query."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "count_object_uses",
                "description": """Count how often an object appears in activities, optionally under a context.

Use first for habit/preference/action-pattern questions where frequency matters,
especially questions comparing concrete object/action counts. For broad strategy,
routine, or workflow questions, use search_patterns first.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_query": {
                            "type": "string",
                            "description": "Object to count, e.g. spatula."
                        },
                        "context_query": {
                            "type": "string",
                            "description": "Optional context filter, e.g. preparing food or cleaning."
                        }
                    },
                    "required": ["object_query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_action_transitions",
                "description": """Aggregate what happens after a trigger action across activities.

Use first for semantic habit/preference/routine questions asking "after X, what happens next?",
"immediately after X", or "what does the user typically do next?". If the question has two
candidate answers, pass them as candidate_a and candidate_b; if there are more alternatives,
pass all A-D options in candidates so the tool can count support.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "trigger_query": {
                            "type": "string",
                            "description": "The action or event that should happen before the answer, e.g. 'picks up cutting board' or 'turns off tap'."
                        },
                        "candidate_a": {
                            "type": "string",
                            "description": "Optional first candidate next action or object."
                        },
                        "candidate_b": {
                            "type": "string",
                            "description": "Optional second candidate next action or object."
                        },
                        "candidates": {
                            "type": "object",
                            "description": "Optional full answer-choice map, e.g. {\"A\":\"wash it\", \"B\":\"put it away\", \"C\":\"cut something\", \"D\":\"dry it\"}."
                        }
                    },
                    "required": ["trigger_query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compare_objects",
                "description": """Compare two object candidates under the same context.

Use first for questions comparing two candidates, e.g. "does the user usually use a spatula or a spoon?".""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_a": {"type": "string", "description": "First candidate object."},
                        "query_b": {"type": "string", "description": "Second candidate object."},
                        "context_query": {"type": "string", "description": "Optional task context, e.g. preparing food."}
                    },
                    "required": ["query_a", "query_b"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_routine_skill",
                "description": """Retrieve consolidated routine templates by semantic match on the goal.

Returns ranked routines from Inferred Knowledge. Each match exposes
`canonical_steps`, `key_objects`, and `supporting_episodes` so a planning agent
can adopt the top hit as a plan skeleton directly.

Use when generating a multi-step plan for a kitchen activity (e.g.
"wash dishes", "make tea", "peel and chop vegetables"). The top hit with
similarity >= 0.6 is on-topic; 0.5-0.6 is a partial/transfer match; below 0.5
falls back to general kitchen knowledge or search_activities for raw episodes.

Returns an `error` field when the loaded memory has no consolidated procedure
templates; fall back to search_activities in that case.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal_query": {
                            "type": "string",
                            "description": "Natural-language goal, e.g. 'wash and dry dishes', 'peel a vegetable', 'stir-fry vegetables in pan'."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top-matching routines to return (default 3)."
                        }
                    },
                    "required": ["goal_query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_preferences",
                "description": """Retrieve this participant's stored preferences and habits.

Returns consolidated preference statements from Inferred Knowledge, with
supporting episode ids and a confidence score.

Use this AFTER get_routine_skill, to personalise a generic routine. Common
queries: "where knives go", "post-meal cleanup", "preferred hand for cutting",
"object storage habits". Returns most-confident preferences when query is
empty.

Returns an `error` field when no preference statements are available.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Optional focus, e.g. 'where knives go', 'post-meal cleanup'. Empty -> most confident preferences."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max number of preferences to return (default 5)."
                        }
                    },
                    "required": []
                }
            }
        }
    ]
