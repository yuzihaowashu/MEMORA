"""Prompts used by the Segment Encoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memora.pipeline.formation_config import FormationConfig

_EXAMPLE_ACTIVITIES = (
    "Person picks up plate with left hand from counter",
    "Person washes plate with sponge under running water",
    "Person places plate on drying rack",
)
_EXAMPLE_STATES = (
    "Plate is on drying rack (right of sink)",
    "Fork is in sink",
    "Cabinet is closed",
)
_EXAMPLE_ENVIRONMENT = (
    "Kitchen has stainless steel sink",
    "Counter is wooden",
    "Blue and yellow tile backsplash",
)


class SegmentEncoderPrompts:
    """Build typed and Flat-1D observation prompts."""

    def __init__(self, config: "FormationConfig"):
        self.cfg = config

    def _timestamp_buckets(self) -> str:
        """Generate timestamp bucket labels like '0-2s/2-4s/.../8-10s'."""
        seg = self.cfg.segment_length_int
        step = max(1, seg // 5)
        buckets = []
        t = 0
        while t < seg:
            end = min(t + step, seg)
            buckets.append(f"{t}-{end}s")
            t = end
        return "/".join(buckets)

    def build_flat_observation_prompt(self) -> str:
        """Build the Flat-1D observation prompt."""
        cfg = self.cfg
        act_lines = "\n".join(f'        "{item}",' for item in _EXAMPLE_ACTIVITIES)
        st_lines = "\n".join(f'        "{item}",' for item in _EXAMPLE_STATES)
        env_lines = "\n".join(f'        "{item}",' for item in _EXAMPLE_ENVIRONMENT)

        return f"""Extract facts directly from this {cfg.experience_description} video segment.

{{segment_instruction}}

You MUST output EXACTLY this structure with ALL sections filled:

```json
{{{{
    "activities": [
        // 8-10 facts about WHAT THE PERSON DID
        // Format: "Person [verb] [object] with [hand]"
{act_lines}
        // ... at least 8 activities
    ],
    "states": [
        // 5-7 facts about WHERE OBJECTS ARE at the END
        // For answering "Where is X?" questions
        // Format: "[Object] is [location/condition]"
{st_lines}
        // ... at least 5 states
    ],
    "environment": [
        // 3-4 facts about PERMANENT SCENE FEATURES
        // Things that never change
{env_lines}
        // ... at least 3 environment facts
    ],
    "summary": "One sentence summary of the segment"
}}}}
```

CRITICAL: Output valid JSON with ALL three arrays filled. Do not skip any section."""

    def build_typed_observation_prompt(self) -> str:
        """Build the Environment, Entity, and Activity observation prompt."""
        cfg = self.cfg
        seg = cfg.segment_length_int
        ts_buckets = self._timestamp_buckets()

        return f"""You are analyzing a {seg}-second {cfg.experience_description} video segment for a comprehensive memory system.

Segment {{turn_id}}, Video Time: [{{start_time}}-{{end_time}}]s
**NOTE**: This segment is {seg} seconds long. In action_breakdown, use RELATIVE timestamps (0-{seg}s), NOT absolute video time.
Example: If video time is [30-{30 + seg}]s and an action happens at video second {30 + seg // 2}, report it as "{seg // 2}s" or "{seg // 2 - 1}-{seg // 2 + 1}s".

{{previous_context}}

{{segment_instruction}}

**OBJECT REGISTRY GUIDELINES** (STRICT):
- Identify entities directly from the current video segment.
- Include entities involved in the observed activity or needed to ground its spatial context.
- Do not invent entities that are not visible or supported by the video.
- Reuse identifiers from the prior context when the same entity persists across segments.
- Focus on objects that the person INTERACTS with (touches, picks up, moves, uses)

Extract three typed observations from this segment. These observations are
later written to MEMORA's four memory stores; Inferred Knowledge is produced
by offline consolidation rather than directly from one segment.

## ENVIRONMENT OBSERVATION (Spatially-Aware)
**WARNING: DESCRIBE WHAT YOU ACTUALLY SEE** - Do NOT copy examples! Each video has a UNIQUE environment.

```json
{{{{
    "layout_description": "[DESCRIBE THE ACTUAL SCENE: What room? What's the main focus? What's visible on left/right/behind? What materials, colors, lighting do you see?]",

    "zones": {{{{
        "[zone_name]": {{{{
            "anchor": "[main object in this zone]",
            "position": "[where in the view: center/left/right/top/bottom]",
            "contents": ["[list actual objects you see in this zone]"],
            "description": "[what this area is used for]"
        }}}}
    }}}},

    "spatial_relations": [
        "[object1] [RELATION] [object2]"
    ],

    "features": ["[list distinctive visual features you actually see]"],
    "lighting": "[describe actual lighting conditions]",
    "ambient": "[describe sounds, steam, motion blur, etc.]"
}}}}
```
**Relations**: LEFT_OF, RIGHT_OF, ABOVE, BELOW, BEHIND, IN_FRONT_OF, ADJACENT_TO, INSIDE, ON_TOP_OF

**IMPORTANT**:
- Describe what YOU SEE in THIS video, not a generic scene
- If environment changes between segments (e.g., person moves to different room), UPDATE the description
- Focus on **what's relevant to the actions** happening in this segment

## ACTIVITY OBSERVATION (Rich Motion Details)
**IMPORTANT**: Timestamps are RELATIVE to segment start (always 0-{seg}s), NOT absolute video time!

**WARNING: DESCRIBE THE ACTUAL ACTIONS** - What is the person DOING in this specific {seg}-second segment?
**NO COPY-PASTE**: Your summary MUST be unique to THIS segment. If the action continues from a previous segment, describe the PROGRESS (e.g., "continues transferring, now most items moved" or "finishes placing last items and wipes tray").

```json
{{{{
    "summary": "[One sentence: WHO does WHAT with WHICH objects]",
    "detailed_narrative": "[4-6 sentences describing the temporal flow of actions in this segment]",
    "action_breakdown": [
        {{{{
            "timestamp": "[{ts_buckets} - RELATIVE to segment start]",
            "action": "[verb: picks up/places/opens/closes/etc.]",
            "object": "[object name supported by the video]",
            "hand": "[left/right/both]",
            "manner": "[quickly/slowly/carefully/firmly/gently]",
            "direction": "[from X to Y / upward / leftward / etc.]"
        }}}}
    ],
    "concurrent_actions": ["[any actions happening simultaneously]"] or null
}}}}
```
**If NO significant action happens** (e.g., person is just standing), set `"summary": "No significant activity"` and leave action_breakdown empty.

## ENTITY OBSERVATIONS (`object_registry` dictionary)
**CRITICAL FORMAT RULES:**
- Output as a DICTIONARY where each object_id is a KEY (not a field!)
- **MINIMUM 3 objects per segment** — a scene always has surfaces, appliances, utensils, and items visible. Look carefully!
- Include ALL visible objects (typically 5-15 per segment), NOT just one!
- **OBJECT ID FORMAT**: lowercase with underscores, e.g., "plate_white", "cutting_board_wooden", "faucet_chrome" (NO spaces, NO hyphens)
- **REUSE IDs**: If an object appeared in a previous segment (see [Known Objects] in VIDEO CONTEXT above), use the EXACT SAME object_id. Do NOT create new IDs for the same object
- **held_by**: Only set for objects a person CAN physically hold. Large/fixed objects (stovetop, sink, countertop, fridge, shelf) MUST have held_by=null

**MOVEMENT TRACKING** (`movement_trajectory` field):
- If object MOVED during this {seg}s segment, list ALL locations: `["start_loc", "mid_loc", "end_loc"]`
- If object did NOT move, set to `null`
- Example: Object picked from shelf, held, then placed on table:
  ```
  "movement_trajectory": ["on shelf", "in hand", "on table"]
  ```

**CORRECT FORMAT** (object_id as KEY):
```json
{{{{
    "[object_type]_[color]": {{{{
        "name": "[descriptive name]",
        "visual_properties": {{{{
            "color": "[actual color you see]",
            "material": "[ceramic/metal/plastic/wood/glass/etc.]",
            "size": "[small/medium/large]",
            "shape": "[round/rectangular/cylindrical/etc.]",
            "condition": "[clean/dirty/wet/dry/etc.]",
            "quantity": 1
        }}}},
        "spatial_info": {{{{
            "location": "[where: on table/in hand/on shelf/etc.]",
            "zone": "[which environment zone]",
            "orientation": "[flat/vertical/tilted/etc.]",
            "relative_to": "[nearest reference object]",
            "movement_trajectory": ["[start_loc]", "[mid_loc]", "[end_loc]"] or null
        }}}},
        "state": {{{{
            "current_state": "[what's happening: idle/being used/held/etc.]",
            "held_by": "[left_hand/right_hand/null]",
            "grip_type": "[power/pinch/fingertip/null]"
        }}}}
    }}}}
}}}}
```
**movement_trajectory examples**:
- Object moved: `["on shelf", "in hand", "on table"]`
- Object stationary: `null`

**WRONG FORMAT** (DO NOT output like this - object_id should NOT be a field):
```json
{{{{
    "object_id": "plate_white",
    "name": "white ceramic plate",
    ...
}}}}
```

Include ALL visible objects: hands, utensils, appliances, items, containers, etc.

OUTPUT FORMAT (three online observation types):
```json
{{{{
    "environment": {{{{...}}}},
    "activity_narrative": {{{{...}}}},
    "object_registry": {{{{...}}}}
}}}}
```
Note: Do NOT output segment_info - it is automatically added by the system.

CRITICAL RULES:
1. Include ALL visible objects with full visual properties
2. Use consistent object_ids across segments
3. Capture motion manner (speed, force, direction)
4. Estimate quantities when multiple items exist
5. All timestamps in action_breakdown MUST be within 0-{seg}s (the segment duration)

**JSON FORMAT RULES** (VERY IMPORTANT):
- Output ONLY valid JSON, NO markdown code blocks (no ```json)
- Use DOUBLE QUOTES for all strings (not single quotes)
- NO trailing commas before closing braces/brackets
- Ensure ALL strings are properly closed
- If unsure about a field value, use null instead of incomplete text
"""
