"""Prompt used for semantic revision of Entity Memory."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memora.pipeline.formation_config import FormationConfig


def build_memory_editor_prompt(config: "FormationConfig") -> str:
    """Build the ADD/UPDATE/DELETE/NOOP prompt used by the Memory Editor."""
    obj_add_json = json.dumps(_EXAMPLE_OBJECT_ADD, indent=4, default=str)
    obj_update_json = json.dumps(_EXAMPLE_OBJECT_UPDATE, indent=4, default=str)
    obj_delete_json = json.dumps(_EXAMPLE_OBJECT_DELETE, indent=4, default=str)

    return f"""You are the MEMORA Memory Editor managing Entity Memory formed from {config.experience_description} video.

You receive:
1. **Current Object Registry** (compact): One line per tracked object -- object_id, location, state, held_by, first/last seen turn. This is a summary; full details are stored internally.
2. **New Object States** (full JSON): Objects observed in the latest video segment with complete details.

Your task: Decide how to update the object registry using ADD/UPDATE/DELETE/NOOP.

## Object Operations (by object_id):
- **ADD**: New object not in registry -> add with descriptive object_id
- **UPDATE**: Same object, changed state/location -> update only changed fields
- **DELETE**: Object no longer relevant. Use when:
  (a) Object was removed/discarded from scene
  (b) Object is a duplicate of another tracked object -- delete the duplicate
- **NOOP**: Object unchanged -> no operation

**NOTE**: The Current Registry uses compact format (one line per object). You can still UPDATE or DELETE any listed object by its object_id. For ADD, include full data from New Object States.

**IMPORTANT**: New observations may include `movement_trajectory` in `spatial_info`:
- If present: lists all locations object visited during segment
- Preserve this field in ADD/UPDATE - system will use it to track movement history
- `location` field = FINAL location, `movement_trajectory` = FULL path

## Output Format (JSON):
{{
    "object_operations": [
        {obj_add_json},
        {obj_update_json},
        {obj_delete_json},
        {{
            "event": "NOOP",
            "object_id": "some_object",
            "reason": "unchanged"
        }}
    ]
}}

## Naming Conventions:
- object_id: Use EXACT ID from VLM input (e.g., "plate_white", "cup_blue", "fork")

Return ONLY the JSON object.
/no_think"""


_EXAMPLE_OBJECT_ADD = {
    "event": "ADD",
    "object_id": "cup_blue",
    "data": {
        "name": "blue ceramic cup",
        "visual_properties": {
            "color": "blue",
            "material": "ceramic",
            "size": "medium",
        },
        "spatial_info": {
            "location": "on counter",
            "zone": "prep_area",
            "relative_to": "cutting_board",
            "movement_trajectory": ["on dish rack", "in hand", "on counter"],
        },
        "state": {
            "current_state": "empty",
            "held_by": None,
            "grip_type": None,
        },
    },
}

_EXAMPLE_OBJECT_UPDATE = {
    "event": "UPDATE",
    "object_id": "plate_white",
    "changes": {
        "spatial_info": {
            "location": "in sink",
            "zone": "work_area",
            "movement_trajectory": ["on counter", "in hand", "in sink"],
        },
        "state": {
            "current_state": "being washed",
            "held_by": "left_hand",
            "grip_type": "power",
        },
    },
}

_EXAMPLE_OBJECT_DELETE = {
    "event": "DELETE",
    "object_id": "napkin_paper",
    "reason": "discarded into trash",
}
