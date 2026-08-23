"""Prompts used to consolidate repeated experience into Inferred Knowledge."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memora.pipeline.formation_config import FormationConfig


def build_preference_prompt(config: "FormationConfig") -> str:
    """Build the participant-preference consolidation prompt."""
    return f"""You are analyzing a user's behavior across {config.experience_description} episodes to understand their organizational preferences and habits.

Participant ID: {{participant_id}}
Evidence scope: {{evidence_scope}}

## Object Registry (what objects were tracked and where)
{{object_registry}}

## Activity Log (what actions were performed)
{{activity_log}}

## Your Task
Analyze the data to identify:
1. **Storage Preferences**: Where does the user typically keep specific objects?
2. **Workflow Preferences**: Where do objects go during use vs after use?
3. **Organizational Habits**: Any patterns in how the user organizes the space?

Output JSON format:
{{{{
  "storage_preferences": [
    {{{{
      "object": "object name",
      "preferred_location": "location description",
      "context": "storage|during_use|after_use|default",
      "confidence": 0.0-1.0,
      "evidence": "brief explanation",
      "supporting_episodes": ["exact source episode IDs"]
    }}}}
  ],
  "organizational_habits": [
    {{{{
      "habit": "description of the habit",
      "objects_involved": ["list", "of", "objects"],
      "confidence": 0.0-1.0,
      "supporting_episodes": ["exact source episode IDs"]
    }}}}
  ],
  "workflow_patterns": [
    {{{{
      "pattern": "description",
      "objects": ["list", "of", "objects"],
      "confidence": 0.0-1.0,
      "supporting_episodes": ["exact source episode IDs"]
    }}}}
  ]
}}}}

Guidelines:
- Focus on CONSISTENT behaviors (appearing multiple times)
- Distinguish between "in-use" positions and "storage" positions
- Be specific about locations
- Confidence should reflect how consistent the pattern is
- Cite only episode IDs that appear in the supplied Activity Log

Return ONLY valid JSON."""


def build_procedure_prompt(config: "FormationConfig") -> str:
    """Build the reusable-procedure consolidation prompt."""
    return f"""You are analyzing a participant's activities across {config.experience_description} episodes to identify reusable procedures.

Evidence scope: {{evidence_scope}}
Total Duration: {{duration}}s
Total Activities: {{num_activities}}

Activities (grouped by episode and ordered within each episode):
{{activities}}

Your task:
1. Identify coherent procedures supported by the recorded episodes
2. Prefer procedures that recur or have multiple supporting observations
3. Preserve the participant's object and location choices
4. Do not invent steps absent from the evidence

Output JSON format:
{{{{
  "participant_id": "{{evidence_scope}}",
  "patterns": [
    {{{{
      "pattern_name": "Name of the task pattern",
      "goal": "What the person is trying to accomplish",
      "activity_type": "category of activity",
      "key_objects": ["list", "of", "objects"],
      "key_steps": ["Step 1", "Step 2"],
      "supporting_episodes": ["exact source episode IDs"],
      "supporting_activity_count": <number of supporting activity records>,
      "confidence": 0.0-1.0
    }}}}
  ]
}}}}

- Cite only episode IDs that appear in the activity lines

Return ONLY valid JSON."""
