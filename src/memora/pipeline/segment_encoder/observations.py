"""Normalize typed observations emitted by the Segment Encoder."""

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def fix_object_registry_format(obj_registry: Any) -> Dict[str, Any]:
    """Normalize a single-object response into an object registry."""
    if not obj_registry or not isinstance(obj_registry, dict):
        return {}

    if "object_id" in obj_registry and "name" in obj_registry:
        obj_id = obj_registry.pop("object_id", "unknown_object")
        if not isinstance(obj_id, str):
            obj_id = str(obj_id)
        obj_id = obj_id.replace(" ", "_").lower()
        logger.debug("Fixed single-object registry entry %r", obj_id)
        return {obj_id: obj_registry}

    is_registry = all(
        isinstance(value, dict)
        and ("name" in value or "visual_properties" in value or "state" in value)
        for value in obj_registry.values()
    )
    if not is_registry:
        logger.warning("Unknown object registry format: %s", list(obj_registry)[:5])
    return obj_registry


_NON_HOLDABLE_KEYWORDS = frozenset({
    "stovetop", "stove", "sink", "fridge", "refrigerator", "countertop",
    "counter", "oven", "dishwasher", "washing_machine", "microwave",
    "cabinet", "shelf", "wall", "floor", "ceiling", "window", "door",
    "table", "rack", "drying_rack",
})
_NUMBERED_SUFFIX_RE = re.compile(r"^(.+?)_(\d+)$")
_MAX_NUMBERED_VARIANTS = 4


def sanitize_object_registry(obj_registry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize object IDs, fixed-scene attributes, and numbered duplicates."""
    if not obj_registry or not isinstance(obj_registry, dict):
        return {}

    cleaned: Dict[str, Any] = {}
    for obj_id, obj_data in obj_registry.items():
        normalized_id = obj_id.strip().lower().replace(" ", "_").replace("-", "_")
        if not isinstance(obj_data, dict):
            cleaned[normalized_id] = obj_data
            continue

        if any(keyword in normalized_id for keyword in _NON_HOLDABLE_KEYWORDS):
            state = obj_data.get("state")
            if isinstance(state, dict):
                state["held_by"] = None
                state["grip_type"] = None
            spatial = obj_data.get("spatial_info")
            if isinstance(spatial, dict):
                spatial["movement_trajectory"] = None
        cleaned[normalized_id] = obj_data

    return _merge_numbered_duplicates(cleaned)


def _merge_numbered_duplicates(registry: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse implausibly numerous numbered variants into one base ID."""
    groups: Dict[str, list] = {}
    unnumbered: Dict[str, Any] = {}
    for obj_id, obj_data in registry.items():
        match = _NUMBERED_SUFFIX_RE.match(obj_id)
        if match:
            groups.setdefault(match.group(1), []).append(
                (int(match.group(2)), obj_id, obj_data)
            )
        else:
            unnumbered[obj_id] = obj_data

    result = dict(unnumbered)
    for base, variants in groups.items():
        if len(variants) <= _MAX_NUMBERED_VARIANTS:
            for _, obj_id, obj_data in variants:
                result[obj_id] = obj_data
            continue
        logger.warning("Merging %d numbered variants of %r", len(variants), base)
        variants.sort(key=lambda item: item[0])
        result.setdefault(base, variants[0][2])
    return result


def _normalize_zone_value(zone: Any) -> Dict[str, Any]:
    if isinstance(zone, dict):
        return zone
    if isinstance(zone, list):
        return {"contents": zone}
    if isinstance(zone, str):
        return {"contents": [zone]}
    return {}


def fix_environment_format(env_data: Any) -> Optional[Dict[str, Any]]:
    """Validate and normalize an Environment Memory observation."""
    if env_data is None:
        return {}
    if not isinstance(env_data, dict):
        logger.warning("Environment observation is not an object; retrying segment")
        return None

    zones = env_data.get("zones")
    if isinstance(zones, dict):
        env_data["zones"] = {
            zone_name: _normalize_zone_value(zone_value)
            for zone_name, zone_value in zones.items()
        }
    elif isinstance(zones, list):
        env_data["zones"] = {
            f"zone_{index}": _normalize_zone_value(zone)
            for index, zone in enumerate(zones)
        }
    elif isinstance(zones, str):
        env_data["zones"] = {"main": {"description": zones}}
    elif zones is not None:
        env_data["zones"] = {}

    features = env_data.get("features")
    if features is not None and not isinstance(features, list):
        if isinstance(features, str):
            env_data["features"] = [features]
        elif isinstance(features, dict):
            env_data["features"] = list(features.values())
        else:
            env_data["features"] = []

    spatial = env_data.get("spatial_relations")
    if spatial is not None and not isinstance(spatial, list):
        env_data["spatial_relations"] = [spatial] if isinstance(spatial, (str, dict)) else []
    return env_data
