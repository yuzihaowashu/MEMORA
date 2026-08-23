"""Normalize and merge Entity Memory records for participant-level retrieval."""

import copy
from typing import Any, Dict


LOCATION_WORDS = frozenset(
    {
        "on", "in", "at", "near", "next", "beside", "under", "above",
        "counter", "countertop", "sink", "stove", "table", "shelf",
        "drawer", "cabinet", "rack", "fridge", "refrigerator", "oven",
        "microwave", "dishwasher", "floor", "wall", "left", "right",
    }
)

KNOWN_COLORS = frozenset(
    {
        "red", "blue", "green", "yellow", "orange", "purple", "pink",
        "black", "white", "gray", "grey", "brown", "beige", "tan",
        "silver", "gold", "chrome", "copper", "clear", "transparent",
        "dark", "light",
    }
)

KNOWN_MATERIALS = frozenset(
    {
        "plastic", "metal", "ceramic", "glass", "wood", "wooden",
        "steel", "stainless", "aluminum", "rubber", "silicone",
        "cotton", "fabric", "paper", "cardboard", "porcelain",
    }
)


def normalize_object_name(name: str) -> str:
    """Normalize an object name for duplicate grouping."""
    normalized = name.lower().strip()
    if normalized.endswith("s") and not normalized.endswith("ss"):
        normalized = normalized[:-1]
    return " ".join(normalized.split())


def strip_attributes_from_name(
    name: str,
    color: str = "",
    material: str = "",
) -> str:
    """Remove color and material adjectives while retaining the base noun."""
    if not name:
        return name
    noise = set(KNOWN_COLORS | KNOWN_MATERIALS)
    for attribute in (color, material):
        noise.update(word.strip() for word in attribute.lower().split() if word.strip())
    cleaned = [token for token in name.split() if token.lower().strip() not in noise]
    result = " ".join(cleaned).strip()
    return result if result else name


def is_valid_attribute(value: str) -> bool:
    """Reject location phrases accidentally written into color or material fields."""
    if not value:
        return False
    return not bool(set(value.lower().split()) & LOCATION_WORDS)


def deduplicate_objects(current_memory: Dict[str, Any]) -> None:
    """Merge duplicate Entity Memory records across a participant's videos."""
    object_registry = current_memory.get("object_registry", {})
    if not object_registry:
        return

    groups: Dict[tuple, list] = {}
    for object_id, object_data in object_registry.items():
        if not isinstance(object_data, dict):
            continue
        name = object_data.get("name", object_id)
        visual = object_data.get("visual_properties") or {}
        if not isinstance(visual, dict):
            visual = {}
        raw_color = visual.get("color") or ""
        raw_material = visual.get("material") or ""
        if isinstance(raw_color, list):
            raw_color = raw_color[0] if raw_color else ""
        if isinstance(raw_material, list):
            raw_material = raw_material[0] if raw_material else ""
        key = (
            normalize_object_name(name),
            str(raw_color).lower().strip(),
            str(raw_material).lower().strip(),
        )
        groups.setdefault(key, []).append((object_id, object_data))

    merged: Dict[str, dict] = {}
    for entries in groups.values():
        if len(entries) == 1:
            object_id, object_data = entries[0]
            merged[object_id] = object_data
            continue

        def completeness(item) -> int:
            _, data = item
            visual = data.get("visual_properties") or {}
            spatial = data.get("spatial_info") or {}
            if not isinstance(visual, dict):
                visual = {}
            if not isinstance(spatial, dict):
                spatial = {}
            return sum(bool(value) for value in visual.values()) + sum(
                bool(value) for value in spatial.values()
            )

        entries_sorted = sorted(
            entries,
            key=lambda item: (-completeness(item), item[0]),
        )
        best_id, best = entries_sorted[0]
        canonical = copy.deepcopy(best)

        locations = []
        seen_locations = set()
        source_videos = set()
        for _, data in entries:
            spatial = data.get("spatial_info") or {}
            if not isinstance(spatial, dict):
                spatial = {}
            location = spatial.get("location") or data.get("last_location", "")
            if location and location not in seen_locations:
                locations.append(location)
                seen_locations.add(location)
            source_video = data.get("source_video", "")
            if source_video:
                source_videos.add(source_video)

        if not isinstance(canonical.get("spatial_info"), dict):
            canonical["spatial_info"] = {}
        if locations:
            canonical["spatial_info"]["location"] = locations[0]
            if len(locations) > 1:
                canonical["spatial_info"]["all_locations"] = locations
        canonical["source_videos"] = sorted(source_videos)
        canonical["_dedup_count"] = len(entries)

        clean_id = best_id
        for source_video in sorted(source_videos):
            if clean_id.startswith(f"{source_video}_"):
                clean_id = clean_id[len(source_video) + 1:]
                break
        merged[clean_id] = canonical

    for object_data in merged.values():
        if not isinstance(object_data, dict):
            continue
        visual = object_data.get("visual_properties") or {}
        if isinstance(visual, dict):
            for field in ("color", "material"):
                value = visual.get(field, "")
                if value and not is_valid_attribute(str(value)):
                    visual[field] = ""

    current_memory["object_registry"] = merged
