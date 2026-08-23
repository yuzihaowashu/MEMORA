"""Small lexical resources used by MEMORA retrieval tools."""

OBJECT_SYNONYMS = {
    # Water-related
    "tap": ["faucet", "water tap", "spout"],
    "faucet": ["tap", "water tap", "spout"],
    # Towels
    "tea towel": ["kitchen towel", "dish towel", "towel", "cloth"],
    "kitchen towel": ["tea towel", "dish towel", "towel", "cloth"],
    "dish towel": ["tea towel", "kitchen towel", "towel", "cloth"],
    # Cooking surfaces
    "stove": ["cooktop", "burner", "hob", "range"],
    "cooktop": ["stove", "burner", "hob", "range"],
    "hob": ["stove", "cooktop", "burner", "range"],
    # Surfaces
    "counter": ["countertop", "worktop", "work surface", "kitchen counter", "table"],
    "countertop": ["counter", "worktop", "work surface", "table"],
    "table": ["counter", "countertop", "worktop", "kitchen table"],
    # Appliances
    "fridge": ["refrigerator", "icebox"],
    "refrigerator": ["fridge", "icebox"],
    # Trash
    "bin": ["trash can", "garbage", "trash bin", "rubbish bin"],
    "trash can": ["bin", "garbage", "trash bin"],
    # Cutting
    "chopping board": ["cutting board", "board"],
    "cutting board": ["chopping board", "board"],
    # Utensils
    "peeler": ["vegetable peeler", "potato peeler"],
    "spatula": ["turner", "flipper", "cooking spatula"],
    "strainer": ["colander", "sieve", "drainer"],
    # Actions
    "turn on": ["switch on", "activate", "open"],
    "turn off": ["switch off", "close"],
    "pick up": ["grab", "take", "lift", "get"],
    "put down": ["place", "set down", "release", "lay down"],
    "wash": ["rinse", "clean", "scrub"],
}

LOCATION_SYNONYMS = {
    # Surfaces
    "on table": ["on counter", "on countertop", "on worktop", "on kitchen table"],
    "on counter": ["on countertop", "on table", "on worktop", "on work surface"],
    "on countertop": ["on counter", "on table", "on worktop"],
    # Dish rack variations
    "on dish rack": ["on drying rack", "in dish rack", "on rack", "in drying rack"],
    "on drying rack": ["on dish rack", "in dish rack", "on rack"],
    "in dish rack": ["on dish rack", "on drying rack", "in drying rack"],
    # Sink variations
    "in sink": ["in sink basin", "in the sink", "at sink"],
    "in sink basin": ["in sink", "in the sink"],
    # Hand/holding variations
    "in hand": ["in left hand", "in right hand", "being held", "in the person's left hand", "in the person's right hand"],
    "in left hand": ["in hand", "being held", "in the person's left hand"],
    "in right hand": ["in hand", "being held", "in the person's right hand"],
    # Appliances
    "in fridge": ["in refrigerator", "in the fridge", "in the refrigerator"],
    "in refrigerator": ["in fridge", "in the fridge"],
    # Drawers/cabinets
    "in drawer": ["in the drawer", "inside drawer"],
    "in cabinet": ["in cupboard", "in the cabinet", "inside cabinet"],
    "in cupboard": ["in cabinet", "in the cupboard"],
}

ACTION_VERBS = {
    "open", "close", "shut", "seal", "unlock", "lock",
    "pick", "grab", "take", "lift", "hold", "carry", "put", "place", "set",
    "drop", "release", "cut", "chop", "slice", "dice", "peel", "grate",
    "mince", "stir", "mix", "pour", "add", "cook", "fry", "boil", "bake",
    "roast", "steam", "heat", "warm", "microwave", "wash", "rinse", "clean",
    "scrub", "wipe", "dry", "soak", "move", "transfer", "bring", "remove",
    "dispose", "throw", "turn", "switch", "start", "stop", "fill", "empty",
    "use", "prefer", "typically", "usually", "always", "often", "habit",
}

STATE_ADJECTIVES = {
    "open", "opened", "closed", "clean", "dirty", "empty", "full", "filled",
    "hot", "cold", "warm", "wet", "dry", "on", "off",
}
