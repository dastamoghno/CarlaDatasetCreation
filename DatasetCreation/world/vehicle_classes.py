"""Shared vehicle class tokens and type_id → class mapping for fleet + labeling."""

MOTORCYCLE_BP_TOKENS = ("motorcycle", "vespa", "yamaha", "kawasaki", "harley")
BICYCLE_BP_TOKENS = ("bicycle", "crossbike", "gazelle", "diamondback", "bh.")
TRUCK_BP_TOKENS = ("firetruck", "ambulance", "truck", "carlacola")

# Kinematic sidewalk bicycles set this role_name at spawn (see bicycle_sidewalk.py).
SIDEWALK_BICYCLE_ROLE = "dataset_bicycle_sidewalk"


def vehicle_class_from_type_id(type_id: str) -> str:
    """Map CARLA vehicle type_id to dataset vehicle class label."""
    type_lower = type_id.lower()
    if any(token in type_lower for token in TRUCK_BP_TOKENS):
        return "truck"
    if "bus" in type_lower:
        return "bus"
    if any(token in type_lower for token in MOTORCYCLE_BP_TOKENS):
        return "motorcycle"
    if any(token in type_lower for token in BICYCLE_BP_TOKENS) or "bike" in type_lower:
        return "bicycle"
    if "van" in type_lower:
        return "van"
    return "car"


def bp_is_truck(bp) -> bool:
    return any(tok in bp.id.lower() for tok in TRUCK_BP_TOKENS)


def bp_is_motorcycle(bp) -> bool:
    return any(tok in bp.id.lower() for tok in MOTORCYCLE_BP_TOKENS)


def bp_is_bicycle(bp) -> bool:
    tid = bp.id.lower()
    return any(tok in tid for tok in BICYCLE_BP_TOKENS)
