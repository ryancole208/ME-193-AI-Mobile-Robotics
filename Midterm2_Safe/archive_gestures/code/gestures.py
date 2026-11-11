"""Gesture classes: the single source of truth for names and order.

The model's output index i means CLASSES[i]. Every script, the webpage and the
exported model_config.json take the list from here, so changing the order here
(and retraining) is the only way it can change.
"""

CLASSES = [
    "pinch_forward",
    "pinch_left",
    "pinch_right",
    "v_forward",
    "v_left",
    "v_right",
    "no_hand",
    "other",
]

# Classes that can be part of a code. no_hand / other exist only so in-between
# frames have somewhere to go; they never register as a gesture.
CODE_GESTURES = CLASSES[:6]

DISPLAY_NAMES = {
    "pinch_forward": "Pinch forward",
    "pinch_left": "Pinch left",
    "pinch_right": "Pinch right",
    "v_forward": "V forward",
    "v_left": "V left",
    "v_right": "V right",
    "no_hand": "No hand",
    "other": "Other",
}

# Shown on the collection page when a class is selected. Directions are from the
# user's point of view ("forward" = away from your body).
HOW_TO = {
    "pinch_forward": "Index + middle together, hand flat, pointing away from you.",
    "pinch_left": "Index + middle together, hand flat, pointing to YOUR left.",
    "pinch_right": "Index + middle together, hand flat, pointing to YOUR right.",
    "v_forward": "Index + middle spread in a V, pointing away from you. Vary the spread.",
    "v_left": "V-spread, pointing to YOUR left. Vary the spread.",
    "v_right": "V-spread, pointing to YOUR right. Vary the spread.",
    "no_hand": "No hand: empty pad, pad with random objects, shadows, a sleeve/arm at the edge.",
    "other": "Fist, open palm, 1 or 3 fingers, slowly switching gestures, hand half out of frame.",
}

assert set(DISPLAY_NAMES) == set(CLASSES) == set(HOW_TO)
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}
