"""Finger-count classes: the single source of truth for names, order and numbers.

The model's output index i means CLASSES[i], which shows CLASS_TO_COUNT[CLASSES[i]]
fingers. Every script, the webpage and the exported model_config.json take the
list from here, so changing the order here (and retraining) is the only way it
can change. (The old pinch/V direction classes live in archive_gestures/.)
"""

CLASSES = ["none", "one", "two", "three", "four", "five"]

CLASS_TO_COUNT = {"none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
COUNT_TO_CLASS = {n: c for c, n in CLASS_TO_COUNT.items()}

# Counts that can be a code digit. `none` (0) never registers.
CODE_COUNTS = [1, 2, 3, 4, 5]

DISPLAY_NAMES = {
    "none": "0 — none",
    "one": "1 finger",
    "two": "2 fingers",
    "three": "3 fingers",
    "four": "4 fingers",
    "five": "5 fingers",
}

_SPREAD = "Spread fingers apart. Between bursts change position, height, rotation, which fingers, and left/right hand."

# Shown on the collection page when a class is selected.
HOW_TO = {
    "none": "Empty pad, a fist, a hand only half in frame, an arm passing through.",
    "one": "Any 1 finger (index, thumb, pinky...). " + _SPREAD,
    "two": "Any 2 fingers (index+middle, thumb+index...). " + _SPREAD,
    "three": "Any 3 fingers. " + _SPREAD,
    "four": "Any 4 fingers (thumb folded, or pinky folded...). " + _SPREAD,
    "five": "Open hand, all 5 fingers. " + _SPREAD,
}

assert set(DISPLAY_NAMES) == set(CLASSES) == set(HOW_TO) == set(CLASS_TO_COUNT)
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}
