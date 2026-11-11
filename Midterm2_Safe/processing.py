"""Frame-processing hook.

Runs once per captured frame inside the camera thread. Later stages add the
finger-counting CNN here and return its results in `info` (e.g. count,
confidence scores) plus any overlay drawn onto the frame. Whatever goes in
`info` is forwarded unchanged to the page via /status.
"""


def process_frame(frame):
    """Return (frame, info_dict). Pass-through for now."""
    return frame, {}
