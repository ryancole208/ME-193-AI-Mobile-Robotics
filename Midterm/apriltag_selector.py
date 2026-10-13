"""
STUB: AprilTag-based start / difficulty selector (for later, once tags are printed).

Plan: show a tag to the camera -> a Command is emitted. The tag -> command map is
config.APRILTAG_COMMANDS (default: tag 0 = start, 1/2/3 = Easy/Medium/Hard). Plug it
into main.py's list of DifficultySources; nothing in game.py needs to change.

What is implemented now: the tag-id -> Command mapping with debouncing (unit
tested). What is left: `_detect_tag_ids(frame)` -- e.g. cv2.aruco with
DICT_APRILTAG_36h11 (opencv-contrib is already in my_env; see ../AprilTagTracking),
and a frame source (either its own camera or frames shared from PoseTracker).
"""

import time

import config
from difficulty_selector import Command, DifficultySource


def tags_to_commands(tag_ids, mapping=None):
    mapping = config.APRILTAG_COMMANDS if mapping is None else mapping
    cmds = []
    for tid in tag_ids:
        if tid in mapping:
            kind, value = mapping[tid]
            cmds.append(Command(kind, value))
    return cmds


class AprilTagSelector(DifficultySource):
    def __init__(self, mapping=None, hold_s=config.APRILTAG_HOLD_S, clock=time.monotonic):
        self.mapping = config.APRILTAG_COMMANDS if mapping is None else mapping
        self.hold_s = hold_s
        self.clock = clock
        self._first_seen = {}   # tag id -> time it appeared
        self._fired = set()     # tags already acted on while still visible
        self._pending = []

    @property
    def status(self):
        return "stub (not implemented)"

    def feed_tag_ids(self, tag_ids):
        """Feed the ids detected in one frame. A tag fires once after being visible
        for hold_s; it can fire again only after it leaves the view."""
        now = self.clock()
        visible = set(tag_ids)
        for tid in list(self._first_seen):
            if tid not in visible:
                del self._first_seen[tid]
                self._fired.discard(tid)
        for tid in visible:
            self._first_seen.setdefault(tid, now)
            if tid not in self._fired and now - self._first_seen[tid] >= self.hold_s:
                self._fired.add(tid)
                self._pending.extend(tags_to_commands([tid], self.mapping))

    def feed_frame(self, frame_bgr):
        self.feed_tag_ids(self._detect_tag_ids(frame_bgr))

    def _detect_tag_ids(self, frame_bgr):
        raise NotImplementedError("AprilTag detection not implemented yet (stub)")

    def poll(self):
        out, self._pending = self._pending, []
        return out
