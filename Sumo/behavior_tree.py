"""
A small, readable behavior tree (no external library).

Every node has tick(bb) -> Status, where `bb` is the shared "blackboard" object
holding the robot's current inputs and outputs. Nodes:

  Condition(name, fn)   fn(bb) -> bool          => SUCCESS / FAILURE (never RUNNING)
  Action(name, fn)      fn(bb) -> Status        => whatever fn returns
  Sequence(name, kids)  "do these in order": fails as soon as one child fails.
                        It has MEMORY: if a child returned RUNNING, the next tick
                        resumes at that child instead of re-checking earlier ones.
                        That is what lets "drive backward for 1 s" finish even
                        after the blue tape is no longer under the sensor.
  Selector(name, kids)  "first one that works": tries children left to right and
                        stops at the first that does not fail. It is REACTIVE: it
                        restarts from the left every tick, so a higher-priority
                        branch can always interrupt a lower one; the interrupted
                        branch is halt()-ed so its memory/timers reset.

Design rule for actions: an action that just sends one command per tick returns
SUCCESS (so its condition is re-checked next tick). Only actions that take time
and must not be interrupted by their own condition (the timed reverse, the
lost/song wait) return RUNNING.
"""

from enum import Enum


class Status(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class Node:
    def __init__(self, name):
        self.name = name

    def tick(self, bb):
        raise NotImplementedError

    def halt(self):
        """Called when a parent abandons this node mid-run; reset any memory."""


class Condition(Node):
    def __init__(self, name, fn):
        super().__init__(name)
        self.fn = fn

    def tick(self, bb):
        return Status.SUCCESS if self.fn(bb) else Status.FAILURE


class Action(Node):
    """Wraps a function returning a Status. For stateful actions, subclass and
    override tick()/halt() instead (see ReverseAction in robot_tree.py)."""

    def __init__(self, name, fn=None):
        super().__init__(name)
        self.fn = fn

    def tick(self, bb):
        result = self.fn(bb)
        if not isinstance(result, Status):
            raise TypeError(f"action {self.name!r} must return a Status, got {result!r}")
        return result


class Sequence(Node):
    def __init__(self, name, children):
        super().__init__(name)
        self.children = list(children)
        self._current = 0          # index of the child to (re)start at

    def tick(self, bb):
        while self._current < len(self.children):
            status = self.children[self._current].tick(bb)
            if status is Status.RUNNING:
                return Status.RUNNING          # remember where we are
            if status is Status.FAILURE:
                self._current = 0
                return Status.FAILURE
            self._current += 1                 # SUCCESS: go on to the next child
        self._current = 0
        return Status.SUCCESS

    def halt(self):
        if self._current < len(self.children):
            self.children[self._current].halt()
        self._current = 0


class Selector(Node):
    def __init__(self, name, children):
        super().__init__(name)
        self.children = list(children)
        self.active_index = None   # child that did not fail on the last tick
        self._running_index = None # child that returned RUNNING on the last tick

    def tick(self, bb):
        for i, child in enumerate(self.children):
            status = child.tick(bb)
            if status is Status.FAILURE:
                continue
            # A different (higher-priority) child won: halt the one that was running.
            if self._running_index is not None and self._running_index != i:
                self.children[self._running_index].halt()
            self._running_index = i if status is Status.RUNNING else None
            self.active_index = i
            return status
        if self._running_index is not None:
            self.children[self._running_index].halt()
            self._running_index = None
        self.active_index = None
        return Status.FAILURE

    def halt(self):
        if self._running_index is not None:
            self.children[self._running_index].halt()
        self._running_index = None
        self.active_index = None

    @property
    def active_child(self):
        return None if self.active_index is None else self.children[self.active_index]


def render(node, indent=0):
    """Text drawing of a tree, printed at startup so the console shows the structure."""
    kind = {Selector: "?", Sequence: "->"}.get(type(node), "")
    if isinstance(node, Condition):
        kind = "(cond)"
    elif isinstance(node, Action) or not isinstance(node, (Selector, Sequence)):
        kind = kind or "[act]"
    lines = ["  " * indent + f"{kind} {node.name}"]
    for child in getattr(node, "children", []):
        lines.append(render(child, indent + 1))
    return "\n".join(lines)
