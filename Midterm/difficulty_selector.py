"""
Where "start" and "set difficulty" commands come from.

The game only consumes Commands from any number of DifficultySource objects, so
the manual (keyboard / on-screen button) selector and the future AprilTag selector
are interchangeable and can run side by side.
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Command:
    kind: str           # "difficulty", "start" or "menu"
    value: str | None = None


def difficulty(name):
    if name not in config.DIFFICULTIES:
        raise ValueError(f"unknown difficulty {name!r}; options: {list(config.DIFFICULTIES)}")
    return Command("difficulty", name)


START = Command("start")
MENU = Command("menu")


class DifficultySource(ABC):
    def start(self):
        pass

    def stop(self):
        pass

    @property
    def status(self):
        return "ready"

    @abstractmethod
    def poll(self) -> list[Command]:
        """Return (and clear) commands produced since the last poll."""


class ManualSelector(DifficultySource):
    """Fed by keyboard shortcuts and on-screen buttons (see main.py)."""

    def __init__(self):
        self._pending = []
        self._lock = threading.Lock()

    def push(self, command):
        with self._lock:
            self._pending.append(command)

    def poll(self):
        with self._lock:
            out, self._pending = self._pending, []
        return out
