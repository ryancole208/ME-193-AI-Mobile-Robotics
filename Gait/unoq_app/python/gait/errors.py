"""Loud, explanatory errors.

Every error the app raises on purpose says three things: what it was doing,
what went wrong, and the most likely fix. That way a log line pasted from the
board is enough to diagnose the problem.
"""


class GaitError(Exception):
    def __init__(self, doing: str, problem: str, fix: str):
        self.doing = doing
        self.problem = problem
        self.fix = fix
        super().__init__(f"While {doing}: {problem}. Likely fix: {fix}")


class ConfigError(GaitError):
    """config.json (or setup.yaml) is missing, unreadable or inconsistent."""


class PacketError(GaitError):
    """A BLE payload does not match the shared protocol."""
