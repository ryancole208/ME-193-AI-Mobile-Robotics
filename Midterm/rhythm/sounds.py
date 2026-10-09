"""
Short sound effects, synthesised in code (no sound files): metronome / preview clicks, the two
warning cues, the tempo-change tick and a soft hit "tock". These are effects, not music -- every
song in the game is a real music file. Every function returns float32 stereo at rate sr.
"""

import numpy as np

_RNG = np.random.default_rng(193)


def _t(sr, dur):
    return np.arange(int(sr * dur)) / sr


def _stereo(x, peak=0.9):
    m = np.max(np.abs(x)) or 1.0
    x = (x / m * peak).astype(np.float32)
    return np.ascontiguousarray(np.stack([x, x], axis=1))


def _fade(x, sr, ms=3):
    n = min(len(x), int(sr * ms / 1000))
    if n:
        x[:n] *= np.linspace(0, 1, n)
        x[-n:] *= np.linspace(1, 0, n)
    return x


def click(sr, accent=False):
    """Metronome click: a bright sine ping plus a tiny noise transient. Accent = higher and louder."""
    t = _t(sr, 0.06)
    f = 2200.0 if accent else 1500.0
    x = np.sin(2 * np.pi * f * t) * np.exp(-t / (0.012 if accent else 0.009))
    x += 0.3 * _RNG.standard_normal(len(t)) * np.exp(-t / 0.002)
    return _stereo(_fade(x, sr, 1), 0.95 if accent else 0.6)


def warning_cue(sr):
    """Downbeat warning: a quick two-note marimba-like "ba-dum" (original)."""
    out = np.zeros(int(sr * 0.22))
    for start, freq in ((0.0, 784.0), (0.075, 1046.5)):
        t = _t(sr, 0.14)
        tone = (np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * freq * 4.0 * t) * np.exp(-t / 0.01))
        tone *= np.exp(-t / 0.045)
        i = int(start * sr)
        out[i:i + len(t)] += tone[:len(out) - i]
    return _stereo(_fade(out, sr), 0.85)


def upbeat_cue(sr):
    """Upbeat warning: a rising "bwee-OOP" glide with a sparkle on top -- clearly different."""
    t = _t(sr, 0.20)
    freq = 420.0 + (1260.0 - 420.0) * (t / t[-1]) ** 1.6
    phase = 2 * np.pi * np.cumsum(freq) / sr
    x = (np.sin(phase) + 0.4 * np.sign(np.sin(phase)) * 0.3) * np.minimum(1, t / 0.02) * np.exp(-np.maximum(0, t - 0.12) / 0.03)
    s = _t(sr, 0.08)
    sparkle = np.sin(2 * np.pi * 2637.0 * s) * np.exp(-s / 0.02)
    i = int(0.12 * sr)
    x[i:i + len(s)] += 0.5 * sparkle[:len(x) - i]
    return _stereo(_fade(x, sr), 0.8)


def tempo_tick(sr):
    """Tempo-change tick: a metronome "tick" -- a woodblock-like knock (two partials with a short
    ring) plus a sharp attack, long and loud enough to cut through the music."""
    t = _t(sr, 0.09)
    x = np.sin(2 * np.pi * 1250.0 * t) * np.exp(-t / 0.022)
    x += 0.55 * np.sin(2 * np.pi * 2600.0 * t) * np.exp(-t / 0.012)
    x += 0.35 * _RNG.standard_normal(len(t)) * np.exp(-t / 0.002)
    return _stereo(_fade(x, sr, 1), 1.0)


def hit_tock(sr):
    """Soft wooden "tock" for a successful return."""
    t = _t(sr, 0.08)
    x = np.sin(2 * np.pi * 380.0 * t) * np.exp(-t / 0.018)
    x += 0.4 * np.sin(2 * np.pi * 1180.0 * t) * np.exp(-t / 0.006)
    x += 0.15 * _RNG.standard_normal(len(t)) * np.exp(-t / 0.004)
    return _stereo(_fade(x, sr, 1), 0.7)


class SoundBank:
    """All effects rendered once at the engine's sample rate."""

    def __init__(self, sr):
        self.sr = sr
        self.click = click(sr)
        self.click_accent = click(sr, accent=True)
        self.warning = warning_cue(sr)
        self.upbeat = upbeat_cue(sr)
        self.tempo_tick = tempo_tick(sr)
        self.hit = hit_tock(sr)
