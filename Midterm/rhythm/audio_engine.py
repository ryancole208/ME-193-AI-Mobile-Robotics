"""
Audio output + the song clock (the game's master clock).

SoundDeviceEngine (default) runs our own mixer inside a PortAudio callback:
  * a global frame counter counts every sample ever written;
  * each callback reports when its first sample reaches the speaker
    (time_info.outputBufferDacTime, on the same clock as time.perf_counter());
  * so "which frame is being heard at wall time t" is known to the sample, smoothed with
    a small EMA and re-synced after underruns;
  * a song is placed at a global frame G0, so song_time(t) = (heard_frame(t) - G0) / rate.
    Negative song time = silent pre-roll before the file starts.
Cue sounds, clicks and hit sounds are Voices mixed in at exact global frames, so they stay
locked to the music no matter what the game loop does.

PygameEngine is the fallback (pygame.mixer, less precise), NullEngine is clock-only.
All engines share the same interface (see EngineBase).
"""

import threading
import time

import numpy as np

import config

clock = time.perf_counter   # == time.monotonic() on Windows / Python 3.13+ (both QueryPerformanceCounter)


# =============================================================================
# Loading
# =============================================================================

def to_stereo(data):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return np.ascontiguousarray(data[:, :2])


def resample(data, sr_in, sr_out):
    if sr_in == sr_out or len(data) == 0:
        return data
    try:
        import soxr
        return np.ascontiguousarray(soxr.resample(data, sr_in, sr_out).astype(np.float32))
    except ImportError:
        n_out = int(round(len(data) * sr_out / sr_in))
        x_old = np.arange(len(data)) / sr_in
        x_new = np.arange(n_out) / sr_out
        return np.stack([np.interp(x_new, x_old, data[:, c]) for c in range(data.shape[1])],
                        axis=1).astype(np.float32)


def load_audio(path, sr):
    """Decode a WAV / OGG / MP3 file to float32 stereo at sample rate sr."""
    import soundfile as sf
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(to_stereo(data), file_sr, sr)


# =============================================================================
# Engines
# =============================================================================

class Voice:
    __slots__ = ("data", "start", "gain", "tag")

    def __init__(self, data, start, gain=1.0, tag=""):
        self.data, self.start, self.gain, self.tag = data, int(start), float(gain), tag


class EngineBase:
    """Interface shared by all engines. Times are wall-clock seconds (clock()), song seconds,
    or global frames."""

    name = "base"
    sr = config.AUDIO_FALLBACK_SAMPLE_RATE

    def __init__(self):
        self.clock = clock           # wall clock (injectable for tests)
        self.song_start = None       # global frame of song sample 0
        self.song_length_s = 0.0
        self.status = "idle"

    # -- clock --------------------------------------------------------------------
    def heard_frame(self, t):
        raise NotImplementedError

    def song_time(self, t=None):
        """Song position being HEARD at wall time t (default: now). None if no song is loaded."""
        if self.song_start is None:
            return None
        t = self.clock() if t is None else t
        return (self.heard_frame(t) - self.song_start) / self.sr

    def wall_time(self, song_t):
        """Inverse of song_time: wall-clock time a song position is heard."""
        now = self.clock()
        return now + (song_t - self.song_time(now))

    # -- playback -------------------------------------------------------------------
    def play_song(self, data, preroll_s=0.0, length_s=None, start_s=0.0):
        """Start a song (stereo float32 array, or None for a silent click track) preroll_s
        seconds from now, from song position start_s (seeking). Returns nothing; use song_time()."""
        raise NotImplementedError

    def schedule(self, data, song_t, gain=1.0, tag="cue"):
        """Play a sound so that it is heard at song time song_t."""
        raise NotImplementedError

    def play_now(self, data, gain=1.0, tag="fx"):
        raise NotImplementedError

    def cancel(self, tag):
        """Drop every scheduled sound with this tag that hasn't finished yet."""
        raise NotImplementedError

    def stop_song(self):
        raise NotImplementedError

    def update(self):
        """Called every frame (only the pygame fallback needs it)."""

    def close(self):
        pass


class SoundDeviceEngine(EngineBase):
    name = "sounddevice"

    def __init__(self, hostapi=None, device=None, latency=None, blocksize=None):
        super().__init__()
        import sounddevice as sd
        self.sd = sd
        hostapi = config.AUDIO_HOSTAPI if hostapi is None else hostapi
        device = config.AUDIO_DEVICE if device is None else device
        if device is None and hostapi:
            for i, api in enumerate(sd.query_hostapis()):
                if hostapi.lower() in api["name"].lower():
                    device = api["default_output_device"]
                    break
        info = sd.query_devices(device, "output")
        self.device_name = info["name"]
        self.sr = int(info.get("default_samplerate") or config.AUDIO_FALLBACK_SAMPLE_RATE)
        self._lock = threading.Lock()
        self._voices = []
        self._frame = 0               # next global frame to be written
        self._offset = None           # smoothed (dac time - frame / sr), seconds
        self.underruns = 0
        self._extra = config.AUDIO_EXTRA_OUTPUT_LATENCY_MS / 1000.0
        self.stream = sd.OutputStream(
            device=device, samplerate=self.sr, channels=2, dtype="float32",
            latency=config.AUDIO_LATENCY if latency is None else latency,
            blocksize=config.AUDIO_BLOCKSIZE if blocksize is None else blocksize,
            callback=self._callback)
        self.stream.start()
        self.latency_ms = self.stream.latency * 1000
        api = sd.query_hostapis(info["hostapi"])["name"]
        self.status = f"{api}: {self.device_name} ({self.latency_ms:.0f} ms)"

    def _callback(self, out, frames, time_info, status):
        if status.output_underflow:
            self.underruns += 1
        g = self._frame
        out.fill(0.0)
        with self._lock:
            voices = self._voices
            keep = []
            for v in voices:
                s = v.start - g
                n = len(v.data)
                if s + n <= 0:
                    continue                      # finished
                keep.append(v)
                if s >= frames:
                    continue                      # not yet
                o0, d0 = max(0, s), max(0, -s)
                cnt = min(frames - o0, n - d0)
                out[o0:o0 + cnt] += v.data[d0:d0 + cnt] * v.gain
            self._voices = keep
        np.clip(out, -1.0, 1.0, out=out)
        dac = time_info.outputBufferDacTime or (time_info.currentTime + self.stream.latency)
        off = dac - g / self.sr
        if self._offset is None or abs(off - self._offset) > config.AUDIO_CLOCK_RESYNC_MS / 1000:
            self._offset = off
        else:
            self._offset += config.AUDIO_CLOCK_SMOOTHING * (off - self._offset)
        self._frame = g + frames

    def heard_frame(self, t):
        off = self._offset
        if off is None:                          # no callback yet: estimate
            return self._frame - (self.stream.latency - 0.0) * self.sr
        return (t - off - self._extra) * self.sr

    def _write_frame_now(self):
        """A global frame safely in the future (not yet written)."""
        return self._frame + int(0.03 * self.sr)

    def play_song(self, data, preroll_s=0.0, length_s=None, start_s=0.0):
        self.stop_song()
        g0 = int(max(self._write_frame_now(), self.heard_frame(self.clock()) + preroll_s * self.sr))
        self.song_length_s = length_s if length_s is not None else (len(data) / self.sr if data is not None else 0.0)
        skip = int(round(max(0.0, start_s) * self.sr))
        self.song_start = g0 - skip
        if data is not None and skip < len(data):
            with self._lock:
                self._voices.append(Voice(data[skip:], g0, config.MUSIC_VOLUME, "music"))

    def schedule(self, data, song_t, gain=1.0, tag="cue"):
        if self.song_start is None or data is None:
            return
        start = self.song_start + int(round(song_t * self.sr))
        if start < self._frame:                  # too late to play in time: skip rather than play late
            return
        with self._lock:
            self._voices.append(Voice(data, start, gain, tag))

    def play_now(self, data, gain=1.0, tag="fx"):
        if data is None:
            return
        with self._lock:
            self._voices.append(Voice(data, self._frame, gain, tag))

    def cancel(self, tag):
        with self._lock:
            self._voices = [v for v in self._voices if v.tag != tag]

    def stop_song(self):
        with self._lock:
            self._voices = [v for v in self._voices if v.tag == "fx"]
        self.song_start = None

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


class NullEngine(EngineBase):
    """No sound; the song clock is just wall time. clock_fn can be faked in tests."""

    name = "null"

    def __init__(self, sr=None, clock_fn=None):
        super().__init__()
        self.sr = sr or config.AUDIO_FALLBACK_SAMPLE_RATE
        self.clock = clock_fn or clock
        self.scheduled = []           # (song_t, tag) -- for tests
        self.played = []
        self.status = "no audio (clock only)"

    def heard_frame(self, t):
        return t * self.sr

    def play_song(self, data, preroll_s=0.0, length_s=None, start_s=0.0):
        self.song_start = (self.clock() + preroll_s - start_s) * self.sr
        self.song_length_s = length_s if length_s is not None else (len(data) / self.sr if data is not None else 0.0)
        self.scheduled = []

    def schedule(self, data, song_t, gain=1.0, tag="cue"):
        self.scheduled.append((song_t, tag))

    def play_now(self, data, gain=1.0, tag="fx"):
        self.played.append(tag)

    def cancel(self, tag):
        self.scheduled = [s for s in self.scheduled if s[1] != tag]

    def stop_song(self):
        self.song_start = None


class PygameEngine(EngineBase):
    """Fallback on pygame.mixer: clock = time since play() + an assumed output latency."""

    name = "pygame"

    def __init__(self):
        super().__init__()
        import pygame
        self.pg = pygame
        self.sr = 44100
        pygame.mixer.init(frequency=self.sr, size=-16, channels=2, buffer=512)
        pygame.mixer.set_num_channels(32)
        self._t0 = None
        self._pending = []            # (wall time, sound, gain)
        self._music = None
        self._latency = config.AUDIO_PYGAME_LATENCY_MS / 1000
        self.status = f"pygame mixer (assumed {config.AUDIO_PYGAME_LATENCY_MS:.0f} ms)"

    def _sound(self, data, gain):
        arr = (np.clip(data * gain, -1, 1) * 32767).astype(np.int16)
        return self.pg.sndarray.make_sound(np.ascontiguousarray(arr))

    def heard_frame(self, t):
        return (t - self._latency) * self.sr

    def play_song(self, data, preroll_s=0.0, length_s=None, start_s=0.0):
        self.stop_song()
        start = clock() + preroll_s           # play() is called then; it is heard _latency later
        skip = int(round(max(0.0, start_s) * self.sr))
        self.song_start = start * self.sr - skip
        self.song_length_s = length_s if length_s is not None else (len(data) / self.sr if data is not None else 0.0)
        if data is not None and skip < len(data):
            self._pending.append((start, self._sound(data[skip:], config.MUSIC_VOLUME), "music"))

    def schedule(self, data, song_t, gain=1.0, tag="cue"):
        if self.song_start is None or data is None:
            return
        when = self.song_start / self.sr + song_t
        self._pending.append((when, self._sound(data, gain), tag))

    def play_now(self, data, gain=1.0, tag="fx"):
        if data is not None:
            self._sound(data, gain).play()

    def cancel(self, tag):
        self._pending = [p for p in self._pending if p[2] != tag]

    def update(self):
        now = clock()
        due = [p for p in self._pending if p[0] <= now]
        self._pending = [p for p in self._pending if p[0] > now]
        for _, snd, tag in due:
            ch = snd.play()
            if tag == "music":
                self._music = ch

    def stop_song(self):
        self._pending = []
        if self._music is not None:
            self._music.stop()
            self._music = None
        self.song_start = None

    def close(self):
        self.pg.mixer.quit()


def make_engine(backend=None):
    """The configured engine, falling back to pygame and then to no sound."""
    backend = config.AUDIO_BACKEND if backend is None else backend
    order = {"sounddevice": ["sounddevice", "pygame", "null"], "pygame": ["pygame", "null"]}.get(backend, ["null"])
    for name in order:
        try:
            if name == "sounddevice":
                try:
                    return SoundDeviceEngine()
                except Exception as e:
                    if config.AUDIO_HOSTAPI is None:
                        raise
                    print(f"[audio] {config.AUDIO_HOSTAPI} failed ({e}); trying the default device")
                    return SoundDeviceEngine(hostapi="")
            if name == "pygame":
                return PygameEngine()
            return NullEngine()
        except Exception as e:
            print(f"[audio] {name} backend unavailable: {e}")
    return NullEngine()
