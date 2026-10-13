"""Audio engine: the sample clock and sample-accurate mixing (callback driven by a fake stream)."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

import config
from audio_engine import NullEngine, SoundDeviceEngine, Voice, load_audio, resample, to_stereo

SR = 48000
BLOCK = 480


def fake_engine():
    """A SoundDeviceEngine without a real device: we call its callback by hand.
    Its wall clock reads the DAC time of the last block (fake time)."""
    e = SoundDeviceEngine.__new__(SoundDeviceEngine)
    SoundDeviceEngine.__bases__[0].__init__(e)
    e.clock = lambda: e._last_dac
    e._last_dac = 0.0
    e.sr = SR
    import threading
    e._lock = threading.Lock()
    e._voices, e._frame, e._offset, e.underruns, e._extra = [], 0, None, 0, 0.0
    e.stream = SimpleNamespace(latency=0.02)
    return e


def run_block(e, dac):
    out = np.zeros((BLOCK, 2), dtype=np.float32)
    e._last_dac = dac
    e._callback(out, BLOCK, SimpleNamespace(outputBufferDacTime=dac, currentTime=dac - 0.02),
                SimpleNamespace(output_underflow=False))
    return out


def test_clock_maps_wall_time_to_heard_frames():
    e = fake_engine()
    t0 = 100.0
    for k in range(50):
        run_block(e, t0 + k * BLOCK / SR + 0.0004 * ((-1) ** k))   # tiny jitter
    # frame 0 is heard at ~t0; after smoothing the mapping is linear and sample-accurate-ish
    assert e.heard_frame(t0 + 0.5) == pytest.approx(0.5 * SR, abs=SR * 0.001)


def test_song_time_and_scheduled_sounds_land_on_exact_frames():
    e = fake_engine()
    run_block(e, 10.0)
    e.play_song(np.zeros((SR, 2), np.float32), preroll_s=0.0)
    g0 = e.song_start
    click = np.ones((10, 2), np.float32)
    e.schedule(click, 0.25, gain=1.0)                  # heard 0.25 s after song start
    target = g0 + int(round(0.25 * SR))
    outs = [run_block(e, 10.0 + (k + 1) * BLOCK / SR) for k in range(40)]
    audio = np.concatenate(outs)
    first = int(np.argmax(np.abs(audio[:, 0]) > 0.5))
    assert first + BLOCK == target                       # block 0 was written before the outs
    # song_time at the moment that frame is heard is 0.25 s
    t_heard = e._offset + target / SR
    assert e.song_time(t_heard) == pytest.approx(0.25, abs=1e-6)


def test_late_schedules_are_dropped_not_played_late():
    e = fake_engine()
    run_block(e, 1.0)
    e.play_song(None, preroll_s=0.0, length_s=5)
    for k in range(20):
        run_block(e, 1.0 + (k + 1) * BLOCK / SR)
    n = len(e._voices)
    e.schedule(np.ones((5, 2), np.float32), 0.0)       # already written: skip
    assert len(e._voices) == n


def test_voice_mixing_is_clipped_and_finished_voices_are_removed():
    e = fake_engine()
    e._voices = [Voice(np.full((100, 2), 0.8, np.float32), 0, 1.0, "fx"),
                 Voice(np.full((100, 2), 0.8, np.float32), 50, 1.0, "fx")]
    out = run_block(e, 1.0)
    assert out[60, 0] == pytest.approx(1.0) and out[10, 0] == pytest.approx(0.8)
    run_block(e, 1.01)
    assert e._voices == []


def test_stop_song_keeps_effects():
    e = fake_engine()
    e._voices = [Voice(np.ones((10, 2)), 5000, 1, "music"), Voice(np.ones((10, 2)), 5000, 1, "fx")]
    e.stop_song()
    assert [v.tag for v in e._voices] == ["fx"] and e.song_start is None


def test_play_from_a_position_seeks_and_cancel_drops_a_tag():
    e = fake_engine()
    run_block(e, 10.0)
    music = np.arange(SR * 4, dtype=np.float32).repeat(2).reshape(-1, 2) / (SR * 4)
    e.play_song(music, preroll_s=0.0, start_s=2.0)
    voice = [v for v in e._voices if v.tag == "music"][0]
    assert len(voice.data) == SR * 2 and voice.data[0, 0] == pytest.approx(0.5)   # starts 2 s in
    assert e.song_start == voice.start - 2 * SR                                    # song time stays true
    e.schedule(np.ones((5, 2), np.float32), 2.5, tag="preview")
    e.schedule(np.ones((5, 2), np.float32), 2.6, tag="cue")
    e.cancel("preview")
    assert sorted(v.tag for v in e._voices) == ["cue", "music"]


def test_null_engine_clock():
    now = [50.0]
    e = NullEngine(clock_fn=lambda: now[0])
    assert e.song_time() is None
    e.play_song(None, preroll_s=1.0, length_s=10)
    assert e.song_time() == pytest.approx(-1.0)
    now[0] += 3.5
    assert e.song_time() == pytest.approx(2.5)
    assert e.wall_time(4.0) == pytest.approx(now[0] + 1.5)


def test_loading_and_resampling(tmp_path):
    import wave
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(22050)
        w.writeframes((np.sin(np.arange(22050) / 10) * 10000).astype("<i2").tobytes())
    data = load_audio(p, 44100)
    assert data.shape == (44100, 2) and data.dtype == np.float32
    assert to_stereo(np.zeros(5)).shape == (5, 2)
    assert resample(np.zeros((100, 2), np.float32), 100, 200).shape == (200, 2)


@pytest.mark.hardware
def test_real_device_clock_tracks_wall_time():
    from audio_engine import SoundDeviceEngine as Real, clock
    e = Real()
    try:
        e.play_song(None, preroll_s=0.2, length_s=5)
        time.sleep(0.5)
        a = e.song_time()
        t = clock()
        time.sleep(1.0)
        b = e.song_time()
        assert b - a == pytest.approx(clock() - t, abs=0.005)
        assert e.underruns == 0
    finally:
        e.close()
