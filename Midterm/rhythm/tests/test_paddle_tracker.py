"""PaddleTracker: on-screen tilt and face from synthetic IMU samples (tests/imu_sim.py)."""

import pytest

import bridge_config as bc
from imu_sim import DEG_PER_RAW, G, Paddle, feed
from paddle_tracker import PaddleTracker, axis_vector, paddle_axes

# A deliberately awkward hub mounting: gravity on -z in the neutral grip, on +y when tilted right.
NEUTRAL, RIGHT = "-z", "+y"


def tracker(**kw):
    """1:1 output (no deadzone / gain / smoothing) unless a test asks otherwise."""
    kw = {"neutral_gravity": NEUTRAL, "tilt_right_gravity": RIGHT, "gyro_deg_per_raw": DEG_PER_RAW,
          "deadzone_deg": 0.0, "gain": 1.0, "smooth_s": 0.0, **kw}
    return PaddleTracker(G, **kw)


def paddle(**kw):
    return Paddle(NEUTRAL, RIGHT, **kw)


def settled(tr, p, seconds=0.6):
    return feed(tr, p.hold(seconds))[-1]


def test_axis_specs_and_paddle_frame():
    assert axis_vector("+x") == (1.0, 0.0, 0.0) and axis_vector("-z") == (0.0, 0.0, -1.0)
    assert axis_vector("y") == (0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        axis_vector("q")
    with pytest.raises(ValueError):
        paddle_axes("+x", "-x")               # must be two different axes
    h, w, n = paddle_axes("+x", "+y")
    assert n == (0.0, 0.0, -1.0)              # N = W x H


@pytest.mark.parametrize("direction", [+1, -1])
def test_tilt_follows_left_and_right(direction):
    tr, p = tracker(), paddle()
    settled(tr, p)
    assert tr.output == (pytest.approx(0.0, abs=0.5), "black")
    feed(tr, p.move(0.5, tilt=60 * direction))
    tilt, face = settled(tr, p)
    assert tilt == pytest.approx(60 * direction, abs=1.5) and face == "black"
    feed(tr, p.move(0.5, tilt=0))
    assert settled(tr, p)[0] == pytest.approx(0.0, abs=1.5)


def test_fast_tilt_is_followed_by_the_gyro_before_the_accel_catches_up():
    tr, p = tracker(), paddle()
    settled(tr, p)
    out = feed(tr, p.move(0.1, tilt=45))           # 450 deg/s: far faster than TRACK_ACCEL_TAU_S
    assert out[-1][0] == pytest.approx(45, abs=3)


def test_a_held_tilt_stays_and_a_gyro_bias_does_not_drift_it():
    tr, p = tracker(), paddle()
    p.gyro_bias = (40.0, -30.0, 25.0)              # 4 / 3 / 2.5 deg/s of bias on every axis
    settled(tr, p)
    feed(tr, p.move(0.4, tilt=-35))
    tilt, face = settled(tr, p, seconds=10.0)
    assert tilt == pytest.approx(-35, abs=1.5) and face == "black"


def test_tilt_direction_is_the_same_on_the_yellow_face():
    tr, p = tracker(), paddle()
    settled(tr, p)
    feed(tr, p.move(0.3, flip=180))
    assert settled(tr, p) == (pytest.approx(0.0, abs=1.5), "yellow")
    feed(tr, p.move(0.5, tilt=50))
    assert settled(tr, p) == (pytest.approx(50, abs=1.5), "yellow")
    feed(tr, p.move(0.5, tilt=-50))
    assert settled(tr, p) == (pytest.approx(-50, abs=1.5), "yellow")


def test_tilt_is_continuous_through_a_flip():
    tr, p = tracker(), paddle()
    settled(tr, p)
    feed(tr, p.move(0.4, tilt=30))
    settled(tr, p)
    out = feed(tr, p.move(0.4, flip=180))          # twist while tilted
    assert all(t == pytest.approx(30, abs=4) for t, _ in out)
    assert settled(tr, p) == (pytest.approx(30, abs=1.5), "yellow")


def test_flip_and_back_never_flickers_near_edge_on():
    tr, p = tracker(), paddle()
    settled(tr, p)
    faces = []
    for target in (95, 85, 100, 80, 105, 75, 98):  # wobbling around edge-on: inside the hysteresis band
        faces += [f for _, f in feed(tr, p.move(0.1, flip=target))]
    assert set(faces) == {"black"}
    faces += [f for _, f in feed(tr, p.move(0.2, flip=180))]
    faces += [f for _, f in feed(tr, p.hold(0.5))]
    for target in (85, 95, 80, 100, 75):           # back to edge-on from the yellow side
        faces += [f for _, f in feed(tr, p.move(0.1, flip=target))]
    faces += [f for _, f in feed(tr, p.move(0.2, flip=0))]
    faces += [f for _, f in feed(tr, p.hold(0.5))]
    changes = [b for a, b in zip(faces, faces[1:]) if a != b]
    assert changes == ["yellow", "black"]


def test_a_flick_shorter_than_the_debounce_does_not_flip():
    tr, p = tracker(flip_debounce_s=0.1), paddle()
    settled(tr, p)
    out = feed(tr, [p.sample_to(flip=130)])        # one sample past the threshold...
    out += feed(tr, p.move(0.04, flip=0))          # ...and straight back
    out += feed(tr, p.hold(0.3))
    assert {f for _, f in out} == {"black"}


def test_flip_drift_settles_back_to_the_face_shown():
    tr, p = tracker(), paddle()
    settled(tr, p)
    tr.phi = 40.0                                  # as if the gyro had drifted 40 deg
    settled(tr, p, seconds=8.0)
    assert abs(tr.phi) < 2.0 and tr.output[1] == "black"


def swing(p, tr, late=3):
    """A forward swing: pitch out and back with 3 g of acceleration, the wrist wobbling the tilt
    by 25 deg on the way. The detector flags it `late` samples after it starts, as the real one does."""
    t0 = p.tilt
    seq = p.move(0.06, pitch=-50, tilt=t0 + 25, accel=(0.0, 0.0, -3.0))
    seq += p.move(0.06, pitch=20, tilt=t0 - 10, accel=(0.0, 0.0, 2.0))
    seq += p.move(0.08, pitch=0, tilt=t0, accel=(0.0, 0.0, 0.0))
    out = feed(tr, seq[:late])
    out += feed(tr, seq[late:], swinging=True)
    return out


def swing_errors(**kw):
    tr, p = tracker(**kw), paddle()
    settled(tr, p)
    feed(tr, p.move(0.4, tilt=30))
    before = settled(tr, p)
    out = swing(p, tr, late=3) + feed(tr, p.hold(0.8))
    assert {f for _, f in out} == {"black"}
    return [abs(t - before[0]) for t, _ in out]


def test_a_swing_does_not_move_the_tilt_or_the_face():
    err = swing_errors()
    # The 3 samples before the detector trips are already sent; from the trip on, the tilt is
    # rewound to before the swing and held through the follow-through.
    assert max(err[3:]) < 2.0


def test_without_rewind_and_hold_the_swing_would_leak_into_the_tilt():
    assert max(swing_errors(swing_rewind_s=0.0)[3:12]) > 10.0
    assert max(swing_errors(swing_rewind_s=0.0, swing_hold_s=0.0)[3:12]) > 10.0


def test_a_twisty_swing_does_not_flip_the_face():
    tr, p = tracker(), paddle()
    settled(tr, p)
    seq = p.move(0.1, pitch=-50, flip=140, accel=(0.0, 0.0, -3.0)) + p.move(0.1, pitch=0, flip=0, accel=(0, 0, 0))
    out = feed(tr, seq[:2]) + feed(tr, seq[2:], swinging=True) + feed(tr, p.hold(0.5))
    assert {f for _, f in out} == {"black"}


def test_haptic_pulse_freezes_the_tracker():
    tr, p = tracker(), paddle()
    settled(tr, p)
    feed(tr, p.move(0.3, tilt=20))
    before = settled(tr, p)
    out = feed(tr, p.move(0.1, tilt=60), quiet=True)
    assert all(o == before for o in out)


def test_zero_makes_the_current_grip_neutral_and_black():
    tr, p = tracker(), paddle(mount_deg=12.0)      # crooked hub: still reads 0 after the first zero
    assert settled(tr, p)[0] == pytest.approx(0.0, abs=0.5)
    feed(tr, p.move(0.3, tilt=25))
    assert settled(tr, p)[0] == pytest.approx(25, abs=1.5)
    tr.request_zero()                              # Godot at song start
    assert settled(tr, p, seconds=0.1) == (pytest.approx(0.0, abs=0.5), "black")
    feed(tr, p.move(0.3, tilt=45))                 # tilting from the new neutral
    assert settled(tr, p)[0] == pytest.approx(20, abs=1.5)


def test_zero_resets_the_face_to_black():
    tr, p = tracker(), paddle()
    settled(tr, p)
    feed(tr, p.move(0.3, flip=180))
    assert settled(tr, p)[1] == "yellow"
    feed(tr, p.move(0.3, flip=0))                  # back to black, then a zero: still black
    tr.request_zero()
    assert settled(tr, p) == (pytest.approx(0.0, abs=0.5), "black")
    tr.phi, tr.face = 180.0, "yellow"              # a zero always means "black toward the screen now"
    tr.request_zero()
    assert settled(tr, p, seconds=0.1)[1] == "black" and tr.phi == pytest.approx(0.0, abs=1.0)


def test_output_deadzone_gain_and_clamp():
    tr, p = tracker(deadzone_deg=4.0, gain=0.6, tilt_max_deg=40.0), paddle()
    settled(tr, p)
    feed(tr, p.move(0.2, tilt=3))
    assert settled(tr, p)[0] == pytest.approx(0.0, abs=0.01)
    feed(tr, p.move(0.3, tilt=-54))
    assert settled(tr, p)[0] == pytest.approx(-30.0, abs=1.0)        # (54 - 4) x 0.6
    feed(tr, p.move(0.5, tilt=90))
    assert settled(tr, p)[0] == pytest.approx(40.0, abs=0.01)


def test_default_config_is_valid():
    t = PaddleTracker(G)
    assert t.output == (0.0, "black") and not t.ready
    assert bc.TRACK_FLIP_TO_BLACK_DEG < 90 < bc.TRACK_FLIP_TO_YELLOW_DEG
    with pytest.raises(TypeError):
        PaddleTracker(G, no_such_setting=1)


def test_imu_debug_line_shows_axes_scale_and_tilt_check():
    from imu_debug import ImuDebug, nearest_axis
    assert nearest_axis((0.1, -0.9, 0.2)) == "-y"
    tr, p = tracker(), paddle()
    dbg = ImuDebug(tr)
    assert "waiting" in dbg.line()
    for s in p.hold(0.5):
        tr.update(s)
        dbg.update(s)
    assert "(" + NEUTRAL + ")" in dbg.line()
    dbg.reset()
    for s in p.move(2.0, tilt=90) + p.hold(1.0):         # step 2 of the procedure: slow 90 deg right
        tr.update(s)
        dbg.update(s)
    line = dbg.line()
    assert "(" + RIGHT + ")" in line
    scale = float(line.split("scale? ")[1].split()[0])
    assert scale == pytest.approx(DEG_PER_RAW, rel=0.05)
    check = line.split("check ")[1].split(" |")[0]       # "accel +90 gyro +90"
    acc, gyro = (float(x) for x in check.replace("accel", "").split(" gyro"))
    assert acc == pytest.approx(90, abs=3) and gyro == pytest.approx(90, abs=5)
