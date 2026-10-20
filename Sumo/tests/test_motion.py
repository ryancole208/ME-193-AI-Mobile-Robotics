"""Motion-direction estimator: wheels + IMU, including slip and braking."""

import config
from motion import Direction, MotionEstimator, MotionSample

DT = 0.03


def est(**kw):
    kw.setdefault("tilt_gain", 0.0)
    return MotionEstimator(log=lambda *_: None, **kw)


class Feeder:
    """Builds raw samples (config signs/units) from a true body speed and wheel speed."""

    def __init__(self, e):
        self.e, self.t, self.v_body = e, 0.0, 0.0
        self.step(0.0, 0.0, 10)        # like the armed phase: a moment at rest first

    def step(self, v_body, v_wheels, n=1, tilt=None):
        for _ in range(n):
            a = (v_body - self.v_body) / DT
            self.v_body = v_body
            raw_w = v_wheels / config.WHEEL_SPEED_TO_MPS
            acc = [0.0, 0.0, 1000.0]
            acc["xyz".index(config.IMU_FORWARD_AXIS)] += config.IMU_FORWARD_SIGN * a / 9.81 * 1000
            self.t += DT
            self.e.update(MotionSample(config.LEFT_WHEEL_SIGN * raw_w, config.RIGHT_WHEEL_SIGN * raw_w,
                                       tuple(acc), t=self.t, tilt=tilt), self.t)
        return self.e.direction


def ramp(f, v0, v1, n=5):
    for i in range(1, n + 1):
        v = v0 + (v1 - v0) * i / n
        f.step(v, v)


def test_still_is_uncertain():
    f = Feeder(est())
    assert f.step(0, 0, 10) is Direction.UNCERTAIN


def test_driving_forward_and_backward():
    f = Feeder(est())
    ramp(f, 0, 0.3)
    assert f.step(0.3, 0.3, 10) is Direction.FORWARD
    ramp(f, 0.3, -0.3, 10)
    assert f.step(-0.3, -0.3, 10) is Direction.BACKWARD


def test_braking_from_forward_never_reads_backward():
    f = Feeder(est())
    ramp(f, 0, 0.3)
    f.step(0.3, 0.3, 20)
    for i in range(1, 6):              # hard stop in 0.15 s
        d = f.step(0.3 * (1 - i / 5), 0.3 * (1 - i / 5))
        assert d is not Direction.BACKWARD
    assert f.step(0, 0, 10) is Direction.UNCERTAIN


def test_shoved_backward_while_wheels_slip_forward_reads_backward_through_long_slide():
    e = est()
    f = Feeder(e)
    ramp(f, 0, 0.2)
    f.step(0.2, 0.2, 10)
    ramp_n = 4
    for i in range(1, ramp_n + 1):                    # shove: body goes to -0.2 m/s ...
        f.step(0.2 - 0.4 * i / ramp_n, 0.2)           # ... wheels keep reading +0.2 (slip)
    assert e.slipping
    for _ in range(40):                               # 1.2 s constant-speed slide, accel ~ 0
        assert f.step(-0.2, 0.2) is Direction.BACKWARD


def test_backdriven_wheels_read_backward_immediately():
    f = Feeder(est())
    ramp(f, 0, -0.2, 3)                               # tyres grip: encoders see the real motion
    assert f.step(-0.2, -0.2, 3) is Direction.BACKWARD


def test_slip_mode_has_a_time_limit():
    e = est(slip_max_s=0.5)
    f = Feeder(e)
    ramp(f, 0, -0.3, 2)                               # IMU says backward, wheels say 0
    f.step(-0.3, 0.0, 5)
    assert e.slipping
    f.step(-0.3, 0.0, 20)                             # > 0.5 s
    assert not e.slipping or e.v > -0.3


def test_rest_bias_is_removed():
    e = est()
    for i in range(200):                              # tilted mount: forward axis reads +50 counts
        acc = [0.0, 0.0, 1000.0]
        acc["xyz".index(config.IMU_FORWARD_AXIS)] = 50.0
        e.update(MotionSample(0, 0, tuple(acc), t=i * DT), i * DT)
    assert e.direction is Direction.UNCERTAIN and abs(e.v) < 0.01


def test_tilt_compensation_cancels_gravity_leak():
    gain = 1.745                                       # counts per decidegree (1000 * sin 0.1 deg)
    e = est(tilt_gain=gain)
    axis = "xyz".index(config.IMU_FORWARD_AXIS)
    t = 0.0
    for tilt in [0] * 10 + [100] * 30:                 # nose lifted 10 degrees, robot not moving
        acc = [0.0, 0.0, 1000.0]
        acc[axis] = gain * tilt                        # gravity leaks into the forward axis
        t += DT
        e.update(MotionSample(0, 0, tuple(acc), t=t, tilt=tilt), t)
    assert abs(e.v) < 0.02


def test_missing_data_decays_to_uncertain():
    f = Feeder(est())
    ramp(f, 0, 0.3)
    f.step(0.3, 0.3, 5)
    for _ in range(40):
        f.e.update(MotionSample(None, None, None), 0)
    assert f.e.direction is Direction.UNCERTAIN


def test_repeated_sample_is_not_integrated_twice():
    e = est()
    s = MotionSample(0, 0, (0.0, 0.0, 1000.0), t=1.0)
    e.update(s, 1.0)
    acc = [0.0, 0.0, 1000.0]
    acc["xyz".index(config.IMU_FORWARD_AXIS)] = 300.0
    s2 = MotionSample(0, 0, tuple(acc), t=1.03)
    e.update(s2, 1.03)
    v1 = e.v
    e.update(s2, 1.05)                                 # same notification, next tick
    assert e.v == v1
