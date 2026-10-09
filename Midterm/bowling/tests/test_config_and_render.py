"""Config inheritance from ../config.py, and a headless renderer smoke test."""

import importlib.util
import os
import random

import pytest

import config

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def load_midterm_config():
    spec = importlib.util.spec_from_file_location("midterm_config_check", config.MIDTERM / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_this_is_the_bowling_config():
    assert config.HERE.name == "bowling"
    assert list(config.DIFFICULTIES) == ["Bumpers", "No Bumpers"]
    assert config.DIFFICULTIES["Bumpers"]["bumpers"] and not config.DIFFICULTIES["No Bumpers"]["bumpers"]
    assert config.DEFAULT_DIFFICULTY in config.DIFFICULTIES


def test_hardware_settings_inherited_from_ping_pong():
    mid = load_midterm_config()
    for name in ("BLE_SCAN_TIMEOUT_S", "IMU_CALIBRATION_SAMPLES", "POSE_X_RANGE", "CAMERA_INDEX",
                 "MQTT_TOPIC_BASE", "PLAYER_NAME", "HAPTIC_PULSE_MS", "POSE_MODEL_PATH"):
        assert getattr(config, name) == getattr(mid, name)
    assert list(mid.DIFFICULTIES) == ["Easy", "Medium", "Hard"]   # ../config.py untouched


def test_reused_modules_come_from_midterm():
    import motor_imu
    import pose_tracker
    assert os.path.dirname(motor_imu.__file__) == str(config.MIDTERM)
    assert os.path.dirname(pose_tracker.__file__) == str(config.MIDTERM)


def test_sane_values():
    assert config.BALL_SPEED_MIN < config.BALL_SPEED_AT_THRESHOLD < config.BALL_SPEED_MAX
    assert config.PIN_WOBBLE_SPEED < config.PIN_TOPPLE_SPEED
    assert config.SPIN_GYRO_AXIS in ("x", "y", "z", None)
    assert config.THROW_MODE in ("accel", "gyro", "either", "both")
    assert 0 < config.OIL_LENGTH_M < config.HEAD_PIN_Y_M < config.PIT_Y_M


def test_renderer_draws_every_state():
    pygame = pytest.importorskip("pygame")
    np = pytest.importorskip("numpy")
    from bowler_input import BowlerInput
    from bowling_game import BowlingGame
    from bowling_render import Renderer
    from pose_tracker import PoseSnapshot
    from throw_detector import ThrowEvent

    pygame.init()
    try:
        screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        r = Renderer(screen)
        g = BowlingGame(BowlerInput(), rng=random.Random(0))
        st = [("Motor", "motor", "scanning"), ("MQTT", "mqtt", "connected"), ("Pose", "pose", "disabled")]
        snap = PoseSnapshot(preview_rgb=np.zeros((180, config.PREVIEW_WIDTH, 3), dtype=np.uint8))

        r.draw(0.0, g, st, snap, ["debug"], "hint")
        assert {"start", "difficulty:Bumpers", "difficulty:No Bumpers"} <= set(r.buttons)
        g.set_difficulty("Bumpers")
        g.start(0.0)
        r.draw(0.5, g, st, snap, None, "")
        assert r.buttons == {}
        t = 1.0
        g.add_throw(ThrowEvent(t, t, 2.0, 0.4, "keyboard"), t)
        while g.state != "aiming" or t < 2:
            t += 1 / 30
            g.update(t)
            r.draw(t, g, st, snap, None, "")
            if g.state == "game_over":
                break
        g.state = "game_over"
        r.draw(t, g, st, None, None, "")
        assert {"start", "menu"} <= set(r.buttons)
    finally:
        pygame.quit()
