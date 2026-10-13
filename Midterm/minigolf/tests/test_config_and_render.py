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


def test_this_is_the_minigolf_config():
    assert config.HERE.name == "minigolf"
    assert list(config.DIFFICULTIES) == ["Easy", "Hard"]
    assert config.DIFFICULTIES["Easy"]["cup_radius_m"] > config.DIFFICULTIES["Hard"]["cup_radius_m"]
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
    assert config.PUTT_SPEED_MIN < config.PUTT_SPEED_AT_THRESHOLD < config.PUTT_SPEED_MAX
    assert config.PUTT_MODE in ("accel", "gyro", "either", "both")
    assert len(config.HOLE_PLANS) == len(config.HOLE_OBSTACLES) == config.HOLES_PER_GAME
    assert config.HOLE_OBSTACLES[0] == config.HOLE_OBSTACLES[1] == ()
    for d in config.DIFFICULTIES.values():
        assert config.GOLF_BALL_RADIUS_M < d["cup_radius_m"] < config.CELL_M / 4
    assert 0 < config.AIM_POSE_DEADZONE < 1


def test_renderer_draws_every_state():
    pygame = pytest.importorskip("pygame")
    np = pytest.importorskip("numpy")
    from golf_game import GolfGame
    from golf_render import Renderer
    from golfer_input import GolferInput
    from pose_tracker import PoseSnapshot
    from putt_detector import PuttEvent

    pygame.init()
    try:
        screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        r = Renderer(screen)
        g = GolfGame(GolferInput(), rng=random.Random(0))
        st = [("Motor", "motor", "scanning"), ("MQTT", "mqtt", "connected"), ("Pose", "pose", "disabled")]
        snap = PoseSnapshot(preview_rgb=np.zeros((180, config.PREVIEW_WIDTH, 3), dtype=np.uint8))

        r.draw(0.0, g, st, snap, ["debug"], "hint")
        assert {"start", "difficulty:Easy", "difficulty:Hard"} <= set(r.buttons)
        g.set_difficulty("Hard")
        g.start(0.0)
        r.draw(0.5, g, st, snap, None, "")
        assert r.buttons == {}
        t = 0.5
        for hole in range(5):
            g.hole_index = hole
            g._begin_hole(t)
            r.draw(t + 0.1, g, st, snap, None, "")             # intro flyover
            g.start(t + 0.2)                                     # skip it
            g.golfer.begin_charge(t + 0.3)
            r.draw(t + 1.0, g, st, None, ["x"], "", overview=hole == 4, live_strength=2.0)
            g.golfer.charge_t = None
            t += config.AIM_ARM_DELAY_S + 1.0
            assert g.add_putt(PuttEvent(t, t, 2.5, "keyboard"), t)
            for _ in range(20):
                t += 1 / 15
                g.update(t)
                r.draw(t, g, st, snap, None, "")
            g.strokes[-1] = 2
        g.state = "game_over"
        g.best = 10
        r.draw(t, g, st, None, None, "")
        assert {"start", "menu"} <= set(r.buttons)
    finally:
        pygame.quit()


def test_camera_projection_and_near_clipping():
    from golf_render import Camera
    cam = Camera(1000)
    cam.set(0.0, 0.0, 1.0, 0.0, 0.0)
    sx, sy, s = cam.project(0.0, 5.0, 1.0)                    # straight ahead at eye height
    assert sx == pytest.approx(500) and sy == pytest.approx(config.VIEW_CENTER_Y_PX)
    assert cam.project(0.0, -1.0, 1.0) is None                # behind
    assert cam.project(1.0, 5.0, 1.0)[0] > 500                # east is right when facing north
    cam.set(0.0, 0.0, 1.0, 90.0, 0.0)                         # facing east
    assert cam.project(5.0, 0.0, 1.0)[0] == pytest.approx(500)
    quad = [(-1, -1, 0), (1, -1, 0), (1, 3, 0), (-1, 3, 0)]
    cam.set(0.0, 0.0, 1.0, 0.0, 20.0)
    poly = cam.polygon(quad)
    assert poly is not None and len(poly) >= 3
