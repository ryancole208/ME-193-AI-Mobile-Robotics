"""Sanity checks on config.py values."""

import config


def test_score_topic_uses_name_segment():
    from mqtt_client import score_topic
    assert score_topic() == "ME193/RQ-D2/Name"
    assert score_topic(name="Alice") == "ME193/RQ-D2/Alice"


def test_difficulties_get_faster():
    assert list(config.DIFFICULTIES) == ["Easy", "Medium", "Hard", "Dynamic"]
    speeds = [config.DIFFICULTIES[n]["ball_speed"] for n in ("Easy", "Medium", "Hard")]
    assert speeds == sorted(speeds) and all(s > 0 for s in speeds)
    assert config.DEFAULT_DIFFICULTY in config.DIFFICULTIES
    assert config.DYNAMIC_DIFFICULTY in config.DIFFICULTIES


def test_dynamic_rl_settings_valid():
    r = config.REWARD
    lo, hi = r["target_band"]
    assert 0 < lo < hi < 1
    assert r["base_return"] >= 0 and r["hard_return_bonus"] > 0 and r["miss_penalty"] > 0
    assert r["base_return"] < r["hard_return_bonus"]           # easy shots only earn a little
    assert r["difficulty_exponent"] >= 1 and r["rally_weight"] >= 0
    assert set(r["difficulty_weights"]) == {"speed", "distance", "spin"}
    assert 0 < r["adjust_rate"] < 0.2 and 0 < r["relax_rate"] < 0.2
    assert r["weight_limits"][0] <= 1 <= r["weight_limits"][1]
    assert r["window"] >= r["min_samples"] > 0
    assert 0 <= config.RL_GAMMA < 1 and 0 < config.RL_ALPHA <= 1 and config.RL_BACKOFF_K >= 0
    assert 0 < config.RL_EPSILON_MIN <= config.RL_EPSILON_START <= 1 and 0 < config.RL_EPSILON_DECAY < 1
    assert config.RL_HITS_PER_STEP >= 1 and config.RL_HIT_COUNT_MODE in ("session", "rally")
    assert config.RL_UNLOCK_TOPBACK_EPSILON <= config.RL_UNLOCK_FAST_EPSILON
    assert set(config.RL_PLACEMENT_X_M) == {"left", "center", "right"}
    assert set(config.RL_SPEEDS_M_S) == {"slow", "medium", "fast"}
    half = config.TABLE_WIDTH_M / 2
    assert all(abs(x) < half for x in config.RL_PLACEMENT_X_M.values())
    assert 0 < config.RL_SPIN_AMOUNT <= config.SPIN_MAX


def test_ble_notification_period_in_library_range():
    assert 15 <= config.BLE_NOTIFICATION_MS <= 1000


def test_swing_settings_valid():
    assert config.SWING_MODE in ("accel", "gyro", "either", "both")
    assert config.SWING_COOLDOWN_S > config.SWING_PEAK_WINDOW_S > 0
    assert 0 < config.IMU_GRAVITY_ALPHA < 1


def test_pose_settings_valid():
    assert config.DOMINANT_HAND in ("left", "right")
    lo, hi = config.POSE_X_RANGE
    assert 0 <= lo < hi <= 1
    assert 0 <= config.POSE_SMOOTHING < 1
    assert config.POSE_DIRECTION_FRAMES >= 2


def test_game_geometry_valid():
    assert config.PLAYER_HIT_Y_M < 0 < config.TABLE_LENGTH_M < config.OPPONENT_HIT_Y_M
    assert config.RETURN_SPEED_MIN_MULT <= 1 <= config.RETURN_SPEED_MAX_MULT
    assert config.RETURN_ANGLE_LEFT_RIGHT_DEG <= config.RETURN_ANGLE_MAX_DEG
    assert config.HAPTIC_MOTOR in ("left", "right", "both")


def test_apriltag_map_references_real_difficulties():
    for kind, value in config.APRILTAG_COMMANDS.values():
        assert kind in ("start", "difficulty", "menu")
        if kind == "difficulty":
            assert value in config.DIFFICULTIES


def test_spin_and_roll_settings_valid():
    assert config.IMU_ROLL_AXIS in ("x", "y", "z")
    assert config.IMU_ROLL_SIGN in (1.0, -1.0) and config.IMU_ACCEL_ROLL_SIGN in (1.0, -1.0)
    assert config.IMU_GYRO_DEG_PER_RAW > 0
    assert 0 < config.ROLL_FILTER_ALPHA < 1
    assert config.SPIN_INPUT in ("rate", "angle", "both")
    assert config.SPIN_TOP_SOURCE in ("imu", "off")
    assert 0 <= config.SPIN_RATE_DEADZONE_DPS < config.SPIN_RATE_FULL_DPS
    assert 0 <= config.SPIN_ANGLE_DEADZONE_DEG < config.SPIN_ANGLE_FULL_DEG
    assert 0 <= config.SPIN_TOP_DEADZONE_G < config.SPIN_TOP_FULL_G
    assert config.SPIN_MAX > 0 and config.MAGNUS_STRENGTH >= 0
    assert set(config.OPPONENT_SPIN_PENALTY) == set(config.DIFFICULTIES)
    assert config.HAPTIC_SETTLE_S >= 0


def test_theme_and_look_settings_valid():
    from theme import THEMES
    assert config.THEME in THEMES
    assert "comic" not in (config.FONT_TITLE + config.FONT_BODY).lower()
    for key in ("PLAYER_PADDLE_FRONT", "PLAYER_PADDLE_BACK", "OPPONENT_PADDLE_FRONT", "TABLE_TOP_COLOR",
                "BALL_COLOR", "BALL_OUTLINE_COLOR", "PADDLE_RIM_COLOR"):
        assert len(getattr(config, key)) == 3
    assert set(config.TRAIL_COLORS) == {"none", "top", "back", "side"}
    assert 0 < config.VIEW_FOV_DEG < 120 and -10 < config.VIEW_CAMERA_PITCH_DEG < 80
