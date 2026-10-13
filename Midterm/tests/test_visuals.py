"""Geometry behind the visuals (no window needed): my paddle's mirrored faces, the skeleton's
grip / forehand-backhand / aim / spin tilt, and non-overlapping decorations."""

import math
import random

import pytest

import config
from camera3d import Camera
from game import GameLogic
from opponent import SimulatedOpponent
from paddle import player_paddle
from player_input import PlayerInput
from skeleton import STANCES, Skeleton, paddle_basis
from spin import Spin
from theme import gameplay_hulls, ghost_rect, layout_decorations, pumpkin_rect, rect_hits_hull, rects_overlap

CAM = Camera(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
MY_CENTER = (0.0, config.PLAYER_HIT_Y_M, config.PADDLE_HEIGHT_M)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# =============================================================================
# My paddle: mirrored faces
# =============================================================================

def test_mirrored_paddle_shows_yellow_in_the_neutral_grip():
    p = player_paddle(invert=True)
    assert not p.front_visible(CAM, MY_CENTER, 0.0)           # the camera sees the back ...
    assert p.back == config.PLAYER_PADDLE_FRONT                # ... which is the yellow face
    assert p.front == config.PLAYER_PADDLE_BACK                # black faces the screen
    q = player_paddle(invert=False)                            # old mapping: black toward me
    assert not q.front_visible(CAM, MY_CENTER, 0.0) and q.back == config.PLAYER_PADDLE_BACK


def test_default_is_mirrored():
    assert config.PADDLE_FACE_INVERT is True
    assert player_paddle().back == config.PLAYER_PADDLE_FRONT


def test_mirroring_keeps_the_twist_direction():
    """Only the colours swap; the geometry (and so the on-screen twist) is identical."""
    a, b = player_paddle(invert=True), player_paddle(invert=False)
    for roll in (-60, -20, 0, 20, 60):
        assert a.basis(roll) == b.basis(roll)
    # + roll: the screen-side face (normal +y) turns toward +x, i.e. clockwise seen from above
    _, _, n = a.basis(30.0)
    assert n[0] > 0 and n[1] > 0
    assert a.front_visible(CAM, MY_CENTER, 0.0) == b.front_visible(CAM, MY_CENTER, 0.0)


def test_twisting_past_edge_on_shows_the_other_face():
    p = player_paddle(invert=True)
    assert not p.front_visible(CAM, MY_CENTER, 30.0)            # still yellow
    assert p.front_visible(CAM, MY_CENTER, 150.0)               # turned round: black


# =============================================================================
# Skeleton: grip, forehand/backhand, aim, spin
# =============================================================================

def _visible_normal(center, basis):
    n = basis[2]
    return n if CAM.facing(center, n) else tuple(-c for c in n)


def _game():
    g = GameLogic(SimulatedOpponent(rng=random.Random(0)), PlayerInput(), rng=random.Random(0))
    g.start(0.0)
    g.update(config.SERVE_DELAY_S)
    return g


def _frames(sk, g, events, t0=5.0, n=60, dt=0.02):
    for kind in events:
        g.event_times = {kind: t0}
        for k in range(n):
            now = t0 - 0.3 + k * dt
            sk.update(now, g)
            p = sk.pose(now, g)
            yield p, sk.arm(p)


@pytest.mark.parametrize("stance", ["forehand", "backhand"])
def test_paddle_stays_locked_in_the_hand_every_frame(stance):
    sk, g = Skeleton(), _game()
    sk.stance = stance
    sk.blend = 1.0 if stance == "backhand" else 0.0
    k = sk.paddle.scale
    along = config.PADDLE_BLADE_H_M * k * 0.42 + config.PADDLE_HANDLE_L_M * k * config.SKELETON_GRIP_ALONG_HANDLE
    count = 0
    for p, arm in _frames(sk, g, ["opponent_hit", "opponent_miss", "player_miss", "none"]):
        count += 1
        assert arm["grip"] == pytest.approx(sk.world(*p["hand"], p))         # grip = the hand joint
        v = arm["basis"][1]
        handle_point = tuple(c - along * vv for c, vv in zip(arm["center"], v))
        assert handle_point == pytest.approx(arm["grip"], abs=1e-9)            # handle sits in the hand
    assert count == 240


def test_paddle_is_a_rigid_extension_of_the_forearm():
    sk, g = Skeleton(), _game()
    for p, arm in _frames(sk, g, ["opponent_hit", "player_miss"]):
        if p["amount"]:
            continue
        v = arm["basis"][1]
        fa = math.radians(arm["forearm_deg"])
        angle = math.degrees(math.acos(max(-1.0, min(1.0, v[0] * math.cos(fa) + v[2] * math.sin(fa)))))
        assert angle == pytest.approx(config.SKELETON_WRIST_DEG, abs=1e-6)


def test_forehand_and_backhand_show_different_faces():
    sk = Skeleton()
    for stance, front_seen in (("forehand", True), ("backhand", False)):
        basis = paddle_basis(150 if stance == "forehand" else 10, stance)
        center = (sk.x - 0.3, sk.y, 0.3)
        assert CAM.facing(center, basis[2]) is front_seen


def test_stance_follows_the_incoming_ball():
    sk, g = Skeleton(), _game()
    g.state = "to_opponent"
    for end_x, expect in ((sk.x - 0.5, "forehand"), (sk.x + 0.5, "backhand")):
        g.flight = type("F", (), {"end_x": end_x, "t_end": 99.0})()
        sk.update(1.0 + (end_x > 0), g)
        assert sk.stance == expect


@pytest.mark.parametrize("stance", ["forehand", "backhand"])
def test_face_aims_toward_the_target(stance):
    center = (0.0, config.OPPONENT_HIT_Y_M + config.SKELETON_BEHIND_HIT_PLANE_M, 0.3)
    right = _visible_normal(center, paddle_basis(120, stance, aim_deg=25.0))
    left = _visible_normal(center, paddle_basis(120, stance, aim_deg=-25.0))
    assert right[0] > 0.3 and left[0] < -0.3
    assert right[1] < 0 and left[1] < 0                         # still facing me


@pytest.mark.parametrize("stance", ["forehand", "backhand"])
def test_face_tilts_for_spin(stance):
    center = (0.0, config.OPPONENT_HIT_Y_M, 0.3)
    flat = _visible_normal(center, paddle_basis(120, stance))
    top = _visible_normal(center, paddle_basis(120, stance, spin=Spin(top=1.0)))
    back = _visible_normal(center, paddle_basis(120, stance, spin=Spin(top=-1.0)))
    side = _visible_normal(center, paddle_basis(120, stance, spin=Spin(side=1.0)))
    assert abs(flat[2]) < 1e-9
    assert top[2] < -0.3          # closed face (brushing up over the ball)
    assert back[2] > 0.3          # open face
    assert side[0] > 0.3          # angled toward the curve
    none = paddle_basis(120, stance, spin=Spin(top=1.0), amount=0.0)
    assert sum(none, ()) == pytest.approx(sum(paddle_basis(120, stance), ()))


def test_swing_angles_the_face_toward_the_actual_shot():
    """After contact the face points at where the skeleton's shot is going."""
    sk, g = Skeleton(), _game()
    shot = g.flight
    hit_t = g.event_times["opponent_hit"]
    p = sk.pose(hit_t + 0.12, g)
    assert p["mood"] == "swing" and p["amount"] > 0.9
    assert p["spin"] == shot.spin
    want = Skeleton._aim_deg(shot.start_x, shot.end_x - shot.curve_x)
    assert p["aim_deg"] == pytest.approx(want)


def test_elbow_never_flips_while_the_hand_moves():
    from skeleton import SHOULDER, _elbow, _lerp
    paths = []
    for st in STANCES.values():
        for a, b in (("ready", "back"), ("back", "contact"), ("contact", "follow"), ("follow", "ready")):
            paths.append([_lerp(st[a], st[b], i / 200) for i in range(201)])
    paths.append([_lerp(STANCES["forehand"]["ready"], STANCES["backhand"]["ready"], i / 200) for i in range(201)])
    for path in paths:
        for h0, h1 in zip(path, path[1:]):
            assert math.dist(_elbow(SHOULDER, h0), _elbow(SHOULDER, h1)) < 4 * math.dist(h0, h1) + 1e-9


# =============================================================================
# Decorations
# =============================================================================

@pytest.mark.parametrize("size", [(1100, 720), (760, 560), (1600, 900), (640, 480)])
def test_decorations_never_overlap_each_other_or_gameplay(size):
    cam = Camera(*size)
    ghosts, pumpkins = layout_decorations(cam)
    assert len(pumpkins) >= 3 and len(ghosts) >= 2
    rects = [ghost_rect(cam, *g) for g in ghosts] + [pumpkin_rect(cam, x, y) for x, y, _ in pumpkins]
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert not rects_overlap(a, b)
        for hull in gameplay_hulls(cam):
            assert not rect_hits_hull(a, hull)


def test_decoration_layout_is_deterministic():
    assert layout_decorations(CAM) == layout_decorations(CAM)
    assert layout_decorations(CAM, seed=1) != layout_decorations(CAM, seed=2)


def test_hand_placed_spots_are_checked(capsys):
    x, y, z = layout_decorations(CAM)[1][0]                    # a spot the automatic layout accepts
    _, pumpkins = layout_decorations(CAM, pumpkin_spots=[(x, y), (x, y), (0.0, 1.0)], ghost_spots=[])
    assert pumpkins == [(x, y, z)]                              # duplicate and on-the-table spots skipped
    assert capsys.readouterr().out.count("skipped") == 2


def test_ball_and_table_colours_contrast():
    r, g, b = config.TABLE_TOP_COLOR
    assert r > g > b                                            # orange
    assert min(config.BALL_COLOR) > 230                         # white ball
    assert sum(config.TABLE_LINE_COLOR) > 700                   # white lines
