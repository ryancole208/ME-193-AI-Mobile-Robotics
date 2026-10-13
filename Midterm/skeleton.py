"""
Friendly cartoon skeleton opponent -- purely visual.

It only reads GameLogic state (opponent_x, flight, state, event_times) plus the optional
Opponent.planned_shot, so the simulated opponent, the Dynamic RL agent or a future physical
robot all drive the same character.

The body is a flat figure in the plane y = OPPONENT_HIT_Y + behind; every body point is
projected through the perspective camera, so the hand and the 3D paddle line up exactly.
Body coordinates are metres: bx sideways (screen-right +), bz up from the floor.

Paddle arm kinematics (all here):
  * stance   -- forehand (ball on its paddle side = screen-left of the hips) or backhand (the
                arm crosses the body); chosen once per incoming ball. Forehand shows the
                paddle's front (purple) face to the camera, backhand its back (lime) face.
  * swing    -- ready -> backswing (starts before the predicted contact) -> contact ->
                follow-through -> recover, as hand keyframes; the elbow comes from 2-bone IK.
  * paddle   -- a rigid extension of the forearm (plus a fixed wrist bend). The handle stays
                locked in the hand every frame; drawn fingers wrap it. From the backswing on,
                the face yaws toward the shot's target and tilts for its spin (closed for
                topspin, open for backspin, angled for sidespin).
"""

import math

import config
from paddle import _rot, opponent_paddle
from spin import NO_SPIN

SHOULDER = (-0.18, 1.2)          # paddle shoulder (its right = screen left), relative to the hips
OTHER_SHOULDER = (0.18, 1.2)
ARM_L1 = ARM_L2 = 0.26
STANCES = {   # hand keyframes (bx, bz)
    "forehand": {"ready": (-0.32, 0.98), "back": (-0.50, 0.90), "contact": (-0.40, 1.03),
                 "follow": (0.06, 1.10)},
    "backhand": {"ready": (-0.06, 0.98), "back": (0.00, 0.93), "contact": (0.22, 1.03),
                 "follow": (0.28, 1.18)},
}
ELBOW_POLE = (-0.3, -1.0)        # the paddle elbow bends toward this direction (down and out) ...
ELBOW_FORWARD = 0.5              # ... and toward the camera when the hand lines up with it
STANCE_BLEND_S = 0.22            # time to turn between the forehand and backhand ready poses
SWING_BEFORE_S, SWING_AFTER_S = 0.18, 0.25
MISS_REACT_S = 1.4
DANCE_S = 1.8


def _ik(shoulder, hand, l1, l2, bend=1):
    """Elbow position for a 2-bone arm in 2D."""
    sx, sz = shoulder
    hx, hz = hand
    dx, dz = hx - sx, hz - sz
    d = max(1e-6, min(l1 + l2 - 1e-4, math.hypot(dx, dz)))
    a = math.acos(max(-1.0, min(1.0, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d))))
    base = math.atan2(dz, dx)
    ang = base - bend * a
    return sx + l1 * math.cos(ang), sz + l1 * math.sin(ang)


def _elbow(shoulder, hand, l1=ARM_L1, l2=ARM_L2, pole=ELBOW_POLE, forward=ELBOW_FORWARD):
    """Pole-vector IK seen from the front: the elbow bends toward `pole`; the part of the bend that
    can't point that way points at the camera (so it only shortens on screen). Unlike picking one
    of the two flat solutions, this never flips while the hand moves."""
    sx, sz = shoulder
    dx, dz = hand[0] - sx, hand[1] - sz
    n = math.hypot(dx, dz) or 1e-6
    d = max(1e-6, min(l1 + l2 - 1e-4, n))
    ux, uz = dx / n, dz / n
    along = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(0.0, l1 * l1 - along * along))
    pm = math.hypot(*pole)
    px, pz = pole[0] / pm, pole[1] / pm
    dot = px * ux + pz * uz
    qx, qz = px - dot * ux, pz - dot * uz            # pole, perpendicular to the upper-arm line
    k = h / math.sqrt(qx * qx + qz * qz + forward * forward)
    return sx + ux * along + qx * k, sz + uz * along + qz * k


def _lerp(a, b, k):
    return a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k


def _ease(k):
    k = max(0.0, min(1.0, k))
    return k * k * (3 - 2 * k)


def _cross(a, b):
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


def paddle_basis(forearm_deg, stance, aim_deg=0.0, spin=NO_SPIN, amount=1.0):
    """(u, v, n) of the skeleton's paddle. v follows the forearm (+ the wrist bend toward up),
    n is the front-face normal; `amount` (0..1) fades the aim/spin angling in and out."""
    wrist = config.SKELETON_WRIST_DEG * (-1 if stance == "forehand" else 1)
    a = math.radians(forearm_deg + wrist)
    v = (math.cos(a), 0.0, math.sin(a))
    n = (0.0, -1.0, 0.0) if stance == "forehand" else (0.0, 1.0, 0.0)   # forehand: front faces me
    u = _cross(n, v)
    basis = [u, v, n]
    m = config.SPIN_MAX or 1.0
    tilt = config.SKELETON_SPIN_TILT_DEG * (spin.top / m) * amount     # + closes the face (topspin)
    side = config.SKELETON_SIDESPIN_DEG * (spin.side / m) * amount
    if tilt:
        basis = [_rot(w, (1.0, 0.0, 0.0), tilt) for w in basis]
    if side:   # handle leans and the face angles toward the curve
        n_vis = (0.0, -1.0, 0.0)
        basis = [_rot(w, n_vis, -0.7 * side) for w in basis]
    yaw = aim_deg * amount + side
    if yaw:
        basis = [_rot(w, (0.0, 0.0, 1.0), yaw) for w in basis]
    return tuple(basis)


class Skeleton:
    def __init__(self):
        self.x = 0.0                     # hips' lateral position (m)
        self.paddle = opponent_paddle()
        self._last = None
        self.y = config.OPPONENT_HIT_Y_M + config.SKELETON_BEHIND_HIT_PLANE_M
        self.stance = "forehand"
        self.blend = 0.0                 # 0 = forehand ready pose, 1 = backhand ready pose
        self._stance_flight = None

    # -- motion --------------------------------------------------------------------
    def _contact_x(self, logic):
        if logic.state == "to_opponent" and logic.flight is not None:
            return logic.flight.end_x
        return logic.opponent_x

    def _target_x(self, logic):
        return self._contact_x(logic) - STANCES[self.stance]["contact"][0]

    def update(self, now, logic):
        dt = 0.0 if self._last is None else max(0.0, min(0.1, now - self._last))
        self._last = now
        f = logic.flight
        if logic.state == "to_opponent" and f is not None and f is not self._stance_flight:
            self._stance_flight = f   # a new ball is coming: forehand or backhand?
            self.stance = "backhand" if f.end_x - self.x > config.SKELETON_BACKHAND_SIDE_M else "forehand"
        elif logic.state in ("serve_wait", "menu"):
            self.stance = "forehand"
        goal = 1.0 if self.stance == "backhand" else 0.0
        db = dt / STANCE_BLEND_S
        self.blend += max(-db, min(db, goal - self.blend))
        target = self._target_x(logic)
        step = config.SKELETON_TRACK_SPEED_M_S * dt
        self.x += max(-step, min(step, target - self.x))
        lim = config.TABLE_WIDTH_M / 2 + 0.5
        self.x = max(-lim, min(lim, self.x))

    @staticmethod
    def _aim_deg(contact_x, target_x):
        dist = config.OPPONENT_HIT_Y_M - config.PLAYER_HIT_Y_M
        deg = math.degrees(math.atan2(target_x - contact_x, dist)) * config.SKELETON_AIM_EXAGGERATION
        m = config.SKELETON_AIM_MAX_DEG
        return max(-m, min(m, deg))

    def pose(self, now, logic):
        """Everything that animates, as a dict (pure given self.x / self.stance -- unit tested)."""
        ev = logic.event_times
        t = now
        st = STANCES[self.stance]
        ready = _lerp(STANCES["forehand"]["ready"], STANCES["backhand"]["ready"], _ease(self.blend))
        p = {"bob": 0.015 * math.sin(t * 2.4), "sway": 0.02 * math.sin(t * 1.3), "jaw": 0.0,
             "head_tilt": 4.0 * math.sin(t * 1.1), "hand": ready, "other_hand": (0.30, 0.86),
             "mood": "idle", "eyes_x": 0.0, "stance": "backhand" if self.blend > 0.5 else "forehand",
             "aim_deg": 0.0, "spin": NO_SPIN, "amount": 0.0}
        ball = logic.ball_position(now)
        if ball is not None:
            p["eyes_x"] = max(-1.0, min(1.0, (ball[0] - self.x) / 0.8))

        f = logic.flight
        hit_t = ev.get("opponent_hit")
        tc = (f.t_end + config.OPPONENT_REACTION_S) if (logic.state == "to_opponent" and f is not None) else None
        if tc is not None and now >= tc - SWING_BEFORE_S:
            # backswing toward the predicted contact; wind up toward a reply that is already chosen
            k = _ease((now - (tc - SWING_BEFORE_S)) / SWING_BEFORE_S)
            p["hand"] = _lerp(ready, st["back"], k)
            p["stance"] = self.stance
            p["mood"] = "swing"
            plan = logic.opponent.planned_shot if hasattr(logic, "opponent") else None
            if plan is not None:
                p["aim_deg"] = self._aim_deg(f.end_x, plan[0])
                p["spin"] = plan[1]
                p["amount"] = 0.5 * k
        elif hit_t is not None and 0 <= now - hit_t <= SWING_AFTER_S + config.SKELETON_RECOVER_S \
                and logic.state != "menu":
            d = now - hit_t
            # its shot is the flight launched at hit_t (already prev_flight if I swung early)
            shot = next((fl for fl in (f, logic.prev_flight) if fl is not None and abs(fl.t0 - hit_t) < 1e-6),
                        None)
            if d <= SWING_AFTER_S:
                k = d / SWING_AFTER_S
                if k < 0.3:
                    p["hand"] = _lerp(st["back"], st["contact"], _ease(k / 0.3))
                else:
                    p["hand"] = _lerp(st["contact"], st["follow"], 1 - (1 - (k - 0.3) / 0.7) ** 2)
                p["amount"] = min(1.0, 0.5 + k * 2)
                p["mood"] = "swing"
            else:
                k = _ease((d - SWING_AFTER_S) / config.SKELETON_RECOVER_S)
                p["hand"] = _lerp(st["follow"], ready, k)
                p["amount"] = 1.0 - k
            p["stance"] = self.stance
            if shot is not None:
                p["aim_deg"] = self._aim_deg(shot.start_x, shot.end_x - shot.curve_x)
                p["spin"] = shot.spin
            else:
                p["amount"] = 0.0

        miss_t = ev.get("opponent_miss")
        if miss_t is not None and 0 <= now - miss_t <= MISS_REACT_S and logic.state != "menu":
            k = 1 - (now - miss_t) / MISS_REACT_S
            p["jaw"] = 0.07 * min(1.0, (now - miss_t) * 6) * (0.4 + 0.6 * k)
            p["head_tilt"] = 14.0 * math.sin((now - miss_t) * 16) * k
            p["hand"] = (ready[0] - 0.05, ready[1] - 0.12)
            p["mood"], p["amount"] = "oops", 0.0

        dance_t = ev.get("player_miss")
        if dance_t is not None and 0 <= now - dance_t <= DANCE_S and logic.state != "menu":
            d = now - dance_t
            p["sway"] = 0.07 * math.sin(d * 9)
            p["bob"] = 0.04 * abs(math.sin(d * 9))
            p["head_tilt"] = 12.0 * math.sin(d * 9 + 0.6)
            p["hand"] = (-0.30 + 0.06 * math.sin(d * 9), 1.55 + 0.05 * math.sin(d * 18))
            p["other_hand"] = (0.30 + 0.06 * math.sin(d * 9 + 3), 1.55 + 0.05 * math.sin(d * 18 + 3))
            p["mood"], p["amount"] = "dance", 0.0
        return p

    # -- kinematics ----------------------------------------------------------------
    def world(self, bx, bz, p):
        """Body point -> world metres (the same mapping for the bones and the 3D paddle)."""
        return (self.x + bx + p["sway"] * bz / 1.5, self.y, -config.TABLE_HEIGHT_M + bz + p["bob"])

    def arm(self, p):
        """Paddle arm + paddle: shoulder/elbow/hand (body), grip (world), basis, blade centre."""
        hand = p["hand"]
        elbow = _elbow(SHOULDER, hand)
        forearm = math.degrees(math.atan2(hand[1] - elbow[1], hand[0] - elbow[0]))
        basis = paddle_basis(forearm, p["stance"], p["aim_deg"], p["spin"], p["amount"])
        grip = self.world(*hand, p)
        return {"shoulder": SHOULDER, "elbow": elbow, "hand": hand, "forearm_deg": forearm, "grip": grip,
                "basis": basis, "center": self.paddle.center_for_grip(grip, basis[1])}

    # -- drawing -------------------------------------------------------------------
    def draw(self, screen, cam, now, logic):
        import pygame as pg
        root = cam.project(self.x, self.y, -config.TABLE_HEIGHT_M)
        if root is None:
            return
        rx, ry, s = root
        p = self.pose(now, logic)
        bone, outline = config.SKELETON_BONE, config.SKELETON_OUTLINE

        def S(bx, bz):
            q = cam.point(*self.world(bx, bz, p))
            return q if q is not None else (rx + bx * s, ry - bz * s)

        lw = max(2, int(0.035 * s))
        ow = lw + max(2, int(0.012 * s))

        def bone_line(a, b, width=lw):
            pa, pb = S(*a), S(*b)
            pg.draw.line(screen, outline, pa, pb, width + (ow - lw))
            pg.draw.circle(screen, outline, pa, (width + ow - lw) / 2)
            pg.draw.circle(screen, outline, pb, (width + ow - lw) / 2)
            pg.draw.line(screen, bone, pa, pb, width)
            for q in (pa, pb):
                pg.draw.circle(screen, bone, q, width / 2)

        def knob(a, r):
            q = S(*a)
            pg.draw.circle(screen, outline, q, r * s + (ow - lw) / 2)
            pg.draw.circle(screen, bone, q, r * s)

        # soft shadow on the floor
        sh = pg.Surface((max(2, int(0.7 * s)), max(2, int(0.16 * s))), pg.SRCALPHA)
        pg.draw.ellipse(sh, (5, 0, 15, 110), sh.get_rect())
        screen.blit(sh, sh.get_rect(center=(rx, ry)))

        # legs (mostly hidden behind the table, but there for the dance)
        hip_z = 0.82
        for side in (-1, 1):
            hip, knee, foot = (side * 0.08, hip_z), (side * 0.10, 0.42), (side * 0.11, 0.05)
            bone_line(hip, knee)
            bone_line(knee, foot)
            knob(knee, 0.03)
            pg.draw.ellipse(screen, bone, (*[v - 0.06 * s for v in S(side * 0.13, 0.05)], 0.12 * s, 0.06 * s))
        # pelvis
        knob((0, hip_z), 0.07)
        # spine + ribs
        bone_line((0, hip_z), (0, 1.22), max(2, lw - 1))
        for i in range(5):
            knob((0, hip_z + 0.05 + i * 0.075), 0.022)
        for i, z in enumerate((1.16, 1.08, 1.00)):
            wdt = 0.17 - i * 0.02
            rect = pg.Rect(0, 0, 2 * wdt * s, 0.11 * s)
            rect.center = S(0, z)
            pg.draw.arc(screen, outline, rect.inflate(ow - lw, ow - lw), math.pi * 1.05, math.pi * 1.95, ow)
            pg.draw.arc(screen, bone, rect, math.pi * 1.05, math.pi * 1.95, lw)
        # bow tie
        bt = S(0, 1.24)
        bw = 0.06 * s
        pg.draw.polygon(screen, config.SKELETON_BOWTIE, [(bt[0], bt[1]), (bt[0] - bw, bt[1] - bw * 0.6),
                                                         (bt[0] - bw, bt[1] + bw * 0.6)])
        pg.draw.polygon(screen, config.SKELETON_BOWTIE, [(bt[0], bt[1]), (bt[0] + bw, bt[1] - bw * 0.6),
                                                         (bt[0] + bw, bt[1] + bw * 0.6)])
        pg.draw.circle(screen, (200, 90, 10), bt, bw * 0.3)

        # free arm
        elbow = _ik(OTHER_SHOULDER, p["other_hand"], ARM_L1, ARM_L2, bend=-1)
        knob(OTHER_SHOULDER, 0.035)
        bone_line(OTHER_SHOULDER, elbow)
        bone_line(elbow, p["other_hand"])
        knob(elbow, 0.026)
        knob(p["other_hand"], 0.035)

        # paddle arm, the paddle locked in its hand, then fingers wrapped round the handle
        arm = self.arm(p)
        knob(SHOULDER, 0.035)
        bone_line(SHOULDER, arm["elbow"])
        bone_line(arm["elbow"], arm["hand"])
        knob(arm["elbow"], 0.026)
        self.paddle.draw_basis(screen, cam, arm["center"], arm["basis"])
        self._draw_fist(pg, screen, cam, arm, s, outline, bone, ow - lw)

        self._draw_skull(pg, screen, S, s, p, outline, bone, ow, lw)

    @staticmethod
    def _draw_fist(pg, screen, cam, arm, s, outline, bone, edge):
        g = cam.point(*arm["grip"])
        v = arm["basis"][1]
        q = cam.point(*(arm["grip"][i] + 0.05 * v[i] for i in range(3)))
        if g is None or q is None:
            return
        dx, dy = q[0] - g[0], q[1] - g[1]
        length = math.hypot(dx, dy) or 1.0
        dx, dy = dx / length, dy / length          # handle direction on screen
        px, py = -dy, dx                            # across the handle
        r = 0.016 * s
        parts = [((g[0] - px * r * 0.6, g[1] - py * r * 0.6), r * 1.7)]                  # palm
        parts += [((g[0] + dx * i * r * 1.1 + px * r, g[1] + dy * i * r * 1.1 + py * r), r)
                  for i in (-1, 0, 1)]                                                   # knuckles
        parts.append(((g[0] - px * r * 1.6 + dx * r * 1.4, g[1] - py * r * 1.6 + dy * r * 1.4), r * 0.75))  # thumb
        for c, rr in parts:
            pg.draw.circle(screen, outline, c, rr + edge / 2 + 1)
        for c, rr in parts:
            pg.draw.circle(screen, bone, c, rr)
        for i in (-0.5, 0.5):                       # finger gaps
            a = (g[0] + dx * i * r * 1.1 + px * r * 0.3, g[1] + dy * i * r * 1.1 + py * r * 0.3)
            b = (a[0] + px * r * 1.3, a[1] + py * r * 1.3)
            pg.draw.line(screen, outline, a, b, max(1, int(r * 0.25)))

    def _draw_skull(self, pg, screen, S, s, p, outline, bone, ow, lw):
        cx, cy = S(0, 1.47)
        r = 0.17 * s
        tilt = math.radians(p["head_tilt"])

        def R(dx, dz):
            """Skull-local metres -> screen, rotated by the head tilt."""
            x, z = dx * math.cos(tilt) - dz * math.sin(tilt), dx * math.sin(tilt) + dz * math.cos(tilt)
            return cx + x * s, cy - z * s

        # jaw (drops on a miss)
        jaw_c = R(0, -0.13 - p["jaw"])
        jaw = pg.Rect(0, 0, 0.2 * s, 0.09 * s)
        jaw.center = jaw_c
        pg.draw.ellipse(screen, outline, jaw.inflate(ow - lw + 2, ow - lw + 2))
        pg.draw.ellipse(screen, bone, jaw)
        # cranium
        pg.draw.circle(screen, outline, (cx, cy), r + (ow - lw) / 2 + 1)
        pg.draw.circle(screen, bone, (cx, cy), r)
        # cheek bones / upper jaw
        up = pg.Rect(0, 0, 0.24 * s, 0.1 * s)
        up.center = R(0, -0.09)
        pg.draw.ellipse(screen, bone, up)
        # eyes: big round sockets with glowing pupils looking at the ball
        er = 0.052 * s
        for dx in (-0.065, 0.065):
            ec = R(dx, 0.01)
            if p["mood"] == "oops":   # dizzy spiral eyes
                pg.draw.circle(screen, config.SKELETON_EYES, ec, er)
                for i in range(3):
                    rr = er * (0.3 + 0.25 * i)
                    pg.draw.arc(screen, config.SKELETON_EYE_GLOW, (ec[0] - rr, ec[1] - rr, 2 * rr, 2 * rr),
                                i * 1.3, i * 1.3 + math.pi * 1.4, max(1, int(er * 0.15)))
            elif p["mood"] == "dance":  # happy closed eyes
                pg.draw.arc(screen, config.SKELETON_EYES, (ec[0] - er, ec[1] - er * 0.6, 2 * er, 1.6 * er),
                            0, math.pi, max(2, int(er * 0.35)))
            else:
                pg.draw.ellipse(screen, config.SKELETON_EYES, (ec[0] - er, ec[1] - er * 1.15, 2 * er, 2.3 * er))
                px = ec[0] + p["eyes_x"] * er * 0.45
                pg.draw.circle(screen, config.SKELETON_EYE_GLOW, (px, ec[1] + er * 0.1), er * 0.38)
                pg.draw.circle(screen, (255, 255, 255), (px - er * 0.15, ec[1] - er * 0.1), max(1, er * 0.13))
        # nose: little upside-down heart
        n = R(0, -0.055)
        nw = 0.022 * s
        pg.draw.polygon(screen, config.SKELETON_EYES, [(n[0] - nw, n[1] - nw * 0.4), (n[0] + nw, n[1] - nw * 0.4),
                                                       (n[0], n[1] + nw)])
        # goofy grin with teeth
        g0, g1 = R(-0.09, -0.11), R(0.09, -0.11)
        mouth = pg.Rect(0, 0, abs(g1[0] - g0[0]) + 2, 0.06 * s)
        mouth.center = R(0, -0.125)
        width = max(2, int(0.012 * s))
        pg.draw.arc(screen, outline, mouth, math.pi * 1.05, math.pi * 1.95, width)
        for i in range(-3, 4):
            a = math.pi * 1.5 + i * 0.12
            x = mouth.centerx + math.cos(a) * mouth.w / 2
            y = mouth.centery - math.sin(a) * mouth.h / 2
            pg.draw.line(screen, outline, (x, y - 0.018 * s), (x, y), max(1, width // 2))
        if p["jaw"] > 0.02:   # open mouth shows a dark gap
            gap = pg.Rect(0, 0, 0.14 * s, p["jaw"] * s)
            gap.midtop = R(0, -0.15)
            pg.draw.ellipse(screen, config.SKELETON_EYES, gap)
