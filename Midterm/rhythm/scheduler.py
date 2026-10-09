"""
Ball scheduling: every flight is computed from YOUR hit beats, and every ball position is a pure
function of song time -- no state is advanced per frame, so a dropped frame can never put the
ball out of time.

Timing rule: the ball spends equal time in the air each way, so the opponent hits exactly
halfway (in beats) between two of your hits. With 4/4 bars:

  [4]     you hit on 4 -> the opponent hits on the next bar's 2: 2 beats each way
  [2, 4]  you hit on 2 and 4 -> the opponent hits on 1 and 3: 1 beat each way

Switching patterns needs no special case: [4] -> [2, 4] puts the opponent on the next bar's 1
(you on 4, then 2: halfway is 1), [2, 4] -> [4] puts it on the next bar's 2. The first serve
comes half of the first gap before your first hit; your last return takes half of the last gap.
If the tempo changes during a flight, the opponent stays on the beat, so the two directions can
differ by a few ms.

Every hit is a Contact (serve beat, your beat, return beat): its flight times in seconds are
stored explicitly (from the tempo map), so later charts can give hits any flight length
(0.5 to 8 beats) without changing the ball code.

World coordinates are ping pong's (metres): x lateral, y along the table (0 = your end),
z above the table surface.
"""

import math
from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Contact:
    """One of your hits: the opponent serves on serve_beat, you hit on beat, your return reaches
    the opponent on return_beat (chart beats)."""
    serve_beat: float
    beat: float
    return_beat: float
    warn: bool = False


def contacts_from_cues(cues):
    """The equal-flight rule: each opponent hit is halfway between two of your hits."""
    beats = [c.beat for c in cues]
    out = []
    for i, c in enumerate(cues):
        prev_gap = beats[i] - beats[i - 1] if i > 0 else (beats[1] - beats[0] if len(beats) > 1 else 4.0)
        next_gap = beats[i + 1] - beats[i] if i + 1 < len(beats) else prev_gap
        out.append(Contact(c.beat - prev_gap / 2, c.beat, c.beat + next_gap / 2, c.warn))
    return out


@dataclass(frozen=True)
class Seg:
    """One ballistic segment: straight in x/y, a parabola in z."""
    t0: float
    t1: float
    p0: tuple
    p1: tuple
    h: float

    def pos(self, t):
        d = self.t1 - self.t0
        f = 0.0 if d <= 0 else min(1.0, max(0.0, (t - self.t0) / d))
        x = self.p0[0] + (self.p1[0] - self.p0[0]) * f
        y = self.p0[1] + (self.p1[1] - self.p0[1]) * f
        z = self.p0[2] + (self.p1[2] - self.p0[2]) * f + 4 * self.h * f * (1 - f)
        return x, y, z

    def velocity_end(self):
        d = max(1e-6, self.t1 - self.t0)
        return ((self.p1[0] - self.p0[0]) / d, (self.p1[1] - self.p0[1]) / d,
                ((self.p1[2] - self.p0[2]) - 4 * self.h) / d)


def make_seg(t0, t1, p0, p1):
    """Segment whose arc height follows real gravity, raised if needed to clear the net."""
    d = max(1e-6, t1 - t0)
    h = config.ARC_GRAVITY_M_S2 * d * d / 8
    net_y = config.TABLE_LENGTH_M / 2
    if (p0[1] - net_y) * (p1[1] - net_y) < 0:
        fn = (net_y - p0[1]) / (p1[1] - p0[1])
        need = config.NET_HEIGHT_M + config.ARC_MIN_NET_CLEARANCE_M + config.BALL_RADIUS_M
        lin = p0[2] + (p1[2] - p0[2]) * fn
        h = max(h, (need - lin) / max(1e-6, 4 * fn * (1 - fn)))
    return Seg(t0, t1, p0, p1, min(h, config.ARC_MAX_HEIGHT_M))


@dataclass
class Flight:
    """Everything about one of your hits: the opponent's serve, the incoming legs, your return."""
    index: int
    beat: float            # chart beat of your hit
    warn: bool
    upbeat: bool
    t_serve: float         # opponent hits (song s)
    t_bounce_in: float     # bounce on your side
    t_arrive: float        # THE CUE: ball at your paddle
    t_bounce_out: float    # your return bounces on the far side
    t_return: float        # your return reaches the opponent (= the next serve)
    flight_in_s: float     # t_arrive - t_serve, stored explicitly
    flight_out_s: float    # t_return - t_arrive
    spb: float             # the music's beat length at your hit (for the keep-ups)
    x_from: float          # where the opponent serves from
    x_arrive: float        # where the ball reaches your end (a visual cue only: aim isn't scored)
    x_target: float        # where your return goes
    t_warn: float | None
    seg_in: tuple          # (opponent -> bounce, bounce -> you)
    seg_out: tuple         # (you -> bounce, bounce -> opponent)


def _spread(i, phase, spread):
    """Varied but deterministic x positions (golden-ratio sequence)."""
    k = (i * 0.6180339887 + phase) % 1.0
    return (2 * k - 1) * spread


def return_target_x(i):
    """Where your i-th return lands."""
    return _spread(i, 0.31, config.RETURN_TARGET_SPREAD_M)


def arrival_x(i):
    """Where the i-th ball reaches your end of the table."""
    return _spread(i, 0.77, config.ARRIVAL_X_SPREAD_M)


def build_schedule(chart, contacts=None):
    """List of Flight, one per hit, in song seconds. Every beat -> time conversion uses the
    chart's tempo map, so flights shrink when the tempo rises."""
    contacts = contacts_from_cues(chart.cues) if contacts is None else contacts
    bt = chart.beat_time
    yp, yo = config.PLAYER_HIT_Y_M, config.OPPONENT_HIT_Y_M
    zp = zo = config.PADDLE_HEIGHT_M
    lead = config.BOUNCE_LEAD_BEATS
    flights = []
    x_from = 0.0
    for i, c in enumerate(contacts):
        f_in, f_out = c.beat - c.serve_beat, c.return_beat - c.beat
        H, T, R = bt(c.serve_beat), bt(c.beat), bt(c.return_beat)
        b_in = bt(c.beat - min(lead, f_in / 2))
        b_out = bt(c.return_beat - min(lead, f_out / 2))
        x_me = arrival_x(i)
        x_t = return_target_x(i)
        # bounce point x: proportional to how far along the table the bounce is
        p_opp = (x_from, yo, zo)
        bx = x_from + (x_me - x_from) * (yo - config.BOUNCE_Y_NEAR_M) / (yo - yp)
        p_bin = (bx, config.BOUNCE_Y_NEAR_M, 0.0)
        p_me = (x_me, yp, zp)
        bx2 = x_me + (x_t - x_me) * (config.BOUNCE_Y_FAR_M - yp) / (yo - yp)
        p_bout = (bx2, config.BOUNCE_Y_FAR_M, 0.0)
        p_back = (x_t, yo, zo)
        warn_t = bt(c.beat - config.WARNING_LEAD_BEATS) if c.warn else None
        flights.append(Flight(
            index=i, beat=c.beat, warn=c.warn, upbeat=abs(c.beat - round(c.beat)) > 1e-6,
            t_serve=H, t_bounce_in=b_in, t_arrive=T, t_bounce_out=b_out, t_return=R,
            flight_in_s=T - H, flight_out_s=R - T, spb=chart.spb_at(c.beat),
            x_from=x_from, x_arrive=x_me, x_target=x_t, t_warn=warn_t,
            seg_in=(make_seg(H, b_in, p_opp, p_bin), make_seg(b_in, T, p_bin, p_me)),
            seg_out=(make_seg(T, b_out, p_me, p_bout), make_seg(b_out, R, p_bout, p_back))))
        x_from = x_t
    return flights


# =============================================================================
# Ball position at a song time, given what has been judged so far
# =============================================================================

@dataclass(frozen=True)
class Outcome:
    """What happened to a cue, as far as the game knows right now."""
    hit: bool
    t_known: float          # song time the game learned it (hits can arrive late from the IMU)


@dataclass(frozen=True)
class BallView:
    pos: tuple
    index: int              # flight index
    phase: str              # "hold", "in", "out", "pending", "fall"
    alpha: float = 1.0
    upbeat: bool = False


def _hold_pos(x, t, t0, spb):
    """Ball bouncing gently on the opponent's paddle (keep-ups on the beat)."""
    k = abs(math.sin(math.pi * (t - t0) / spb))
    return x, config.OPPONENT_HIT_Y_M - 0.04, config.PADDLE_HEIGHT_M + 0.06 + 0.16 * k


def _overshoot(flight, t):
    seg = flight.seg_in[1]
    v = seg.velocity_end()
    tau = config.MISS_OVERSHOOT_TAU_S
    k = tau * (1 - math.exp(-max(0.0, t - seg.t1) / tau))
    return seg.p1[0] + v[0] * k, seg.p1[1] + v[1] * k, seg.p1[2] + v[2] * k


def flight_index_at(flights, t):
    """Index of the last flight whose serve is at or before t (-1 = before the first serve)."""
    lo, hi = 0, len(flights)
    while lo < hi:
        mid = (lo + hi) // 2
        if flights[mid].t_serve <= t:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


def ball_at(flights, t, outcomes, x_at=None):
    """BallView at song time t (or None). outcomes: {index: Outcome} for judged/registered cues.

    x_at (the "follow" game mode): x_at(index) -> the x the ball should reach at your end
    instead of its planned x_arrive (your paddle's x). The path bends smoothly: the shift grows
    with how far the ball has come toward you, and shrinks again on your return."""
    view = _ball_at(flights, t, outcomes)
    if x_at is None or view is None or view.phase not in ("in", "out", "pending", "fall"):
        return view
    fl = flights[view.index]
    x, y, z = view.pos
    yp, yo = config.PLAYER_HIT_Y_M, config.OPPONENT_HIT_Y_M
    near = min(1.0, max(0.0, (yo - y) / (yo - yp)))          # 0 at the opponent, 1 at your paddle
    k = near if view.phase in ("in", "out") else 1.0
    return BallView((x + (x_at(view.index) - fl.x_arrive) * k, y, z), view.index, view.phase, view.alpha,
                    view.upbeat)


def _ball_at(flights, t, outcomes):
    if not flights:
        return None
    i = flight_index_at(flights, t)
    if i < 0:
        f0 = flights[0]
        if t >= f0.t_serve - 2 * f0.spb:       # opponent waits with the ball before the first serve
            return BallView(_hold_pos(f0.x_from, t, f0.t_serve, f0.spb), 0, "hold", upbeat=f0.upbeat)
        return None
    fl = flights[i]
    if t <= fl.t_arrive:
        seg = fl.seg_in[0] if t < fl.t_bounce_in else fl.seg_in[1]
        alpha = 1.0
        prev = outcomes.get(i - 1)
        if i > 0 and not (prev and prev.hit):  # previous ball was missed: this one fades in
            alpha = min(1.0, (t - fl.t_serve) / config.SERVE_FADE_S)
        return BallView(seg.pos(t), i, "in", alpha, fl.upbeat)

    out = outcomes.get(i)
    if out is not None and out.hit:
        if t <= fl.t_return:
            seg = fl.seg_out[0] if t < fl.t_bounce_out else fl.seg_out[1]
            p = seg.pos(t)
            if out.t_known > fl.t_arrive and t < out.t_known + config.LATE_HIT_BLEND_S:
                # The hit registered late (IMU delay): blend from the overshoot onto the return.
                q = _overshoot(fl, t)
                k = 0.0 if t < out.t_known else (t - out.t_known) / config.LATE_HIT_BLEND_S
                p = tuple(a + (b - a) * k for a, b in zip(q, p))
            return BallView(p, i, "out")
        return BallView(_hold_pos(fl.x_target, t, fl.t_return, fl.spb), i, "hold")
    if out is None:
        return BallView(_overshoot(fl, t), i, "pending")
    # a Miss: drop off the table and fade
    dt = max(0.0, t - out.t_known)
    if dt >= config.MISS_FALL_S:
        return None
    x, y, z = _overshoot(fl, out.t_known)
    return BallView((x, y - 0.6 * dt, z - 4.9 * dt * dt), i, "fall", 1.0 - dt / config.MISS_FALL_S)


def opponent_x(flights, t):
    """The floating paddle glides toward where your return will land (pure function of t)."""
    if not flights:
        return 0.0
    i = flight_index_at(flights, t)
    if i < 0:
        return flights[0].x_from
    fl = flights[i]
    t0 = fl.t_serve + config.OPPONENT_SWING_S / 2
    t1 = max(t0 + 1e-3, fl.t_return - config.OPPONENT_SWING_S / 2)
    k = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
    k = k * k * (3 - 2 * k)
    return fl.x_from + (fl.x_target - fl.x_from) * k


def opponent_swing_phase(flights, t):
    """0..1 while the opponent swings at a serve (centred on the hit), else None."""
    i = flight_index_at(flights, t + config.OPPONENT_SWING_S / 2)
    if i < 0:
        return None
    phase = (t - (flights[i].t_serve - config.OPPONENT_SWING_S / 2)) / config.OPPONENT_SWING_S
    return phase if 0.0 <= phase <= 1.0 else None
