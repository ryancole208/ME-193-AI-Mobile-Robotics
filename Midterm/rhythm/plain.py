"""
Plain look for Concert Rally: a dark backboard, a floor and the table -- nothing else.

    PlainTheme     -- plugs into ../scene.Scene like the ping-pong themes (pre-rendered once)
    OpponentPaddle -- the floating paddle at the far end: glides to the ball and swings on its hits
"""

import config
from paddle import Paddle, swing_pose
from theme import Theme, _alpha_surface, mix

BACKBOARD_Y_M = 5.5          # the backboard, metres beyond your end of the table


class PlainTheme(Theme):
    name = "plain"

    def __init__(self):
        self.colors = config.PLAIN_COLORS
        super().__init__()

    def make_backdrop(self, pg, cam):
        w, h = cam.w, cam.h
        surf = pg.Surface((w, h))
        zf = -config.TABLE_HEIGHT_M
        foot = cam.point(0, BACKBOARD_Y_M, zf)
        wall_bottom = int(foot[1]) if foot else int(cam.horizon_y)
        # backboard: a soft vertical gradient
        top, bottom = self.color("wall_top"), self.color("wall_bottom")
        for row in range(max(1, wall_bottom)):
            pg.draw.line(surf, mix(top, bottom, row / max(1, wall_bottom)), (0, row), (w, row))
        # floor: lighter toward you
        far, near = self.color("floor_far"), self.color("floor_near")
        for row in range(wall_bottom, h):
            pg.draw.line(surf, mix(far, near, (row - wall_bottom) / max(1, h - wall_bottom)), (0, row), (w, row))
        # where the backboard meets the floor
        pg.draw.rect(surf, self.color("baseboard"), (0, wall_bottom - 4, w, 5))
        # a faint shadow band at the foot of the backboard
        band = _alpha_surface(pg, w, 24)
        for r in range(24):
            pg.draw.line(band, (0, 0, 0, int(70 * (1 - r / 24))), (0, r), (w, r))
        surf.blit(band, (0, wall_bottom + 1))
        return surf


class OpponentPaddle:
    def __init__(self):
        self.paddle = Paddle(config.OPPONENT_PADDLE_FRONT, config.OPPONENT_PADDLE_BACK,
                             scale=config.PADDLE_DRAW_SCALE)

    def draw(self, screen, cam, x, swing_phase):
        dx = dy = dz = tilt = 0.0
        if swing_phase is not None:
            dx, dy, dz, tilt = swing_pose(swing_phase)
        center = (x - dx, config.OPPONENT_HIT_Y_M + 0.05 - dy * 0.6, config.PADDLE_HEIGHT_M + 0.05 + dz)
        self.paddle.draw(screen, cam, center, 0.0, base_yaw_deg=180.0, tilt_deg=-tilt)
        return center
