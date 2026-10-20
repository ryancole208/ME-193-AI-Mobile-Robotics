"""
Runtime sanity checks that only print warnings (they never drive the robot).

DirectDriveWatchdog
    Retail descriptions say a Controller can drive a motor directly when they
    share a Connection Card. Lever input must only reach OUR program. While we
    command 0 % and a lever is pushed, the wheels must not turn; if they do for
    DIRECT_DRIVE_TIME_S, we warn loudly. (Our MotorSender also re-sends its own
    command every MOTOR_RESEND_S, which takes control back during a round.)

WheelSignWatchdog
    LEFT/RIGHT_WHEEL_SIGN in config.py must make "forward" positive. If we command
    both wheels clearly forward for a while and a measured wheel speed is
    consistently negative, the sign is probably wrong (this is the input to the
    line-crossing direction logic, so a wrong sign would be serious).
"""

import config


class DirectDriveWatchdog:
    def __init__(self, log=print):
        self.log = log
        self._since = None
        self.tripped = False

    def update(self, now, commanded, sample, levers):
        if self.tripped or sample.left_speed is None or levers is None:
            return
        pushing = max(abs(levers[0]), abs(levers[1])) > config.TELEOP_DEADBAND
        moving = max(abs(sample.left_speed), abs(sample.right_speed)) > config.DIRECT_DRIVE_SPEED
        if commanded == (0, 0) and pushing and moving:
            if self._since is None:
                self._since = now
            elif now - self._since >= config.DIRECT_DRIVE_TIME_S:
                self.tripped = True
                self.log("[WARNING] The wheels turn while the program commands 0 % and a lever is "
                         "pushed: the Controller seems to drive the Double Motor DIRECTLY. Lever "
                         "input must only reach this program. Put the Controller on a different "
                         "Connection Card than the Double Motor (and update the card settings), "
                         "or check README > Controller pairing.")
        else:
            self._since = None


class WheelSignWatchdog:
    def __init__(self, log=print, hold_s=0.6):
        self.log = log
        self.hold_s = hold_s
        self._since = None
        self.tripped = False

    def update(self, now, commanded, sample):
        if self.tripped or sample.left_speed is None:
            return
        l = config.LEFT_WHEEL_SIGN * sample.left_speed
        r = config.RIGHT_WHEEL_SIGN * sample.right_speed
        forward_cmd = commanded[0] > 40 and commanded[1] > 40
        if forward_cmd and (l < -config.DIRECT_DRIVE_SPEED or r < -config.DIRECT_DRIVE_SPEED):
            if self._since is None:
                self._since = now
            elif now - self._since >= self.hold_s:
                self.tripped = True
                side = "LEFT" if l < 0 else "RIGHT"
                self.log(f"[WARNING] Commanding forward but the {side} wheel reads backward "
                         f"(left={sample.left_speed}, right={sample.right_speed}). If the robot "
                         f"really drives forward, flip {side}_WHEEL_SIGN in config.py "
                         f"(run hardware_check.py --spin to measure).")
        else:
            self._since = None
