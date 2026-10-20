"""
Bluetooth side: the Double Motor (wheels + IMU), Color Sensor and Controller.

Threads involved (all created by this module or the legoeducation library):
  * legoeducation's event-loop thread: receives notifications and runs our
    notification callbacks. Callbacks only record data and timestamps; they must
    never call the synchronous API (it would deadlock).
  * MotorSender thread: the ONLY place motor commands are sent from. The main loop
    drops the latest (left, right) into a mailbox and moves on. The sender sends it
    with blocking=False (BLE write-without-response, no RPC acknowledgment awaited),
    so the behavior-tree loop never waits on Bluetooth.
  * Reconnect thread: if a device drops mid-round, it keeps trying to reconnect
    while the main loop goes on (stale data makes the tree stop safely).

The same interface is implemented by simulator.SimHardware, so the main loop
cannot tell the difference.
"""

import math
import threading
import time
from collections import deque

import legoeducation as le

import config
from motion import MotionSample


def _finite(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


# ---------------------------------------------------------------------------
# Robot frame <-> Double Motor frame
# The whole program thinks in the ROBOT frame: "forward" = toward the color
# sensor, "left" = the robot's left wheel. These two functions are the only
# place that knows how the Double Motor is actually mounted (config.DRIVE_*).
# ---------------------------------------------------------------------------

def to_motor_frame(left, right, inverted=None, swap=None):
    """Robot (left %, right %) -> arguments for movement_move_tank(motor left, motor right)."""
    inverted = config.DRIVE_INVERTED if inverted is None else inverted
    swap = config.DRIVE_SWAP_SIDES if swap is None else swap
    if swap:
        left, right = right, left
    if inverted:
        left, right = -left, -right
    return left, right


def from_motor_frame(m0_value, m1_value, swap=None):
    """(motor[0] reading, motor[1] reading) -> (robot-left reading, robot-right reading).
    Only the sides can swap here; the sign of each reading is handled by
    LEFT/RIGHT_WHEEL_SIGN, which hardware_check.py measures in the robot frame."""
    swap = config.DRIVE_SWAP_SIDES if swap is None else swap
    return (m1_value, m0_value) if swap else (m0_value, m1_value)


class Drive:
    """Holds the commanded wheel speeds. Subclasses decide what 'apply' means."""

    def __init__(self):
        self.commanded = (0.0, 0.0)

    def set(self, left, right):
        cmd = (round(max(-100.0, min(100.0, left))), round(max(-100.0, min(100.0, right))))
        if cmd != self.commanded:
            self.commanded = cmd
            self._apply(*cmd)

    def stop(self):
        self.set(0, 0)

    def _apply(self, left, right):
        pass


class NullDrive(Drive):
    """--no-motors: never touches the wheels, just reports what it would do."""

    def _apply(self, left, right):
        print(f"[dry-run] motors would be set to left={left:+d}% right={right:+d}%")


class MotorSender(Drive):
    """Real motors, sent from a background thread (latest command wins)."""

    def __init__(self, doublemotor, resend_s=config.MOTOR_RESEND_S):
        super().__init__()
        self._dm = doublemotor
        self._resend_s = resend_s
        self._cv = threading.Condition()
        self._pending = True
        self._stop = False
        self.errors = 0
        self.sends = 0          # successful movement_move_tank calls (shown in the status line)
        self.last_sent = None   # motor-frame (left, right) of the last successful send
        self._thread = threading.Thread(target=self._run, name="MotorSender", daemon=True)
        self._thread.start()

    def _apply(self, left, right):
        with self._cv:
            self._pending = True
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._pending or self._stop, timeout=self._resend_s)
                if self._stop:
                    return
                self._pending = False
                cmd = self.commanded
            # Unchanged commands are re-sent every resend_s too: if anything else
            # (e.g. a Controller paired directly) moved the wheels, we take over again.
            if self._dm.connected:
                try:
                    frame = to_motor_frame(*cmd)
                    self._dm.movement_move_tank(*frame, blocking=False)
                    self.sends += 1
                    self.last_sent = frame
                except Exception as e:
                    self.errors += 1
                    if self.errors <= 5 or self.errors % 100 == 0:
                        print(f"[motor] send failed ({self.errors}): {e}")

    def close(self):
        """Stop the thread; after this returns no more commands will be sent by it."""
        with self._cv:
            self._stop = True
            self._cv.notify()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)


class LegoHardware:
    """Owns the three LEGO devices. Use connect_all() once, then poll from the main loop."""

    def __init__(self, *, motors_enabled=True, clock=time.monotonic):
        self.clock = clock
        self.dm = le.DoubleMotor()
        self.cs = le.ColorSensor()
        self.ct = le.Controller()
        self.devices = {"Double Motor": self.dm, "Color Sensor": self.cs, "Controller": self.ct}
        self._last_rx = {name: -math.inf for name in self.devices}
        self._color_q = deque(maxlen=500)
        self.dm.set_notification_callback(self._cb("Double Motor"))
        self.cs.set_notification_callback(self._color_cb)
        self.ct.set_notification_callback(self._cb("Controller"))
        self.motors_enabled = motors_enabled
        self.drive = None
        self._stop = threading.Event()
        self._reconnect_thread = None
        self.finished = False   # (the simulator uses this; real hardware never "finishes")

    # -- notification callbacks (LEGO event-loop thread: record only!) -------------
    def _cb(self, name):
        def cb(_notification):
            self._last_rx[name] = self.clock()
        return cb

    def _color_cb(self, _notification):
        t = self.clock()
        self._last_rx["Color Sensor"] = t
        # The library replaced cs.sensor with a fresh object just before calling us,
        # so keeping a reference to it is safe.
        self._color_q.append((t, self.cs.sensor))

    # -- connection -----------------------------------------------------------------
    def _connect_one(self, name, dev):
        dev.connect(card_color=config.CARD_COLOR, card_serial=config.CARD_SERIAL,
                    device_notification_delay=config.NOTIFICATION_MS)
        return dev.connected

    def connect_all(self):
        """Blocks until all three devices are connected (Ctrl+C to give up)."""
        for name, dev in self.devices.items():
            attempt = 0
            while not dev.connected:
                attempt += 1
                print(f"[lego] connecting to {name} (card {config.CARD_SERIAL}), attempt {attempt} ...")
                try:
                    if self._connect_one(name, dev):
                        break
                except Exception as e:
                    if type(e).__name__ == "VersionMismatchError":
                        fw = ".".join(map(str, getattr(dev, "version_firmware", None) or ())) or "?"
                        rpc = ".".join(map(str, getattr(dev, "version_rpc", None) or ())) or "?"
                        print(f"[lego] *** {name} is incompatible (firmware {fw}, RPC {rpc}). "
                              f"Update THIS device at https://code.legoeducation.com/ ***")
                        raise
                    print(f"[lego] {name}: {e}")
                print(f"[lego] {name} not found. Is it on, charged, and on card {config.CARD_SERIAL}?")
                time.sleep(config.BLE_RECONNECT_INTERVAL_S)
            print(f"[lego] {name} connected (firmware {'.'.join(map(str, dev.version_firmware))})")
        self.drive = MotorSender(self.dm) if self.motors_enabled else NullDrive()
        self._reconnect_thread = threading.Thread(target=self._reconnect_loop, name="LegoReconnect",
                                                  daemon=True)
        self._reconnect_thread.start()

    def _reconnect_loop(self):
        while not self._stop.wait(config.BLE_RECONNECT_INTERVAL_S):
            for name, dev in self.devices.items():
                if self._stop.is_set() or dev.connected:
                    continue
                print(f"[lego] {name} DISCONNECTED -> trying to reconnect")
                try:
                    if self._connect_one(name, dev):
                        print(f"[lego] {name} reconnected")
                except Exception as e:
                    print(f"[lego] {name} reconnect failed: {e}")

    # -- main-loop interface ------------------------------------------------------------
    def poll(self, now):
        pass   # real hardware updates itself via callbacks

    def drain_color(self):
        """All color readings since the last call, oldest first: [(t, sensor), ...]."""
        out = []
        while self._color_q:
            out.append(self._color_q.popleft())
        return out

    def color_fresh(self, now):
        return self.cs.connected and now - self._last_rx["Color Sensor"] <= config.COLOR_STALE_S

    def levers(self, now):
        """(left %, right %) if the Controller is connected and fresh, else None."""
        if not self.ct.connected or now - self._last_rx["Controller"] > config.CONTROLLER_STALE_S:
            return None
        s = self.ct.sensor
        if not (_finite(s.leftPercent) and _finite(s.rightPercent)):
            return None
        return float(s.leftPercent), float(s.rightPercent)

    def motion_sample(self, now):
        if not self.dm.connected or now - self._last_rx["Double Motor"] > config.MOTOR_STALE_S:
            return MotionSample(None, None, None)
        imu = self.dm.imu_device
        left, right = from_motor_frame(self.dm.motor[0], self.dm.motor[1])
        ls = left.speed if _finite(left.speed) else None
        rs = right.speed if _finite(right.speed) else None
        acc = (imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ)
        tilt = getattr(imu, config.IMU_TILT_FIELD) if config.IMU_TILT_FIELD else None
        return MotionSample(ls, rs, acc if all(_finite(a) for a in acc) else None,
                            t=self._last_rx["Double Motor"], tilt=tilt if _finite(tilt) else None)

    def motor_states(self):
        """The Double Motor's own state per side (READY, RUNNING, STALLED, ...), for the status line."""
        if not self.dm.connected:
            return "disconnected"
        names = {getattr(le, n): n.removeprefix("MOTOR_STATE_") for n in dir(le) if n.startswith("MOTOR_STATE_")}
        states = "/".join(str(names.get(self.dm.motor[i].motorState, self.dm.motor[i].motorState)) for i in range(2))
        info = self.dm.info_device
        usb = " USB-PLUGGED-IN" if info.UsbPowerState == le.USB_POWER_STATE_USB_CONNECTED else ""
        return f"{states} (battery {info.batteryLevel}%{usb})"

    def emergency_stop(self):
        """Stop the wheels NOW (used on every exit path).

        The sender thread is shut down first, so an older "drive" frame it was
        about to send can't arrive after our stop frames."""
        if isinstance(self.drive, MotorSender):
            self.drive.close()
        if self.drive is not None:
            self.drive.commanded = (0, 0)
        if not self.motors_enabled or not self.dm.connected:
            return
        for fn in (lambda: self.dm.movement_move_tank(0, 0, blocking=False),
                   lambda: self.dm.movement_stop(blocking=False)):
            try:
                fn()
            except Exception as e:
                print(f"[motor] emergency stop error: {e}")

    def close(self):
        self._stop.set()
        self.emergency_stop()
        time.sleep(0.1)   # let the stop frames leave before the link goes down
        for name, dev in self.devices.items():
            if dev.connected:
                try:
                    dev.disconnect()
                except Exception as e:
                    print(f"[lego] {name} disconnect error: {e}")
