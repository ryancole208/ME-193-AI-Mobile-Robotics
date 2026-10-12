"""
Hardware bridge for the Godot rebuild: Python reads the paddle, Godot runs the game.
Packets are JSON over UDP on localhost; ports, rates and flags are in bridge_config.py.

    python bridge.py                           # motor + webcam + MQTT
    python bridge.py --no-pose                 # motor only (x stays 0.5)
    python bridge.py --no-motor                # webcam only (roll 0, face black)
    python bridge.py --no-pose --no-motor      # AprilTags only (the tag scanner opens the webcam)
    python bridge.py --no-hardware             # no motor, no webcam: status packets only
    add --no-mqtt / --no-tags / --no-display to any of them to leave the network / the AprilTags /
    the camera view window alone

The camera view window shows the mirrored webcam image with the pose overlay (as in the other
games) and every detected tag's outline and ID, whenever the webcam is in use.

Python -> Godot (GODOT_PORT):
    {"type":"state","x":0-1,"roll":<calmed deg>,"face":"black"|"yellow"}   STATE_RATE_HZ
    {"type":"swing","strength":<float>,"age_ms":<ms from the swing peak to the send>}   each swing
    {"type":"tag","id":<int>}   once when an allowed tag16h5 tag becomes stable (song selection)
    {"type":"tag","id":-1}      once when no stable tag has been seen for TAG_LOST_S
    {"type":"status","motor":<calibrated>,"pose":<person tracked>,"tag":<stable ID or -1>}
                                every STATUS_INTERVAL_S (the tag is repeated here, so Godot never misses it)

Godot -> Python (BRIDGE_PORT), each printed on one line; malformed packets are ignored:
    {"type":"zero"}     re-zero the roll (like C in rhythm_main); ignored until the motor is calibrated
    {"type":"haptic"}   HAPTIC_PULSE_MS pulse, if HAPTIC_ENABLED and the motor is calibrated
    {"type":"score","streak":<int>,"max_streak":<int>}   the record (highest max_streak since the
                        bridge started) is published to MQTT as a float whenever it goes up

Hold the paddle still with BLACK toward the screen while the motor calibrates: the grip at the
end of calibration becomes zero roll (again after a reconnect). MQTT publishes 0.0 at start.
Ctrl+C quits.

The pygame game (rhythm_main.py) is separate and unchanged.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # this folder's modules win
import bridge_config as bc  # noqa: E402
from udp_link import UdpReceiver, UdpSender, encode  # noqa: E402

clock = time.perf_counter   # == audio_engine.clock: RhythmMotor stamps its IMU samples with it


class _Disabled:
    status = "disabled"
    published = ()

    def publish_score(self, score):
        pass


def _safe_console():
    """Never let a print fail (the Double Motor's name starts with an emoji)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-hardware", action="store_true", help="no motor, no webcam: status packets only")
    p.add_argument("--no-motor", action="store_true", help="don't connect to the Double Motor")
    p.add_argument("--no-pose", action="store_true", help="don't open the webcam")
    p.add_argument("--no-mqtt", action="store_true", help="don't touch the network")
    p.add_argument("--no-tags", action="store_true", help="no AprilTag song selection")
    p.add_argument("--no-display", action="store_true", help="no camera view window")
    return p.parse_args(argv)


def start_mqtt():
    import config  # noqa: F401  rhythm/config.py: puts ../ on sys.path for mqtt_client
    from mqtt_client import ScorePublisher       # ../mqtt_client.py (uses ../../mqttlib.py)
    mqtt = ScorePublisher(topic=bc.MQTT_TOPIC)   # not ../config.py's topic: that one is shared by the other games
    mqtt.start()
    mqtt.publish_score(0.0)                      # sent as soon as the broker connection is up
    print(f"[mqtt] publishing the hit-streak record to {mqtt.topic}")
    return mqtt


def face_for(raw_roll_deg):
    """Side physically toward the screen: black in the neutral grip, yellow once twisted past FACE_FLIP_DEG."""
    if raw_roll_deg is None:
        return "black"
    return "black" if abs(raw_roll_deg) <= bc.FACE_FLIP_DEG else "yellow"


def x_for(paddle_x_m, table_width_m):
    return min(1.0, max(0.0, paddle_x_m / table_width_m + 0.5))


COMMANDS = ("zero", "haptic", "score")


def _count(v):
    """A non-negative whole number (Godot may send 12 or 12.0), else None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float) and not v.is_integer():
        return None
    return int(v) if v >= 0 else None


def parse_command(data):
    """One UDP packet from Godot -> (msg dict, None) or (None, reason it was ignored)."""
    try:
        msg = json.loads(data.decode("utf-8"))
    except UnicodeDecodeError:
        return None, "not UTF-8"
    except (ValueError, RecursionError):
        return None, "not JSON"
    if not isinstance(msg, dict):
        return None, "not a JSON object"
    kind = msg.get("type")
    if kind not in COMMANDS:
        return None, f"unknown type {kind!r}"
    if kind == "score":
        for key in ("streak", "max_streak"):
            n = _count(msg.get(key))
            if n is None:
                return None, f"score needs a whole number >= 0 for {key!r}"
            msg[key] = n
    return msg, None


class Bridge:
    """Turns the motor / pose threads into packets. motor and pose may be None (disabled)."""

    def __init__(self, motor=None, pose=None, *, godot=None, swing_link=None, publish_score=None,
                 tags=None, tag_link=None):
        import config
        self.config = config
        self.motor = motor
        self.pose = pose
        self.godot = godot or UdpSender(bc.GODOT_HOST, bc.GODOT_PORT)
        # Swings are sent from the motor's BLE thread: their own socket, never shared across threads.
        self.swing_link = swing_link or UdpSender(bc.GODOT_HOST, bc.GODOT_PORT)
        self.tags = tags                  # TagScanner or None
        self.tag_link = tag_link or UdpSender(bc.GODOT_HOST, bc.GODOT_PORT)   # the scanner thread's own socket
        self.roll_display = None
        if motor is not None:
            from rhythm_motor import RollDisplay
            self.roll_display = RollDisplay()
        self.last_x = 0.5                 # held while the wrist is lost (status.pose says so)
        self.last_swing_t = None          # peak time of the latest swing (clock())
        self._zeroed_roll = None          # the RollFilter we last zeroed (a reconnect makes a new one)
        self.last_state = None
        self.publish_score = publish_score or (lambda score: None)
        self.record = 0                   # highest max_streak since the bridge started (MQTT has 0.0)

    # -- motor thread -----------------------------------------------------------
    def on_swing(self, ev):
        """RhythmSwing from the BLE thread: send at once, age = peak sample -> now."""
        self.last_swing_t = ev.t
        age_ms = (clock() - ev.t) * 1000.0
        data = self.swing_link.send({"type": "swing", "strength": round(ev.strength, 3),
                                     "age_ms": round(age_ms, 1)})
        if data is not None and bc.PRINT_SENDS:
            print(f"[swing] {data.decode('utf-8')}")

    # -- tag scanner thread -----------------------------------------------------
    def on_tag(self, tag_id, tag_filter=None):
        """A tag event from TagFilter: send it once, print one line."""
        self.tag_link.send({"type": "tag", "id": tag_id})
        if tag_id == -1:
            print(f"[tag] lost (no stable tag for {bc.TAG_LOST_S:g} s) -> sent id -1")
        else:
            side = getattr(tag_filter, "last_side_px", None)
            print(f"[tag] stable {tag_id}" + (f" (side {side:.0f} px)" if side else "") + f" -> sent id {tag_id}")

    # -- main thread ------------------------------------------------------------
    def zero_after_calibration(self):
        """calibrate_neutral() once per RollFilter, as soon as it has seen a sample (before that
        it would do nothing). The motor creates a new RollFilter after every (re)calibration."""
        m = self.motor
        r = getattr(m, "roll", None) if m is not None else None
        if r is None or r is self._zeroed_roll or r.angle is None:
            return False
        m.calibrate_neutral()
        self.roll_display.reset()
        self._zeroed_roll = r
        print("[motor] roll zeroed: this grip (BLACK toward the screen) is neutral")
        return True

    # -- commands from Godot (main thread) ---------------------------------------
    def handle(self, msg):
        """Act on one parsed command; returns what happened, for the console."""
        return getattr(self, f"_on_{msg['type']}")(msg)

    def _on_zero(self, msg):
        m = self.motor
        r = getattr(m, "roll", None) if m is not None else None
        if r is None or r.angle is None:
            return "ignored (motor not calibrated)"
        m.calibrate_neutral()               # same as C in rhythm_main
        self.roll_display.reset()
        self._zeroed_roll = r
        return "roll zeroed"

    def _on_haptic(self, msg):
        if not bc.HAPTIC_ENABLED:
            return "ignored (HAPTIC_ENABLED is False)"
        if not self.motor_ready():
            return "ignored (motor not calibrated)"
        self.motor.pulse(bc.HAPTIC_PULSE_MS)   # only sets a flag: the motor thread sends the BLE command
        return f"pulse {bc.HAPTIC_PULSE_MS} ms"

    def _on_score(self, msg):
        best = msg["max_streak"]
        if best <= self.record:
            return f"record {self.record}"
        self.record = best
        self.publish_score(float(best))
        return f"new record {best} -> MQTT {float(best)}"

    def motor_ready(self):
        m = self.motor
        return bool(m is not None and m.connected and m.detector is not None)

    def pose_ready(self):
        return bool(self.pose is not None and self.pose.tracking)

    def _swinging(self, now):
        m = self.motor
        det = getattr(m, "detector", None)
        recent = (self.last_swing_t is not None
                  and now - self.last_swing_t < self.config.ROLL_DISPLAY_SWING_HOLD_S)
        return (det is not None and det._active is not None) or getattr(m, "quiet", False) or recent

    def state(self, now):
        if self.pose is not None:
            px = self.pose.snapshot().paddle_x
            if px is not None:
                self.last_x = x_for(px, self.config.TABLE_WIDTH_M)
        roll, face = 0.0, "black"
        if self.motor is not None:
            raw = self.motor.roll_deg
            roll = self.roll_display.update(now, raw, self._swinging(now))
            face = face_for(raw)
        self.last_state = {"type": "state", "x": round(self.last_x, 4), "roll": round(roll, 2), "face": face}
        return self.last_state

    def status(self):
        tag = self.tags.filter.status_id if self.tags is not None else -1
        return {"type": "status", "motor": self.motor_ready(), "pose": self.pose_ready(), "tag": tag}

    def close(self):
        self.godot.close()
        self.swing_link.close()
        self.tag_link.close()


def main(argv=None):
    _safe_console()
    args = parse_args(argv)
    hardware = not args.no_hardware
    use_motor = hardware and bc.MOTOR_ENABLED and not args.no_motor
    use_pose = hardware and bc.POSE_ENABLED and not args.no_pose
    use_mqtt = bc.MQTT_ENABLED and not args.no_mqtt
    use_tags = hardware and bc.TAGS_ENABLED and not args.no_tags
    use_view = (use_pose or use_tags) and bc.DISPLAY_ENABLED and not args.no_display

    try:                                  # first, so a second bridge stops before touching any hardware
        commands = UdpReceiver(bc.BRIDGE_HOST, bc.BRIDGE_PORT)
    except OSError as e:
        sys.exit(f"Can't listen on {bc.BRIDGE_HOST}:{bc.BRIDGE_PORT} ({e}). Is another bridge.py running?")

    mqtt = start_mqtt() if use_mqtt else _Disabled()
    import config  # rhythm/config.py (puts ../ on sys.path for motor_imu / pose_tracker)
    config.SHOW_CAMERA_PREVIEW = bc.POSE_PREVIEW

    bridge = None
    motor = pose = tags = view = None
    if use_motor:
        from rhythm_motor import RhythmMotor
        motor = RhythmMotor(on_swing=lambda ev: bridge.on_swing(ev))
    if use_tags:
        from tags import TagScanner
        # Shares PoseTracker's camera; with --no-pose nobody else has it open, so the scanner opens it.
        tags = TagScanner(on_event=lambda tag_id, f: bridge.on_tag(tag_id, f), own_camera=not use_pose)
    if use_view:
        from camera_view import CameraView
        view = CameraView(tags=tags)
        if tags and tags.own_camera:
            tags.frame_listener = view.on_frame
    if use_pose:
        from pose_tracker import PoseTracker
        readers = [r.on_frame for r in (tags, view) if r is not None]   # everyone sharing the webcam

        def on_frame(t, frame):
            for read in readers:
                read(t, frame)

        pose = PoseTracker(on_frame=on_frame if readers else None)
        if view:
            view.pose = pose
    # Built before any thread starts, so on_swing / on_tag can't miss it.
    bridge = Bridge(motor, pose, publish_score=mqtt.publish_score, tags=tags)
    if motor:
        motor.start()
        print("[motor] scanning -- then hold the paddle still, BLACK toward the screen, to calibrate")
    if tags:
        tags.start()
        print(f"[tag] scanning tag16h5 IDs {sorted(bc.ALLOWED_TAG_IDS)} at {bc.TAG_RATE_HZ:g} Hz "
              f"({'own camera' if tags.own_camera else 'pose camera frames'})")
    if pose:
        pose.start()
    if view:
        view.start()
        print("[view] camera view window open (Esc / Q in it, or closing it, hides only the view)")

    print(f"[bridge] sending to Godot at {bc.GODOT_HOST}:{bc.GODOT_PORT}  "
          f"(motor {'on' if motor else 'off'}, pose {'on' if pose else 'off'}, tags {'on' if tags else 'off'}, "
          f"state {'%d Hz' % bc.STATE_RATE_HZ if hardware else 'off'}); "
          f"listening on {bc.BRIDGE_HOST}:{bc.BRIDGE_PORT}  Ctrl+C to quit")
    state_dt = 1.0 / bc.STATE_RATE_HZ
    t0 = clock()
    next_state, next_status, next_print = t0, t0, t0 + bc.PRINT_STATE_INTERVAL_S
    states_sent, print_t0 = 0, t0
    try:
        while True:
            for data in commands.drain():
                msg, why = parse_command(data)
                text = data[:200].decode("utf-8", errors="replace")
                print(f"[recv] {text} -> {bridge.handle(msg) if msg else 'ignored: ' + why}")
            now = clock()
            if motor:
                bridge.zero_after_calibration()
            if hardware and now >= next_state:
                if bridge.godot.send(bridge.state(now)) is not None:
                    states_sent += 1
                next_state += state_dt
                if next_state < now:                 # fell behind: skip, don't burst
                    next_state = now + state_dt
            if now >= next_status:
                data = bridge.godot.send(bridge.status())
                if data is not None and bc.PRINT_SENDS:
                    print(f"[status] {data.decode('utf-8')}   motor: {motor.status if motor else 'off'} | "
                          f"pose: {pose.status if pose else 'off'} | tags: {tags.status if tags else 'off'} | "
                          f"view: {view.status if view else 'off'} | "
                          f"mqtt: {mqtt.status}")
                next_status += bc.STATUS_INTERVAL_S
                if next_status < now:
                    next_status = now + bc.STATUS_INTERVAL_S
            if hardware and now >= next_print:
                rate = states_sent / max(1e-6, now - print_t0)
                print(f"[state]  {encode(bridge.last_state).decode('utf-8')}   ({rate:.1f} Hz)")
                states_sent, print_t0 = 0, now
                next_print = now + bc.PRINT_STATE_INTERVAL_S
            wake = min(next_state, next_status) if hardware else next_status
            # Sleeps (so the BLE / camera threads run) until the next send -- or a command arrives.
            commands.wait(min(0.05, wake - clock()))
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down...")
        if pose:
            pose.stop()
        if tags:
            tags.stop()
        if view:
            view.stop()
        if motor:
            motor.stop()
        bridge.close()
        commands.close()
        if use_mqtt:
            mqtt.stop()
        print("Bye.")


if __name__ == "__main__":
    main()
