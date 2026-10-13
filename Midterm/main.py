"""
Human side of Human-vs-Robot ping pong.

    python main.py              # motor + webcam + MQTT (keyboard still works as backup)
    python main.py --keyboard   # no hardware: space = swing, arrows = paddle
    python main.py --no-mqtt    # don't touch the network

Keys: 1/2/3/4 difficulty (4 = Dynamic, the learning opponent), Enter start, Esc menu,
      Space swing, Left/Right paddle, hold Q/E (curve left/right) or W/S (topspin/backspin)
      while pressing Space, C re-zero the handle twist, R (in the menu) reset the Dynamic
      opponent's learning, H show/hide the controls bar, F1 debug overlay,
      Ctrl+Q / window close quit.
"""

import argparse
import os
import queue
import signal
import time

import config
from difficulty_selector import MENU, START, ManualSelector, difficulty
from events import SwingEvent
from opponent import ModeOpponent, SimulatedOpponent
from player_input import KeyboardPaddle, PlayerInput

CONTROLS = ["1-4 difficulty (4 = Dynamic AI)", "Enter start", "Esc menu", "Space swing",
            "+Q/E curve", "+W/S top/back", "←/→ paddle", "C zero twist", "R reset AI (menu)",
            "F1 debug", "H hide controls", "Ctrl+Q quit"]


class _Disabled:
    status = "disabled"
    connected = False
    tracking = False
    roll_deg = None

    def calibrate_neutral(self):
        pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keyboard", action="store_true", help="keyboard only: no motor, no camera")
    p.add_argument("--no-motor", action="store_true")
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--no-mqtt", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    use_motor = config.MOTOR_ENABLED and not (args.keyboard or args.no_motor)
    use_camera = config.CAMERA_ENABLED and not (args.keyboard or args.no_camera)
    use_mqtt = config.MQTT_ENABLED and not args.no_mqtt

    swings = queue.Queue()

    # -- background services ----------------------------------------------------
    motor = _Disabled()
    if use_motor:
        from motor_imu import MotorIMU
        motor = MotorIMU(on_swing=swings.put)
        motor.start()

    pose = None
    if use_camera:
        from pose_tracker import PoseTracker
        pose = PoseTracker()
        pose.start()

    mqtt = _Disabled()
    if use_mqtt:
        from mqtt_client import ScorePublisher
        mqtt = ScorePublisher()
        mqtt.start()
        mqtt.publish_score(0.0)  # sent as soon as the broker connection is up

    keyboard = KeyboardPaddle()
    player = PlayerInput(pose=pose, keyboard=keyboard)
    from rl_opponent import RLOpponent
    dynamic = RLOpponent(paddle_x=player.current_paddle_x)
    opponent = ModeOpponent(SimulatedOpponent(), {config.DYNAMIC_DIFFICULTY: dynamic})
    opponent.start()
    selector = ManualSelector()
    sources = [selector]
    if config.APRILTAG_ENABLED:
        from apriltag_selector import AprilTagSelector
        sources.append(AprilTagSelector())

    from game import GameLogic
    from renderer import PlayerView, Renderer
    logic = GameLogic(opponent, player,
                      on_score=mqtt.publish_score if use_mqtt else None,
                      on_hit=motor.request_haptic if use_motor else None)

    # -- pygame -----------------------------------------------------------------
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("Ping Pong - Human side")
    renderer = Renderer(screen)
    clock = pygame.time.Clock()
    debug = config.DEBUG_OVERLAY_DEFAULT
    show_controls = config.CONTROLS_BAR_DEFAULT

    running = True

    def on_sigint(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, on_sigint)

    difficulty_keys = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3,
                       pygame.K_KP1: 0, pygame.K_KP2: 1, pygame.K_KP3: 2, pygame.K_KP4: 3}
    names = list(config.DIFFICULTIES)
    last_swing = None  # (time registered, direction) for the paddle swing animation

    try:
        while running:
            dt = clock.tick(config.FPS) / 1000.0
            now = time.monotonic()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_q and ev.mod & pygame.KMOD_CTRL:
                        running = False
                    elif ev.key == pygame.K_c:
                        motor.calibrate_neutral()
                    elif ev.key in difficulty_keys and difficulty_keys[ev.key] < len(names):
                        selector.push(difficulty(names[difficulty_keys[ev.key]]))
                    elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        selector.push(START)
                    elif ev.key == pygame.K_ESCAPE:
                        selector.push(MENU)
                    elif ev.key == pygame.K_F1:
                        debug = not debug
                    elif ev.key == pygame.K_h:
                        show_controls = not show_controls
                    elif (ev.key == pygame.K_r and logic.state == "menu"
                          and logic.difficulty == config.DYNAMIC_DIFFICULTY):
                        opponent.reset_learning()   # menu only, so it can't happen mid-rally
                    elif ev.key == pygame.K_SPACE and config.KEYBOARD_FALLBACK:
                        side, top = keyboard_spin(pygame.key.get_pressed(), pygame)
                        swings.put(SwingEvent(t=now, strength=config.KEYBOARD_SWING_STRENGTH, source="keyboard",
                                              spin_side=side, spin_top=top))
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    for name, rect in renderer.buttons.items():
                        if rect.collidepoint(ev.pos):
                            selector.push(START if name == "start" else difficulty(name.split(":", 1)[1]))

            for src in sources:
                for cmd in src.poll():
                    if cmd.kind == "difficulty":
                        logic.set_difficulty(cmd.value)
                    elif cmd.kind == "start":
                        if logic.state == "menu":
                            motor.calibrate_neutral()   # current grip = neutral (BLACK face toward the screen)
                        logic.start(now)
                    elif cmd.kind == "menu":
                        logic.to_menu()
                        opponent.checkpoint()          # saves the Dynamic opponent's learning

            keys = pygame.key.get_pressed()
            if config.KEYBOARD_FALLBACK:
                keyboard.update(now, dt, keys[pygame.K_LEFT], keys[pygame.K_RIGHT])

            while True:
                try:
                    swing = swings.get_nowait()
                except queue.Empty:
                    break
                logic.add_swing(swing)
                last_swing = (now, player.direction_at(swing.t))
            logic.update(now)

            snap = pose.snapshot() if pose else None
            statuses = [
                ("Motor", "motor", motor.status),
                ("MQTT", "mqtt", mqtt.status),
                ("Pose", "pose", pose.status if pose else "disabled"),
            ]
            debug_lines = build_debug_lines(now, logic, motor, pose, player, mqtt, clock) if debug else None
            hint = CONTROLS if show_controls else ["H show controls"]
            view = PlayerView(x=player.current_paddle_x(), roll_deg=paddle_roll(motor, keys, pygame),
                              swing_t=last_swing[0] if last_swing else None,
                              direction=last_swing[1] if last_swing else "center")
            renderer.draw(now, logic, view, statuses, snap, debug_lines, hint)
            pygame.display.flip()
    finally:
        print("Shutting down...")
        pygame.quit()
        if pose:
            pose.stop()
        if use_motor:
            motor.stop()
        if use_mqtt:
            mqtt.stop()
        opponent.stop()
        for src in sources:
            src.stop()
        print("Bye.")


def keyboard_spin(keys, pygame):
    """(side, top) from the spin keys held while Space is pressed, or (None, None)."""
    k = config.KEYBOARD_SPIN_AMOUNT
    side = (k if keys[pygame.K_e] else 0.0) - (k if keys[pygame.K_q] else 0.0)
    top = (k if keys[pygame.K_w] else 0.0) - (k if keys[pygame.K_s] else 0.0)
    if side == 0.0 and top == 0.0:
        return None, None
    return side, top


def paddle_roll(motor, keys, pygame):
    """On-screen paddle twist: the IMU roll plus a fixed twist while Q/E are held."""
    roll = getattr(motor, "roll_deg", None) or 0.0
    if config.KEYBOARD_FALLBACK:
        d = config.KEYBOARD_ROLL_DISPLAY_DEG
        roll += (d if keys[pygame.K_e] else 0.0) - (d if keys[pygame.K_q] else 0.0)
    return roll


def build_debug_lines(now, logic, motor, pose, player, mqtt, clock):
    lines = [f"FPS {clock.get_fps():5.1f}   state={logic.state}   input={player.source}"]
    det = getattr(motor, "detector", None)
    s = getattr(motor, "latest", None)
    if s is not None:
        lines.append(f"IMU raw a=({s.ax:.0f},{s.ay:.0f},{s.az:.0f}) g=({s.gx:.0f},{s.gy:.0f},{s.gz:.0f})")
    if det is not None:
        lines.append(f"dyn |a| {det.last_dyn_accel_g:5.2f} g  (thr {det.accel_threshold_g:.2f}, 1g={det.raw_per_g:.0f} raw)")
        lines.append(f"|gyro|  {det.last_gyro:7.0f} raw (thr {det.gyro_threshold:.0f})  mode={det.mode}")
    elif getattr(motor, "status", "disabled") != "disabled":
        lines.append("IMU: waiting for calibration")
    roll = getattr(motor, "roll", None)
    if roll is not None:
        raw = "  n/a " if roll.accel_roll_deg is None else f"{roll.accel_roll_deg:+6.1f}"
        flags = ("still" if roll.still else "moving") + (" QUIET(haptic)" if roll.frozen else "")
        lines.append(f"roll filt {roll.roll_deg:+6.1f}  raw(accel) {raw}  rate {roll.rate_dps:+6.0f} dps  {flags}")
    if pose is not None:
        snap = pose.snapshot()
        wx = "--" if snap.wrist_x is None else f"{snap.wrist_x:.3f}"
        px = "--" if snap.paddle_x is None else f"{snap.paddle_x:+.2f} m"
        lines.append(f"wrist x {wx}  paddle {px}  dir {pose.direction_at(now)}  cam {pose.fps:.0f} fps")
    lines.append(f"paddle used {player.current_paddle_x():+.2f} m   tol ±{config.LATERAL_HIT_TOLERANCE_M:.2f} m")
    tta = logic.time_to_arrival(now)
    if tta is not None:
        lines.append(f"ball arrives in {tta:+.2f}s  x={logic.flight.end_x:+.2f} m")
    r = logic.last_result
    if r:
        if r.get("timing_s") is None:
            lines.append(f"last: MISS {r.get('reason')}")
        else:
            verdict = "HIT" if r.get("hit") else "MISS"
            lines.append(f"last: {verdict} dt={r['timing_s']:+.2f}s err={r['lateral_err_m']:+.2f}m "
                         f"str={r['strength']:.2f} {r.get('direction', '')}")
            if r.get("hit"):
                lines.append(f"spin side {r['spin_side']:+.2f} top {r['spin_top']:+.2f}  curve {r['curve_m']:+.2f} m")
    if hasattr(mqtt, "published") and mqtt.published:
        topic, payload = mqtt.published[-1]
        lines.append(f"mqtt last: {topic} = {payload}")
    return lines


if __name__ == "__main__":
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    main()
