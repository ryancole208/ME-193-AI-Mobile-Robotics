"""
Wii Sports-style bowling with the Double Motor (IMU throw) and webcam (position/aim).

    python bowling.py              # motor + webcam + MQTT (keyboard still works as backup)
    python bowling.py --keyboard   # no hardware: Space = throw, arrows = move, A/D = aim
    python bowling.py --no-mqtt    # don't touch the network

Keys: 1/2 Bumpers / No Bumpers, Enter start, Esc menu, Space throw, Left/Right move,
      A/D aim, Up/Down keyboard power, hold Z/C for a left/right hook, F1 debug, Q quit.
"""

import argparse
import os
import queue
import signal
import time

import config  # first: puts ../ (Midterm) on sys.path for the reused modules
from bowler_input import BowlerInput
from difficulty_selector import MENU, START, ManualSelector, difficulty
from throw_detector import ThrowEvent


class _Disabled:
    status = "disabled"
    connected = False
    tracking = False


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

    throws = queue.Queue()

    # -- background services ----------------------------------------------------
    motor = _Disabled()
    if use_motor:
        from throw_detector import BowlingMotor
        motor = BowlingMotor(on_throw=throws.put)
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
        mqtt.publish_score(0.0)

    bowler = BowlerInput(pose)
    selector = ManualSelector()
    sources = [selector]
    if config.APRILTAG_ENABLED:
        from apriltag_selector import AprilTagSelector
        sources.append(AprilTagSelector())

    from bowling_game import BowlingGame
    from bowling_render import Renderer
    game = BowlingGame(bowler,
                       on_score=mqtt.publish_score if use_mqtt else None,
                       on_impact=motor.request_haptic if use_motor else None)

    # -- pygame -----------------------------------------------------------------
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("Bowling")
    renderer = Renderer(screen)
    clock = pygame.time.Clock()
    debug = config.DEBUG_OVERLAY_DEFAULT

    running = True

    def on_sigint(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, on_sigint)

    difficulty_keys = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_KP1: 0, pygame.K_KP2: 1}
    names = list(config.DIFFICULTIES)

    try:
        while running:
            dt = clock.tick(config.FPS) / 1000.0
            now = time.monotonic()
            keys = pygame.key.get_pressed()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_q:
                        running = False
                    elif ev.key in difficulty_keys:
                        selector.push(difficulty(names[difficulty_keys[ev.key]]))
                    elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        selector.push(START)
                    elif ev.key == pygame.K_ESCAPE:
                        selector.push(MENU)
                    elif ev.key == pygame.K_F1:
                        debug = not debug
                    elif ev.key == pygame.K_UP:
                        bowler.change_strength(+1)
                    elif ev.key == pygame.K_DOWN:
                        bowler.change_strength(-1)
                    elif ev.key == pygame.K_SPACE and config.KEYBOARD_FALLBACK:
                        spin = config.KEYBOARD_SPIN * ((1 if keys[pygame.K_c] else 0) - (1 if keys[pygame.K_z] else 0))
                        throws.put(ThrowEvent(t_onset=now, t_release=now, strength=bowler.kb_strength,
                                              spin=spin, source="keyboard"))
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    for name, rect in renderer.buttons.items():
                        if rect.collidepoint(ev.pos):
                            if name == "start":
                                selector.push(START)
                            elif name == "menu":
                                selector.push(MENU)
                            else:
                                selector.push(difficulty(name.split(":", 1)[1]))

            for src in sources:
                for cmd in src.poll():
                    if cmd.kind == "difficulty":
                        game.set_difficulty(cmd.value)
                    elif cmd.kind == "start":
                        game.start(now)
                    elif cmd.kind == "menu":
                        game.to_menu()

            kb = config.KEYBOARD_FALLBACK
            bowler.update(now, dt, left=kb and keys[pygame.K_LEFT], right=kb and keys[pygame.K_RIGHT],
                          aim_left=keys[pygame.K_a], aim_right=keys[pygame.K_d])

            while True:
                try:
                    game.add_throw(throws.get_nowait(), now)
                except queue.Empty:
                    break
            game.update(now)

            snap = pose.snapshot() if pose else None
            statuses = [
                ("Motor", "motor", motor.status),
                ("MQTT", "mqtt", mqtt.status),
                ("Pose", "pose", pose.status if pose else "disabled"),
            ]
            debug_lines = build_debug_lines(now, game, motor, bowler, mqtt, clock) if debug else None
            hint = "1/2 bumpers   Enter start   Esc menu   Space throw   ←/→ move   A/D aim   ↑/↓ power   " \
                   "Z/C hook   F1 debug   Q quit"
            renderer.draw(now, game, statuses, snap, debug_lines, hint)
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
        for src in sources:
            src.stop()
        print("Bye.")


def build_debug_lines(now, game, motor, bowler, mqtt, clock):
    lines = [f"FPS {clock.get_fps():5.1f}   state={game.state}   input={bowler.source}"]
    det = getattr(motor, "detector", None)
    s = getattr(motor, "latest", None)
    if s is not None:
        lines.append(f"IMU raw a=({s.ax:.0f},{s.ay:.0f},{s.az:.0f}) g=({s.gx:.0f},{s.gy:.0f},{s.gz:.0f})")
    if det is not None:
        lines.append(f"dyn |a| {det.last_dyn_accel_g:5.2f} g  (thr {det.accel_threshold_g:.2f})  "
                     f"{'THROWING' if det.active else ''}")
        lines.append(f"|gyro|  {det.last_gyro:7.0f} raw (thr {det.gyro_threshold:.0f})  mode={det.mode}")
        ev = det.last_event
        if ev is not None:
            tw = ", ".join(f"{v:.0f}" for v in ev.twist_raw)
            lines.append(f"last throw: str {ev.strength:.2f}  dur {ev.t_release - ev.t_onset:.2f}s to release")
            lines.append(f"  twist (gx,gy,gz)=({tw})  spin {ev.spin:+.2f} [axis {config.SPIN_GYRO_AXIS}]")
    elif getattr(motor, "status", "disabled") != "disabled":
        lines.append("IMU: waiting for calibration")
    lines.append(f"stand {bowler.current_position():+.2f} m   preset aim {bowler.preset_aim_deg:+.1f}°")
    t = game.last_throw
    if t:
        lines.append(f"launch x={t['x']:+.2f} angle={t['angle_deg']:+.2f}° v={t['speed']:.1f} m/s "
                     f"spin={t['spin']:+.2f} ({t['source']})")
    if game.roll is not None:
        b = game.roll.ball
        lines.append(f"ball ({b.x:+.2f}, {b.y:5.2f}) m  gutter={b.in_gutter}  t={game.roll.t:.2f}s")
    if hasattr(mqtt, "published") and mqtt.published:
        topic, payload = mqtt.published[-1]
        lines.append(f"mqtt last: {topic} = {payload}")
    return lines


if __name__ == "__main__":
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    main()
