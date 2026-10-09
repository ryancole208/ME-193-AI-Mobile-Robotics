"""
Wii Sports-style Halloween minigolf with the Double Motor (twist = putter angle, quick
swing = putt) and webcam (push/pull).

    python golf.py              # motor + webcam + MQTT (keyboard still works as backup)
    python golf.py --keyboard   # no hardware: hold/release Space = putt, ←/→ = aim
    python golf.py --no-mqtt    # don't touch the network

Keys: 1/2 Easy / Hard, Enter start (or skip the hole flyover), Esc menu, ←/→ or A/D aim
      (hold Shift = fine), C recenter the motor twist, hold Space to charge / release to
      putt, hold V overhead view, F1 debug, Q quit.
"""

import argparse
import os
import queue
import signal
import time

import config  # first: puts ../ (Midterm) on sys.path for the reused modules
from difficulty_selector import MENU, START, ManualSelector, difficulty
from golfer_input import GolferInput
from putt_detector import PuttEvent


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

    putts = queue.Queue()

    # -- background services ----------------------------------------------------
    motor = _Disabled()
    if use_motor:
        from putt_detector import GolfMotor
        motor = GolfMotor(on_putt=putts.put)
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

    golfer = GolferInput(pose, twist=motor if use_motor else None)
    selector = ManualSelector()
    sources = [selector]
    if config.APRILTAG_ENABLED:
        from apriltag_selector import AprilTagSelector
        sources.append(AprilTagSelector())

    from golf_game import GolfGame
    from golf_render import Renderer
    game = GolfGame(golfer,
                    on_score=mqtt.publish_score if use_mqtt else None,
                    on_holed=motor.request_haptic if use_motor else None)

    # -- pygame -----------------------------------------------------------------
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("Spooky Minigolf")
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
                    elif ev.key == pygame.K_c:
                        golfer.recenter()
                    elif ev.key == pygame.K_SPACE and config.KEYBOARD_FALLBACK and game.state == "aiming":
                        golfer.begin_charge(now)
                elif ev.type == pygame.KEYUP and ev.key == pygame.K_SPACE:
                    t0 = golfer.charge_t
                    strength = golfer.release_charge(now)
                    if strength is not None:
                        putts.put(PuttEvent(t_onset=t0, t_impact=now, strength=strength, source="keyboard"))
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
            if game.state != "aiming":
                golfer.charge_t = None

            shift = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            golfer.update(now, dt, left=keys[pygame.K_LEFT] or keys[pygame.K_a],
                          right=keys[pygame.K_RIGHT] or keys[pygame.K_d], fine=shift,
                          frozen=game.state != "aiming" or golfer.charge_t is not None)

            while True:
                try:
                    game.add_putt(putts.get_nowait(), now)
                except queue.Empty:
                    break
            game.update(now)

            snap = pose.snapshot() if pose else None
            statuses = [
                ("Motor", "motor", motor.status),
                ("MQTT", "mqtt", mqtt.status),
                ("Pose", "pose", pose.status if pose else "disabled"),
            ]
            det = getattr(motor, "detector", None)
            live = det.live_strength if det is not None and game.state == "aiming" else 0.0
            debug_lines = build_debug_lines(game, motor, golfer, mqtt, clock) if debug else None
            hint = "1/2 difficulty   Enter start/skip   Esc menu   twist / ←/→ aim   C recenter   " \
                   "swing or hold Space putt   V overview   F1 debug   Q quit"
            renderer.draw(now, game, statuses, snap, debug_lines, hint, overview=keys[pygame.K_v],
                          live_strength=live)
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


def build_debug_lines(game, motor, golfer, mqtt, clock):
    lines = [f"FPS {clock.get_fps():5.1f}   state={game.state}   input={golfer.source}"]
    det = getattr(motor, "detector", None)
    s = getattr(motor, "latest", None)
    if s is not None:
        lines.append(f"IMU raw a=({s.ax:.0f},{s.ay:.0f},{s.az:.0f}) g=({s.gx:.0f},{s.gy:.0f},{s.gz:.0f})")
    if det is not None:
        lines.append(f"dyn |a| {det.last_dyn_accel_g:5.2f} g  (thr {det.accel_threshold_g:.2f})  "
                     f"{'PUTTING' if det.active else ''}")
        lines.append(f"swing  {det.last_gyro:7.0f} raw (thr {det.gyro_threshold:.0f})  mode={det.mode}")
        lines.append(f"twist  {det.last_twist_rate:+7.0f} raw/s  angle {det.twist_deg:+6.1f}°  "
                     f"[{config.TWIST_AXIS}]")
        ev = det.last_event
        if ev is not None:
            lines.append(f"last putt: str {ev.strength:.2f}  peak {ev.t_impact - ev.t_onset:.2f}s after firing")
    elif getattr(motor, "status", "disabled") != "disabled":
        lines.append("IMU: waiting for calibration")
    lines.append(f"aim {golfer.aim_deg:+.1f}° = base {golfer.base_aim_deg:+.1f} + twist {golfer.twist_offset_deg:+.1f}"
                 f"   wrist stick {golfer.stick:+.2f}")
    t = game.last_putt
    if t:
        lines.append(f"putt heading={t['heading_deg']:+.1f}° (aim {t['aim_deg']:+.1f} push {t['push_deg']:+.1f}) "
                     f"v={t['speed']:.2f} m/s ({t['source']})")
    if game.sim is not None:
        b = game.sim.ball
        lines.append(f"ball ({b.x:+.2f}, {b.y:+.2f}) m  v={b.speed:.2f}  walls={game.sim.wall_hits}  "
                     f"{game.sim.outcome or ''}")
    if game.hole:
        lines.append(f"hole {game.hole.number}: {len(game.hole.path)} cells, {game.hole.turns} turns, "
                     f"{len(game.hole.obstacles)} obstacles")
    if hasattr(mqtt, "published") and mqtt.published:
        topic, payload = mqtt.published[-1]
        lines.append(f"mqtt last: {topic} = {payload}")
    return lines


if __name__ == "__main__":
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    main()
