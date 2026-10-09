"""
Concert Rally -- rhythm ping pong. The ball arrives on the beat; swing on time.

Modes (M on song select, or --mode): aim = timing + your paddle at the ball (default),
timing = timing only, follow = the ball always comes to your paddle (timing only).

    python rhythm_main.py               # motor + webcam + audio (keyboard always works as backup)
    python rhythm_main.py --keyboard    # no hardware: Space = swing, hold Left/Right = move the paddle
    python rhythm_main.py --no-camera   # motor swings, arrow keys move the paddle
    python rhythm_main.py --list-songs  # show the playable songs
    python rhythm_main.py --keyboard --song polka   # skip the menus, play a song straight away
    python rhythm_main.py --mode follow             # start in another mode

No MQTT: this game never imports or connects to it (see scoring.py for the future hook).

Keys
  menus:   Up/Down choose, Enter select, Esc back, C calibrate timing, V calibrate visuals,
           M game mode (song select), R rescan songs (song select) / retry (results)
  play:    swing (or Space), hold Left/Right to move the paddle (as in ping pong), C re-zero twist,
           Esc leave the song
  always:  H show/hide the controls bar, F1 debug overlay, Ctrl+Q / window close quit
"""

import argparse
import os
import queue
import signal
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # this folder's config.py wins
import config  # noqa: E402  (also puts ../ on sys.path for the shared ping-pong modules)
from audio_engine import clock, make_engine  # noqa: E402
from player_input import KeyboardPaddle, PlayerInput  # noqa: E402

CONTROLS = {
    "title": ["Up/Down choose", "Enter select", "C calibrate timing", "V calibrate visuals",
              "H hide controls", "Ctrl+Q quit"],
    "select": ["Up/Down song", "Enter play", "M game mode", "C calibrate timing", "V calibrate visuals",
               "R rescan songs", "Esc back", "H hide controls", "Ctrl+Q quit"],
    "play": ["Swing the paddle (or Space) on the beat", "Move: your arm / hold Left Right",
             "C zero twist", "Esc leave song", "F1 debug", "H hide controls", "Ctrl+Q quit"],
    "calib_input": ["Swing (or Space) on every click", "Enter save", "R retry", "Esc cancel", "H hide controls"],
    "calib_av": ["Left/Right adjust (Shift = 1 ms)", "Enter save", "Esc cancel", "H hide controls"],
    "results": ["Enter song select", "R retry", "Esc title", "C calibrate timing", "H hide controls",
                "Ctrl+Q quit"],
}


class FrameClock:
    """Frame limiter that SLEEPS (releasing the GIL) instead of spinning.

    pygame's tick_busy_loop spins in C while holding the GIL, which starved the motor's BLE
    thread: the scan never finished and the motor stayed on "scanning". time.sleep is
    accurate to ~1 ms on Windows (Python 3.11+ uses a high-resolution timer) and lets the
    BLE / camera threads run. Same tick() / get_fps() interface as pygame.time.Clock."""

    def __init__(self):
        self._last = time.perf_counter()
        self._frames = deque(maxlen=30)

    def tick(self, fps):
        if fps > 0:
            wait = self._last + 1.0 / fps - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        self._frames.append(dt)
        return dt * 1000.0

    def get_fps(self):
        total = sum(self._frames)
        return len(self._frames) / total if total > 0 else 0.0


def _safe_console():
    """Never let a print fail: the Double Motor's name starts with an emoji (e.g. "⬜️ Double
    Motor"), which crashes print() in a non-UTF-8 console and so fails every connect attempt."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


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
    p.add_argument("--audio", choices=["sounddevice", "pygame", "null"], default=None,
                   help="override AUDIO_BACKEND")
    p.add_argument("--song", default=None,
                   help="skip the menus and play this song (title or file name, partial is fine)")
    p.add_argument("--mode", choices=list(config.GAME_MODES), default=None,
                   help="game mode: aim (timing + paddle at the ball), timing, follow (overrides GAME_MODE)")
    p.add_argument("--list-songs", action="store_true", help="list the playable songs and exit")
    return p.parse_args(argv)


def list_songs():
    from chart import scan_songs
    entries, _skipped = scan_songs()          # skipped charts are reported as they're found
    for e in entries:
        ch = e.chart
        print(f"  {ch.title:<22} {ch.author:<30} {ch.bpm_label():>8} BPM  {len(ch.cues):>3} cues   "
              f"--song \"{ch.song_id}\"")


def check_config():
    """Stop with a clear message if HIT_BEATS or GAME_MODE is invalid."""
    from chart import ChartError, check_hit_beats
    from rhythm_game import check_mode
    try:
        check_hit_beats()
        check_mode(config.GAME_MODE)
    except (ChartError, ValueError) as e:
        sys.exit(f"config.py: {e}")


def main(argv=None):
    _safe_console()
    args = parse_args(argv)
    if args.mode:
        config.GAME_MODE = args.mode
    check_config()
    if args.list_songs:
        list_songs()
        return
    use_motor = config.MOTOR_ENABLED and not (args.keyboard or args.no_motor)
    use_camera = config.CAMERA_ENABLED and not (args.keyboard or args.no_camera)

    swings = queue.Queue()
    motor = _Disabled()
    if use_motor:
        from rhythm_motor import RhythmMotor
        motor = RhythmMotor(on_swing=swings.put)
        motor.start()
    pose = None
    if use_camera:
        from pose_tracker import PoseTracker
        pose = PoseTracker()
        pose.start()

    import pygame
    pygame.display.init()          # NOT pygame.init(): the audio device belongs to our engine
    pygame.font.init()
    engine = make_engine(args.audio)
    print(f"[audio] {engine.name}: {engine.status}")

    keyboard = KeyboardPaddle()        # the ping-pong arrow-key paddle: hold Left/Right to glide
    player = PlayerInput(pose=pose, keyboard=keyboard)
    from rhythm_game import RhythmGame
    from rhythm_render import PlayerView, RhythmRenderer
    game = RhythmGame(engine, player, motor=motor if use_motor else None)

    flags = pygame.RESIZABLE if config.WINDOW_RESIZABLE else 0
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), flags)
    pygame.display.set_caption(config.WINDOW_TITLE)
    renderer = RhythmRenderer(screen)
    from rhythm_motor import RollDisplay
    roll_display = RollDisplay()

    def zero_twist():
        motor.calibrate_neutral()
        roll_display.reset()

    frame_clock = FrameClock()
    debug = config.DEBUG_OVERLAY_DEFAULT
    show_controls = config.CONTROLS_BAR_DEFAULT
    running = True

    def on_sigint(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, on_sigint)

    if args.song:
        err = game.play_by_name(args.song)
        if err:
            print(f"[songs] {err}")
            game.state = "select"
            game.say(err, 5)
        else:
            zero_twist()

    try:
        while running and not game.quit:
            # sleeps, so the motor's BLE thread gets to run (see FrameClock)
            dt = frame_clock.tick(config.FPS) / 1000.0
            now = clock()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.VIDEORESIZE:
                    w, h = max(640, ev.w), max(420, ev.h)
                    screen = pygame.display.set_mode((w, h), flags)
                    renderer = RhythmRenderer(screen)
                elif ev.type == pygame.KEYDOWN:
                    k, shift = ev.key, ev.mod & pygame.KMOD_SHIFT
                    st = game.state
                    if k == pygame.K_q and ev.mod & pygame.KMOD_CTRL:
                        running = False
                    elif k == pygame.K_h:
                        show_controls = not show_controls
                    elif k == pygame.K_F1:
                        debug = not debug
                    elif k == pygame.K_SPACE and config.KEYBOARD_FALLBACK:
                        game.swing(clock(), "keyboard")
                    elif st == "calib_av" and k in (pygame.K_LEFT, pygame.K_RIGHT):
                        step = 1 if shift else config.CALIB_AV_STEP_MS
                        game.on_adjust(step if k == pygame.K_RIGHT else -step)
                    elif k == pygame.K_UP:
                        game.on_up()
                    elif k == pygame.K_DOWN:
                        game.on_down()
                    elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        game.on_enter()
                    elif k == pygame.K_ESCAPE:
                        game.on_escape()
                    elif k == pygame.K_r:
                        game.on_retry()
                    elif k == pygame.K_c:
                        if st == "play":
                            zero_twist()
                        else:
                            game.on_calibrate()
                    elif k == pygame.K_m:
                        game.cycle_mode()
                    elif k == pygame.K_v:
                        game.on_calibrate(visual=True)

            if config.KEYBOARD_FALLBACK:            # held arrows glide the paddle, as in ping pong
                keys = pygame.key.get_pressed()
                glide = game.state == "play"
                keyboard.update(now, dt, glide and keys[pygame.K_LEFT], glide and keys[pygame.K_RIGHT])

            while True:
                try:
                    sw = swings.get_nowait()
                except queue.Empty:
                    break
                game.swing(sw.t, "imu")
            was = game.state
            game.update(now)
            if game.state == "play" and was != "play":
                zero_twist()                       # current grip = neutral (BLACK face toward the screen)

            snap = pose.snapshot() if pose else None
            statuses = [("Motor", motor.status), ("Pose", pose.status if pose else "disabled"),
                        ("Audio", engine.status)]
            det = getattr(motor, "detector", None)
            swinging = (det is not None and det._active is not None) or getattr(motor, "quiet", False) or (
                game.last_swing_wall is not None and now - game.last_swing_wall < config.ROLL_DISPLAY_SWING_HOLD_S)
            roll = roll_display.update(now, getattr(motor, "roll_deg", None), swinging)
            view = PlayerView(x=player.current_paddle_x(), roll_deg=roll, swing_t=game.last_swing_wall)
            hint = CONTROLS.get(game.state, []) if show_controls else ["H show controls"]
            lines = debug_lines(now, game, motor, pose, player, engine, frame_clock) if debug else None
            renderer.draw(now, game, view, statuses, snap, lines, hint)
            pygame.display.flip()
    finally:
        print("Shutting down...")
        engine.close()
        pygame.quit()
        if pose:
            pose.stop()
        if use_motor:
            motor.stop()
        print("Bye.")


def debug_lines(now, game, motor, pose, player, engine, frame_clock):
    from judge import contact_distance
    st = engine.song_time(now)
    lines = [f"FPS {frame_clock.get_fps():5.1f}  state={game.state}  input={player.source}  "
             f"song t={'--' if st is None else f'{st:7.3f}'} s",
             f"audio {engine.name}: {engine.status}  underruns={getattr(engine, 'underruns', 0)}",
             f"offsets: input {config.INPUT_OFFSET_MS}  visual {config.VISUAL_OFFSET_MS:+.0f} ms",
             f"windows: perfect ±{config.PERFECT_WINDOW_MS} good ±{config.GOOD_WINDOW_MS} ms  "
             f"cooldown {getattr(motor, 'cooldown_s', config.RHYTHM_SWING_COOLDOWN_S) * 1000:.0f} ms"]
    det = getattr(motor, "detector", None)
    if det is not None:
        lines.append(f"dyn |a| {det.last_dyn_accel_g:5.2f} g (thr {det.accel_threshold_g:.2f})  "
                     f"|gyro| {det.last_gyro:6.0f} (thr {det.gyro_threshold:.0f})  quiet={motor.quiet}")
    if pose is not None:
        snap = pose.snapshot()
        px = "--" if snap.paddle_x is None else f"{snap.paddle_x:+.2f} m"
        lines.append(f"pose paddle {px}  cam {pose.fps:.0f} fps")
    lines.append(f"paddle x {player.current_paddle_x():+.2f} m  mode {game.mode}"
                 + (f" (contact ±{contact_distance():.2f} m)" if game.mode == "aim" else ""))
    if game.chart is not None and st is not None:
        bar, beat = game.chart.bar_of(game.chart.time_beat(st))
        lines.append(f"bar {bar} beat {beat:4.2f}  tempo {game.chart.tempo.bpm_at(game.chart.time_beat(st)):g} BPM")
    j = game.last_judgment
    if j is not None:
        err = "--" if j.error_ms is None else f"{j.error_ms:+.0f} ms"
        lines.append(f"last: cue {j.index} {j.grade} {err} {j.reason}  haptic {game.last_haptic_ms} ms")
    return lines


if __name__ == "__main__":
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    main()
