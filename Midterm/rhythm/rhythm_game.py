"""
Game flow and play logic for Concert Rally (no pygame here -- rhythm_render.py draws it,
rhythm_main.py feeds it keys and swings).

    title -> select -> (calibration) -> play (countdown on the song's beats) -> results

Clocks: swings and frames come in as wall time (audio_engine.clock()); the audio engine
converts wall time to SONG time, which drives everything in play: ball positions, cue
sounds and tempo ticks (pre-scheduled into the mixer), the tempo LED and judging. Beats are
turned into song time only through the chart's tempo map. Nothing depends on the frame rate.
"""

import math
from dataclasses import dataclass

import config
import calibration
from audio_engine import clock, load_audio
from chart import pattern_text, scan_songs
from judge import Judge, paddle_near
from rhythm_motor import cooldown_for, haptic_plan, haptic_pulse_ms
from scheduler import Outcome, build_schedule
from scoring import BestScores, ScoreKeeper, rank_for
from sounds import SoundBank

TITLE_ITEMS = ["Play", "Calibrate timing", "Calibrate visuals", "Quit"]


def check_mode(mode):
    if mode not in config.GAME_MODES:
        raise ValueError(f"GAME_MODE must be one of {', '.join(config.GAME_MODES)}; got {mode!r}")
    return mode


def mode_label(mode):
    return f"{config.GAME_MODE_NAMES[mode]} ({config.GAME_MODE_HELP[mode]})"


@dataclass
class SongInfo:
    """Chart checks shown on the song select screen."""
    cooldown_s: float
    cooldown_shortened: bool
    haptic_short: int        # cues whose pulse is shortened
    haptic_skipped: int      # cues whose pulse is skipped
    length_s: float
    notes: list


class RhythmGame:
    def __init__(self, engine, player, *, motor=None, best=None, songs_dir=None,
                 clock_fn=None):
        self.engine = engine
        self.clock = clock_fn or clock
        self.player = player
        self.motor = motor
        self.best = best or BestScores()
        self.songs_dir = songs_dir
        self.sounds = SoundBank(engine.sr)
        self.state = "title"
        self.quit = False
        self.menu_index = 0
        self.songs = []
        self.skipped = []            # (chart file, reason) of charts that couldn't be loaded
        self.sel_index = 0
        self._info_cache = {}
        self._audio_cache = {}
        self.message = ""
        self.mode = check_mode(config.GAME_MODE)
        self.message_until = 0.0
        # play
        self.chart = None
        self.entry = None
        self.flights = []
        self.judge = None
        self.score = None
        self.outcomes = {}
        self.popups = []             # (Judgment, wall time)
        self.last_swing_wall = None
        self.last_judgment = None
        self.last_haptic_ms = None
        self.end_song_t = 0.0
        self.first_cue_beat = 0.0
        self.tempo_changes = []
        self.results = None
        # calibration
        self.calib = None
        self.calib_return = "title"
        self.av_offset_ms = config.VISUAL_OFFSET_MS
        self._av_clicks_until = 0.0
        self.rescan()

    # =============================================================================
    # Menus
    # =============================================================================
    def rescan(self):
        self.songs, self.skipped = scan_songs(self.songs_dir)
        self._info_cache.clear()
        self.sel_index = min(self.sel_index, max(0, len(self.songs) - 1))

    @property
    def valid_songs(self):
        return self.songs

    def find_song(self, query):
        """Playable song whose file name or title matches `query` (case-insensitive; an exact
        match wins, else a unique partial match). Returns (entry, None) or (None, error message)."""
        q = query.strip().lower().replace("_", " ")
        songs = self.valid_songs

        def names(e):
            return {e.chart.title.lower(), e.chart.song_id.lower().replace("_", " ")}
        exact = [e for e in songs if q in names(e)]
        partial = [e for e in songs if any(q in n for n in names(e))]
        hits = exact or partial
        if len(hits) == 1:
            return hits[0], None
        listing = ", ".join(e.chart.title for e in (hits or songs))
        return None, (f"'{query}' matches several songs: {listing}" if hits
                      else f"no song matches '{query}'. Songs: {listing}")

    def play_by_name(self, query):
        """Jump straight into a song (from the --song option). Returns an error message or None."""
        entry, err = self.find_song(query)
        if entry is None:
            return err
        self.sel_index = self.songs.index(entry)
        self.state = "select"                  # Esc / the results screen go back to song select
        self.start_song(entry)
        return None

    def selected_entry(self):
        return self.songs[self.sel_index] if self.songs else None

    def say(self, text, seconds=2.5):
        self.message, self.message_until = text, self.clock() + seconds

    def on_up(self):
        if self.state == "title":
            self.menu_index = (self.menu_index - 1) % len(TITLE_ITEMS)
        elif self.state == "select" and self.songs:
            self.sel_index = (self.sel_index - 1) % len(self.songs)

    def on_down(self):
        if self.state == "title":
            self.menu_index = (self.menu_index + 1) % len(TITLE_ITEMS)
        elif self.state == "select" and self.songs:
            self.sel_index = (self.sel_index + 1) % len(self.songs)

    def on_enter(self):
        if self.state == "title":
            item = TITLE_ITEMS[self.menu_index]
            if item == "Play":
                self.rescan()
                self.state = "select"
            elif item == "Calibrate timing":
                self.start_input_calibration("title")
            elif item == "Calibrate visuals":
                self.start_av_calibration("title")
            else:
                self.quit = True
        elif self.state == "select":
            e = self.selected_entry()
            if e is not None:
                self.start_song(e)
        elif self.state == "results":
            self.state = "select"
        elif self.state == "calib_input" and self.calib and self.calib.result is not None:
            r = self.calib.result
            if r.ok:
                calibration.save(input_offsets={self.calib.source: r.offset_ms})
                self.say(f"Saved: {self.calib.source} offset {r.offset_ms:+.0f} ms")
            self.end_calibration()
        elif self.state == "calib_av":
            calibration.save(visual_offset_ms=self.av_offset_ms)
            self.say(f"Saved: visual offset {self.av_offset_ms:+.0f} ms")
            self.end_calibration()

    def on_escape(self):
        if self.state == "play":
            self.engine.stop_song()
            self.state = "select"
        elif self.state in ("calib_input", "calib_av"):
            self.av_offset_ms = config.VISUAL_OFFSET_MS
            self.end_calibration()
        elif self.state in ("select", "results"):
            self.state = "title"

    def on_retry(self):
        if self.state == "results" and self.entry is not None:
            self.start_song(self.entry)
        elif self.state == "calib_input":
            self.start_input_calibration(self.calib_return)
        elif self.state == "select":
            self.rescan()
            extra = f", skipped {len(self.skipped)} (see the console)" if self.skipped else ""
            self.say(f"Found {len(self.songs)} playable song(s){extra}")

    def on_calibrate(self, visual=False):
        if self.state in ("title", "select", "results"):
            ret = "select" if self.state != "title" else "title"
            if visual:
                self.start_av_calibration(ret)
            else:
                self.start_input_calibration(ret)

    def on_adjust(self, delta_ms):
        if self.state == "calib_av":
            self.av_offset_ms = max(-500.0, min(500.0, self.av_offset_ms + delta_ms))

    # =============================================================================
    # Game mode
    # =============================================================================
    def cycle_mode(self):
        """M on song select: aim -> timing -> follow -> aim."""
        if self.state != "select":
            return
        modes = config.GAME_MODES
        self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
        self.say(f"Mode: {mode_label(self.mode)}")

    def score_key(self, chart):
        """Best scores are kept per song AND mode."""
        return f"{chart.song_id}@{self.mode}"

    def best_for(self, chart):
        return self.best.get(self.score_key(chart))

    def ball_x_at(self):
        """For scheduler.ball_at: in follow mode, where each ball should reach your end (your
        paddle's x: live until the ball gets there, then frozen). None in the other modes."""
        if self.mode != "follow":
            return None
        live = self.player.current_paddle_x()
        return lambda i: self.hit_x.get(i, live)

    # =============================================================================
    # Song checks (select screen)
    # =============================================================================
    def song_info(self, entry):
        if entry is None:
            return None
        key = entry.path
        if key not in self._info_cache:
            ch = entry.chart
            flights = build_schedule(ch)
            cd, short = cooldown_for(ch.min_gap_s()) if len(ch.cues) > 1 else (config.RHYTHM_SWING_COOLDOWN_S, False)
            plan = haptic_plan(flights, config.INPUT_OFFSET_MS.get("imu", 0.0))
            full = config.RHYTHM_HAPTIC_PULSE_MS
            notes = []
            if ch.override is not None:
                notes.append(f"Hit pattern {pattern_text(ch.override)} in every bar "
                             "(HIT_BEATS in config.py overrides the chart; None = the chart's own)")
            else:
                secs = "".join(f", bars {s.first_bar}-{s.last_bar} {pattern_text(s.pattern)}" for s in ch.sections)
                notes.append(f"Hit pattern {pattern_text(ch.pattern)}{secs}  (bars {ch.start_bar}-{ch.end_bar})")
            changes = ch.tempo_changes()
            if changes:
                notes.append("Tempo jumps: " + ", ".join(
                    f"{c.old_bpm:.0f} -> {c.new_bpm:.0f} BPM at {int(c.t // 60)}:{c.t % 60:04.1f}" for c in changes))
            if short:
                notes.append(f"swing cooldown shortened to {cd * 1000:.0f} ms for the closest cues")
            n_short = sum(1 for p in plan if 0 < p < full)
            n_skip = sum(1 for p in plan if p == 0)
            if config.RHYTHM_HAPTIC_ENABLED and (n_short or n_skip):
                notes.append(f"haptic: {n_short} pulse(s) shortened, {n_skip} skipped (close cues)")
            length = max(flights[-1].t_return + flights[-1].spb, 0.0)
            self._info_cache[key] = SongInfo(cd, short, n_short, n_skip, length, notes)
        return self._info_cache[key]

    # =============================================================================
    # Play
    # =============================================================================
    def _music(self, chart):
        p = chart.song_path
        if p not in self._audio_cache:
            self._audio_cache.clear()          # keep one song in memory
            self._audio_cache[p] = load_audio(p, self.engine.sr)
        return self._audio_cache[p]

    def miss_wait_s(self):
        worst = max([0.0] + [v for v in config.INPUT_OFFSET_MS.values()]) / 1000
        return worst + config.RHYTHM_SWING_PEAK_WINDOW_S + config.MISS_GRACE_MS / 1000

    def start_song(self, entry):
        ch = entry.chart
        try:
            music = self._music(ch)
        except Exception as e:
            self.say(f"Couldn't load {ch.song}: {e}", 4)
            return
        self.entry, self.chart = entry, ch
        self.flights = build_schedule(ch)
        self.judge = Judge(self.flights, miss_wait_s=self.miss_wait_s())
        self.score = ScoreKeeper(song_id=self.score_key(ch), total_cues=len(ch.cues))
        self.outcomes, self.popups = {}, []
        self.hit_x = {}              # follow mode: your paddle's x when each ball reached you
        self.last_judgment = self.results = None
        info = self.song_info(entry)
        if self.motor is not None and hasattr(self.motor, "set_cooldown"):
            self.motor.set_cooldown(info.cooldown_s)
        self.first_cue_beat = ch.cues[0].beat
        self.tempo_changes = ch.tempo_changes()
        countdown_t0 = ch.beat_time(self.first_cue_beat - config.COUNTDOWN_BEATS)
        preroll = max(0.0, config.PREROLL_MIN_S - countdown_t0)
        music_len = len(music) / self.engine.sr
        self.end_song_t = max(music_len, self.flights[-1].t_return + self.flights[-1].spb) + 0.3
        self.engine.play_song(music, preroll_s=preroll, length_s=self.end_song_t)
        # every cue sound is placed in the mixer now, at its exact song time
        if config.CUE_SOUNDS_ENABLED:
            for fl in self.flights:
                if fl.t_warn is not None:
                    snd = self.sounds.upbeat if fl.upbeat else self.sounds.warning
                    self.engine.schedule(snd, fl.t_warn, config.CUE_VOLUME)
        if config.TEMPO_TICK_ENABLED:
            for change in self.tempo_changes:
                for t in change.blink_times():
                    self.engine.schedule(self.sounds.tempo_tick, t, config.TEMPO_TICK_VOLUME, tag="tempo")
        mode = config.COUNTDOWN_CLICKS
        if mode != "off":
            for k in range(config.COUNTDOWN_BEATS):
                t = ch.beat_time(self.first_cue_beat - config.COUNTDOWN_BEATS + k)
                if mode == "always" or t < 0:
                    self.engine.schedule(self.sounds.click_accent if k == 0 else self.sounds.click, t,
                                         config.METRONOME_VOLUME)
        self.state = "play"

    def song_time(self, now=None):
        return self.engine.song_time(now)

    def draw_time(self, now=None):
        """Song time used for DRAWING (shifted by the visual offset)."""
        t = self.engine.song_time(now)
        if t is None:
            return None
        off = self.av_offset_ms if self.state == "calib_av" else config.VISUAL_OFFSET_MS
        return t - off / 1000

    def countdown_label(self, song_t):
        """'3', '2', '1', 'GO!' on the beats before the first cue, else None."""
        if self.chart is None or song_t is None:
            return None
        b = self.chart.time_beat(song_t)
        k = math.floor(b - (self.first_cue_beat - config.COUNTDOWN_BEATS) + 1e-9)
        if 0 <= k < config.COUNTDOWN_BEATS:
            n = config.COUNTDOWN_BEATS - 1 - k
            return str(n) if n > 0 else "GO!"
        return None

    def progress(self, song_t):
        if not self.end_song_t or song_t is None:
            return 0.0
        return min(1.0, max(0.0, song_t / self.end_song_t))

    def next_flight(self, song_t):
        """The next cue nobody has judged yet (for the arrival hint), or None."""
        for fl in self.flights:
            if fl.index not in self.outcomes and fl.t_arrive >= song_t - 0.05:
                return fl
        return None

    # =============================================================================
    # Input
    # =============================================================================
    def swing(self, t_wall, source):
        """A swing: t_wall = clock() time of the IMU peak (or the key press)."""
        self.last_swing_wall = t_wall
        if self.state == "play" and self.judge is not None:
            offset = config.INPUT_OFFSET_MS.get(source, 0.0) / 1000
            st = self.engine.song_time(t_wall)
            now_song = self.engine.song_time()
            if st is None:
                return None
            aim_ok = None
            if self.mode == "aim":
                # contact is checked when the ball reaches you (wall clock), not at the swing
                wall_of = t_wall - st                                  # song time -> wall time
                aim_ok = lambda fl: paddle_near(self.paddle_x_at, fl.x_arrive, wall_of + fl.t_arrive)  # noqa: E731
            j = self.judge.swing(st - offset, now_song, source, aim_ok)
            if j is not None:
                self._apply(j)
            return j
        if self.state == "calib_input" and self.calib is not None and self.calib.result is None:
            st = self.engine.song_time(t_wall)
            if st is not None:
                self.calib.add_swing(st, source)
        return None

    def paddle_x_at(self, t):
        """Paddle x at wall time t for the contact check: the webcam's while it is tracking you
        (None if it has no sample near t -- never the keyboard paddle, which would sit wherever
        you last left it), else the arrow-key paddle."""
        if self.player.source == "pose":
            return self.player.pose.paddle_x_at(t)
        return self.player.keyboard.x_at(t)

    def _apply(self, j):
        now = self.clock()
        self.hit_x.setdefault(j.index, self.player.current_paddle_x())
        self.score.add(j)
        self.outcomes[j.index] = Outcome(j.hit, j.t_known)
        self.popups.append((j, now))
        self.popups = self.popups[-6:]
        self.last_judgment = j
        if j.hit:
            if config.HIT_SOUND_ENABLED:
                self.engine.play_now(self.sounds.hit, config.HIT_SOUND_VOLUME)
            self._haptic(j)

    def _haptic(self, j):
        if not (config.RHYTHM_HAPTIC_ENABLED and self.motor is not None and getattr(self.motor, "connected", False)):
            return
        nxt = j.index + 1
        if nxt < len(self.flights):
            open_t = (self.flights[nxt].t_arrive - config.GOOD_WINDOW_MS / 1000
                      + config.INPUT_OFFSET_MS.get("imu", 0.0) / 1000)
            avail_ms = (open_t - self.engine.song_time()) * 1000
            ms = haptic_pulse_ms(avail_ms)
        else:
            ms = config.RHYTHM_HAPTIC_PULSE_MS
        self.last_haptic_ms = ms
        if ms > 0:
            self.motor.pulse(ms)

    # =============================================================================
    # Per frame
    # =============================================================================
    def update(self, now=None):
        self.engine.update()
        now = self.clock() if now is None else now
        if self.state == "play":
            st = self.engine.song_time(now)
            if st is None:
                return
            for j in self.judge.update(st):
                self._apply(j)
            if self.mode == "follow":                      # freeze where each ball met your paddle
                x = self.player.current_paddle_x()
                for fl in self.flights:
                    if fl.t_arrive > st:
                        break
                    self.hit_x.setdefault(fl.index, x)
            if st >= self.end_song_t:
                self.finish_song()
        elif self.state == "calib_input" and self.calib is not None and self.calib.result is None:
            st = self.engine.song_time(now)
            if st is not None and st >= self.calib.length_s:
                self.calib.finish()
                self.engine.stop_song()
        elif self.state == "calib_av":
            st = self.engine.song_time(now)
            if st is not None and st + 2.0 > self._av_clicks_until:
                self._schedule_av_clicks(self._av_clicks_until, st + 4.0)

    def finish_song(self):
        self.engine.stop_song()
        self.score.finish()
        acc = self.score.accuracy
        new_acc, new_streak = self.best.record(self.score_key(self.chart), acc, self.score.max_streak)
        self.results = {
            "accuracy": acc, "rank": rank_for(acc), "counts": dict(self.score.counts),
            "max_streak": self.score.max_streak, "mean_error_ms": self.score.mean_error_ms,
            "new_best": new_acc, "new_streak": new_streak, "title": self.chart.title,
        }
        self.state = "results"

    # =============================================================================
    # Calibration
    # =============================================================================
    def start_input_calibration(self, return_state):
        self.calib_return = return_state
        self.calib = calibration.InputCalibration()
        self.engine.play_song(None, preroll_s=0.8, length_s=self.calib.length_s)
        for k, t in enumerate(self.calib.click_times):
            accent = k % 4 == 0
            self.engine.schedule(self.sounds.click_accent if accent else self.sounds.click, t, config.METRONOME_VOLUME)
        self.state = "calib_input"

    def start_av_calibration(self, return_state):
        self.calib_return = return_state
        self.av_offset_ms = config.VISUAL_OFFSET_MS
        self.av_spb = 60.0 / config.CALIB_AV_BPM
        self.engine.play_song(None, preroll_s=0.5, length_s=3600.0)
        self._av_clicks_until = 0.0
        self._schedule_av_clicks(0.0, 4.0)
        self.state = "calib_av"

    def _schedule_av_clicks(self, t0, t1):
        spb = self.av_spb
        k = math.ceil(t0 / spb - 1e-9)
        while k * spb < t1:
            self.engine.schedule(self.sounds.click_accent if k % 4 == 0 else self.sounds.click, k * spb,
                                 config.METRONOME_VOLUME)
            k += 1
        self._av_clicks_until = k * spb

    def end_calibration(self):
        self.engine.stop_song()
        self.calib = None
        self.state = self.calib_return
