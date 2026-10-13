# Concert Rally (rhythm ping pong)

The ping-pong mechanics, set to music. You hold the same LEGO Education **Double Motor** (the hub
is the handle) and stand in front of the same webcam. A floating opponent paddle hits the ball to
you **on the beat** of a song; **swing** on time (IMU) to send it back. Each ball you meet is graded
Perfect, Good or Miss. Your paddle moves exactly as in ping pong: a continuous slider that follows
your wrist (webcam pose) or the arrow keys. There are three [game modes](#game-modes): in **Aim**
(the default) the paddle must also be at the ball; in **Timing** and **Follow** only timing counts.
The look is deliberately plain: a dark backboard, a floor and a blue table.

The songs are real music files: eight royalty-free tracks by Kevin MacLeod (see
[Credits](#credits)). No song in this project is AI- or code-generated. The only sounds made in
code are short effects: clicks, warning cues, the tempo-change tick and the hit "tock"
(`sounds.py`).

Nothing in `../` (the ping-pong game), `../bowling` or `../minigolf` was changed. This folder reuses
these modules by import:
- `../motor_imu.py`, subclassed in `rhythm_motor.py`
- `../spin.py` (handle roll)
- `../pose_tracker.py`, `../player_input.py`
- `../camera3d.py`, `../scene.py`, `../paddle.py`, `../theme.py`

`config.py` loads `../config.py` first, so the motor, IMU, roll, webcam and pose settings tuned
for ping pong carry over. Hardware settings stay in one place.

**No MQTT.** This game never imports or connects to MQTT; a test checks this. `scoring.py` sends
every score change through one function, `ScoreKeeper.emit()`, so publishing can be added later
with one listener (see [MQTT later](#mqtt-later)).

## Setup
Use the same venv as ping pong (`../../my_env`, Python 3.14), plus three audio packages:
```powershell
cd "ME 193 AI Mobile Robotics\Midterm"
..\my_env\Scripts\python.exe -m pip install -r requirements.txt
```
- `sounddevice` handles audio output and the clock.
- `soundfile` decodes WAV, OGG and MP3.
- `librosa` is optional and improves tempo detection in `make_chart.py`.

All three have Python 3.14 wheels.

`songs/` holds eight songs ready to play (see [Songs in songs/](#songs-in-songs)).

## Running
```powershell
cd "ME 193 AI Mobile Robotics\Midterm\rhythm"
..\..\my_env\Scripts\python.exe rhythm_main.py              # motor + webcam + audio
..\..\my_env\Scripts\python.exe rhythm_main.py --keyboard   # no hardware at all
..\..\my_env\Scripts\python.exe rhythm_main.py --no-camera  # motor swings, arrow keys move the paddle
..\..\my_env\Scripts\python.exe rhythm_main.py --mode follow # start in another game mode
```
- **Choosing a song:** song select (Up/Down, Enter) lists every song in `songs/` with a valid
  chart. For quick testing, skip the menus:
  ```powershell
  ..\..\my_env\Scripts\python.exe rhythm_main.py --list-songs                  # what's playable
  ..\..\my_env\Scripts\python.exe rhythm_main.py --keyboard --song polka       # play it straight away
  ..\..\my_env\Scripts\python.exe rhythm_main.py --song "electrodoodle"
  ```
  `--song` matches the title or file name, ignoring case; a unique partial match is enough. When
  the song ends (or you press Esc) you land on song select as usual.
- `--no-motor` turns off the motor only.
- `--audio pygame` / `--audio null` pick another audio backend (see [Timing engine](#timing-engine)).
- Hold the motor still for about a second after it connects so it can calibrate gravity.
- The game takes your current grip as "neutral twist" when a song starts. The neutral grip has the
  BLACK face toward the screen, so the game shows YELLOW toward you, like ping pong.
- An invalid `HIT_BEATS` or `GAME_MODE` in `config.py` stops the game at startup with a message
  saying what's allowed.

**Flow:** title → song select → (optional) calibration → countdown on the song's beats → play →
results.

### Controls
| Key | Action |
|---|---|
| swing the paddle / **Space** | hit (Space works everywhere as a fallback) |
| move your arm / hold **← →** | move the paddle, as in ping pong: holding ←/→ glides it smoothly across the table (`KEYBOARD_PADDLE_SPEED_M_S`) and it stays where you let go. In Aim mode it must be at the ball |
| M | song select: cycle the game mode (Aim → Timing → Follow) |
| ↑ / ↓, Enter, Esc | menus: choose, select, back. Esc during a song leaves it |
| C | menus: timing calibration. During a song: re-zero the paddle twist |
| V | visual (A/V) calibration |
| R | song select: rescan `songs/`. Results: retry. Calibration: try again |
| H | show / hide the controls bar (it wraps to fit any window size) |
| F1 | debug overlay: song clock, bar / beat / tempo, offsets, IMU, pose, last judgment, haptic pulse |
| Ctrl+Q / close window | quit |

The window is resizable, and everything re-lays out.

**Paddle twist on screen** is cosmetic, since spin isn't scored. Swinging every beat gives the
twist filter few still moments to correct gyro drift, and each haptic pulse kicks the hub, so the
shown twist is calmed:
- it ignores the hub's movement during haptic pulses;
- it holds during swings;
- it ignores twists under `ROLL_DISPLAY_DEADZONE_DEG`;
- it lets slow drift fade back to straight over `ROLL_DISPLAY_RECENTER_S`.

Press **C** during a song to re-zero it, or set `ROLL_DISPLAY_ENABLED = False` to keep the paddle
straight. If small movements count as swings, raise `RHYTHM_SWING_THRESHOLD_SCALE` (e.g. 1.5).

If the webcam is tracking you, it moves the paddle, so the arrows have no effect until it loses
you. Use `--no-camera` to move it with the keyboard.

## Game modes
Press **M** on song select to cycle the mode (it's shown under "Choose a song" and in the play
HUD). `GAME_MODE` in `config.py` (or `--mode`) sets the one the game starts in.

| Mode | The ball goes to | A hit needs |
|---|---|---|
| **Aim** (default) | a different spot each time (`ARRIVAL_X_SPREAD_M`); a soft patch on the table shows where (as wide as the zone your paddle centre must be in) | a swing on time (IMU) **and** the ball actually meeting your paddle: when the ball reaches you, the drawn blade must overlap the drawn ball (paddle centre within about 14 cm of the ball, `AIM_CONTACT_M`). Checked at the ball's arrival, from `AIM_LOOKBACK_MS` (30) before to `AIM_LOOKAHEAD_MS` (80) after, since webcam samples arrive late; not at the swing. If the webcam is tracking you but has no sample then, it's no contact (never the arrow-key paddle). Good timing without contact is a **Miss ("missed the ball")** |
| **Timing** | a different spot each time (a visual cue only) | good timing; the paddle's position doesn't matter |
| **Follow** | always **your paddle**: the flight bends smoothly toward wherever the paddle is, and your return starts from where you hit it | only the timing of your flick (IMU) |

Best scores are saved **per song and mode**, so an easier mode can't overwrite a harder mode's best.
Song select shows the best for the current mode.

## Scoring
- **Timing:** your swing's time (the IMU sample where the swing peaks, minus your calibrated input
  offset) is compared with the hit's beat (converted to song time through the chart's tempo map):
  - **Perfect:** within ±`PERFECT_WINDOW_MS` (45 ms).
  - **Good:** within ±`GOOD_WINDOW_MS` (100 ms).
  - Otherwise the swing doesn't count.

  Each swing goes to the nearest hit that hasn't been judged yet.
- **Aim** (Aim mode only): see [Game modes](#game-modes).
- **No swing:** a hit nobody swings at becomes a Miss once a late IMU event can no longer arrive.
- **Accuracy** (the big number) is the average over the hits judged so far of `GRADE_POINTS`:
  Perfect 100 %, Good 50 %, Miss 0 %. It's shown with one decimal and updates after every hit.
- A Miss resets your **streak**, but the song keeps playing.
- **Stray swings** with no ball near are ignored (`STRAY_SWING_PENALTY = True` makes them a Miss).
- **Results:** final accuracy, Perfect/Good/Miss counts, max streak, your average early/late time,
  and a rank from `RANKS` (S ≥ 95, A ≥ 85, B ≥ 70, C ≥ 55, else D).
- **Best scores:** your best accuracy and max streak per song and mode are saved to `scores.json`
  and shown on song select.

## Latency calibration (do this first)
The BLE IMU, the swing detection, the audio output and your own habits all shift when a swing
*registers* compared with when you *meant* it. Bluetooth headphones add another 150–250 ms. Use
wired headphones or speakers if you can; calibration still handles a steady offset.

1. **Timing (C, or "Calibrate timing" on the title screen).**
   - Swing on every click: a 4-click count-in, then 16 measured clicks at `CALIB_BPM`.
   - The screen gives no visual beat, on purpose: tune to your ears.
   - The game measures each swing's distance from its nearest click and drops swings more than
     `CALIB_OUTLIER_MS` from the median. It then shows the **median offset**, the spread, and how
     far off you were with the old offset ("before"). After saving, that error is about 0.
   - Press **Enter** to save. Offsets are stored per input: calibrate once with the paddle and
     once with Space if you use both.
2. **Visuals (V).**
   - A ball bounces on the table on every click.
   - Press ←/→ (Shift for 1 ms steps) until it touches the table exactly on the click, then Enter.
   - This only moves the picture (`VISUAL_OFFSET_MS`, positive = draw later), never the judging.

Both values go to `calibration.json` (ignored by git). `config.py` loads it at startup and
overrides the `INPUT_OFFSET_MS` / `VISUAL_OFFSET_MS` defaults. Rerun calibration any time from the
title, song select or results screen.

## Timing engine
**The song's playback position is the master clock.** Ball positions, opponent swings, cue sounds,
the tempo LED and judging are all computed from song time, never from frame counts. A dropped
frame just skips a picture.

- **Audio (`audio_engine.py`, `AUDIO_BACKEND = "sounddevice"`).** Our own mixer runs inside a
  PortAudio callback on Windows WASAPI. Each callback reports when its first sample reaches the
  speaker, on the same clock as `time.perf_counter()` (which equals `time.monotonic()` here). So
  "which sample is being heard at time t" is known to the sample.
  - Warning cues, tempo-change ticks, countdown clicks and metronome clicks are mixed in at
    **exact sample positions**, scheduled when the song starts.
  - Measured on this laptop: song-clock jitter < 1 ms (std 0.3 ms), no underruns, about 22 ms
    output latency.
  - `pygame.mixer` was not used as the clock: its position counts from `play()`, not from the
    speaker, and drifts with OGG/MP3. It's kept only as a fallback backend.
- **Tempo map (`tempo.py`).** Every beat ↔ song-time conversion goes through the chart's tempo
  map, so hit times, flight durations, warning cues, judgment windows and the tempo-change
  indicator all follow tempo changes. When the tempo rises, flights get shorter in seconds.
- **Swings** are timestamped at their **peak** IMU sample (the contact moment), not when the game
  notices them. The event is sent as soon as the peak is over (`SWING_PEAK_DROP`), at most
  `RHYTHM_SWING_PEAK_WINDOW_S` after the onset.
- **Timing rule** (`scheduler.py`): the ball spends **equal time in the air each way**, so the
  opponent hits exactly halfway (in beats) between two of your hits. You never hit on beat 3.

  | Pattern | you hit on | opponent hits on | each flight |
  |---|---|---|---|
  | `[4]` | 4 | 2 | 2 beats |
  | `[2, 4]` | 2 and 4 | 1 and 3 | 1 beat |

  - **Switching patterns** needs no special case:
    - [4] → [2, 4]: you hit on 4, then on 2 of the next bar, so the opponent hits on that bar's 1.
    - [2, 4] → [4]: you hit on 4, then on 4 of the next bar, so the opponent hits on that bar's 2.

    The flight around the switch is half of the gap between your two hits.
  - The first serve comes half of the first gap before your first hit. Your last return takes
    half of the last gap.
  - If the tempo changes during a flight, the opponent stays on the beat, so the two directions can
    differ by a few ms.
  - Each hit stores its serve, hit and return times and its flight lengths explicitly. A later
    chart format can give hits any flight length (0.5 to 8 beats, upbeats) without touching the
    ball code (`scheduler.Contact`).
  - The incoming ball bounces on your side `BOUNCE_LEAD_BEATS` (half a beat) before it reaches you
    (halfway, for shorter flights). In Aim and Timing mode it arrives at a different x each time
    (`ARRIVAL_X_SPREAD_M`), marked by a soft spot on the table (`ARRIVAL_HINT`). In Follow mode the
    path bends to your paddle; only x changes, so timing is identical in every mode.
  - Arc height follows real gravity for the flight time, raised if needed to clear the net. Ball
    speed therefore comes purely from timing: there are no difficulty levels.
- **Late IMU hits:** the BLE swing event arrives 100–200 ms after you swing. Until it does, the
  ball eases to a stop just past your paddle, then blends onto its return path over
  `LATE_HIT_BLEND_S`. Perfect/Good shows as soon as the swing arrives. A Miss shows when the
  deadline passes.
- **Swing cooldown:** `RHYTHM_SWING_COOLDOWN_S` (220 ms) is checked against the loaded chart's
  closest hits and shortened if needed. Song select shows a note when that happens.
- **Haptics:** a successful hit pulses the motor for at most `RHYTHM_HAPTIC_PULSE_MS`. The IMU is
  ignored during the pulse plus `RHYTHM_HAPTIC_SETTLE_MS`. Before every pulse, the game works out
  how long it has until the next hit's timing window opens, then shortens the pulse or skips it.
  Song select reports how many pulses a chart shortens or skips. [2, 4] at 160 BPM (750 ms between
  hits) always fits.

## Tempo change indicator
Before each **sudden tempo change**, a round LED (in a dark bezel, above the table) blinks
**3 times at the new tempo**, with a metronome tick on each blink: **amber before a speed-up, blue
before a slowdown**. The blinks land 3, 2 and 1 new-tempo beats before the change, so the new
tempo's first beat is the next count ("3, 2, 1, go"). The ticks tell you *that* the tempo is
changing and *to what*. While the music is still at the old tempo, they deliberately run ahead of
it (speed-up) or lag behind it (slowdown). The LED also shows the new tempo (e.g. "160 BPM").

**What counts as sudden** (`TempoMap.sudden_changes`): at each tempo-segment boundary, the new
tempo is compared with the tempo `TEMPO_SUDDEN_WITHIN_BEATS` (2) beats earlier.
- A rise **or fall** of **more than** `TEMPO_JUMP_MIN_BPM` (5 BPM) triggers it.
- A single boundary is an instant change, so it counts by itself. Several small steps in the same
  direction close together count when they add up within the window.
- Gradual drift (small steps further apart) never triggers it.

| Setting | What |
|---|---|
| `TEMPO_JUMP_MIN_BPM`, `TEMPO_SUDDEN_WITHIN_BEATS` | the detection rule above |
| `TEMPO_TICK_ENABLED`, `TEMPO_TICK_VOLUME` (1.0) | the metronome tick, separate from music and warning cues |
| `TEMPO_LED_BLINKS`, `TEMPO_LED_BLINK_BEATS` | how many blinks, how long each stays lit |
| `TEMPO_LED_POS`, `TEMPO_LED_RADIUS_PX`, `TEMPO_LED_COLOR` (speed-up), `TEMPO_LED_SLOW_COLOR` (slowdown), `TEMPO_LED_SHOW_S`, `TEMPO_LED_LABEL` | look and placement |

Song select lists each chart's tempo jumps. Electrodoodle has one: 120 → 160 BPM at 2:10.0, with the ticks on its drum fill at 2:08.9–2:09.65.
None of
the current songs slows down suddenly.

## Which beats you hit (patterns and `HIT_BEATS`)
Each chart says which beats of each bar you hit (see [Chart format](#chart-format)). By default
every bar uses `[4]`, and a chart can switch ranges of bars to `[2, 4]`. **Only `[4]` and `[2, 4]`
exist**; anything else is an error with a message saying so.

`HIT_BEATS` in `config.py` overrides every chart while you play:

| `HIT_BEATS` | You hit on |
|---|---|
| `None` (default) | each chart's own pattern and sections |
| `[4]` | beat 4 of every bar (opponent on 2) |
| `[2, 4]` | beats 2 and 4 of every bar (opponent on 1 and 3) |

- The override covers the chart's `start_bar` to `end_bar`. The chart files on disk are never
  changed.
- Song select shows the pattern in use.
- Best scores are saved per song, whatever the pattern.

## Songs in `songs/`
| Song | File played | Chart |
|---|---|---|
| Electrodoodle | `Electrodoodle.wav` | **120 → 160 BPM** at 2:10.0 (bar 66, beat 260), hits in bars 3–88, **[2, 4] in bars 66–73** (2:10.0–2:22.0), [4] elsewhere |
| Latin Industries | `Latin_Industries.wav` | 122 BPM, hits in bars 3–100, [4] |
| Spazzmatica Polka | `Spazzmatica_Polka.wav` | 140 BPM, hits in bars 3–55, [4] |
| Cut and Dry | `Cut_and_Dry.wav` | 122 BPM, hits in bars 3–111, [4] |
| Itty Bitty 8 Bit | `Itty_Bitty_8_Bit.wav` | 109 BPM, hits in bars 3–87, [4] |
| Kick Shock | `Kick_Shock.wav` | 138 BPM, hits in bars 3–34, [4] |
| Theme for Harold var 3 | `Theme_for_Harold_var_3.wav` | 176 BPM, hits in bars 3–55, [4]. About a third of the song also fits 117 BPM (3:2, a triplet feel); if 176 sounds wrong, re-run detect with `--bpm 117.3 --force` |
| Pinball Spring 160 | `Pinball_Spring_160.wav` | 160 BPM, hits in bars 3–111, [4] |

- The game plays the WAVs (decoded from the MP3s next to them with `--to-wav`), whose timing is
  sample-exact.
- All charts came from `make_chart.py detect --incompetech`. **Check each one by ear with
  `make_chart.py preview`** and nudge if needed.
- **Electrodoodle's groove is built from dotted eighths**: notes 0.375 s apart, which is exactly
  one beat at 160 BPM. So a "160 BPM pulse" runs through the 120 BPM part too, and a detector can
  mistake one for the other. The music changes with a four-kick drum fill at 0.375 s spacing
  (2:08.90, 2:09.28, 2:09.65), landing on a big kick at **2:10.0, bar 66**, where the 160 BPM
  section starts. The three LED ticks fall exactly on that fill. (The first chart put the change
  at 2:08.5, mid-bar at the start of the fill, so everything came a bar early.)
- **Latin Industries** has strong off-beat percussion. Its kick and its off-beats are nearly
  equally strong, so if the clicks sit between the beats, press **B** in preview (half a beat).
- The previous single-BPM charts had beat 0 at 24.7 ms (Electrodoodle), 15 ms (Latin
  Industries) and 26.7 ms (Spazzmatica Polka). The new detection gives 19.7 ms, 269.6 ms (half a
  beat after 15 ms) and 45.3 ms.

## Adding a song
1. Put a **WAV** or **OGG** file in `songs/`. These are preferred, since they decode
   sample-exactly. **MP3 works too, but MP3 files can start 25–50 ms late** (encoder delay and
   padding vary by encoder). Check the offset by ear, or use OGG. No streaming: local files only.
2. Make a chart:
   ```powershell
   ..\..\my_env\Scripts\python.exe make_chart.py detect songs\my_song.wav
   ..\..\my_env\Scripts\python.exe make_chart.py detect songs\my_song.wav --author "Someone" --force
   ..\..\my_env\Scripts\python.exe make_chart.py detect "songs\Some Incompetech Song.mp3" --to-wav --incompetech
   ```
   This writes `songs/my_song.chart.json` with:
   - the tempo map;
   - beat 0 and which beat is bar 1's downbeat;
   - the bars with hits: after the intro and before the ending;
   - the `[4]` pattern on every bar.

   Other options:
   - `--to-wav` decodes an MP3 / OGG once to `songs/<Name_With_Underscores>.wav` and charts that
     (the title keeps the spaces). Recommended for MP3s.
   - `--incompetech` fills in the author and the CC BY 4.0 attribution.
   - `--bpm 128` forces one fixed tempo.
   - Re-running with `--force` keeps your title, author, attribution, pattern and sections (the old
     file is saved as `.chart.json.bak`).
3. **Check it by ear** with preview (below), then press **R** on song select (or reopen it).
   A chart that's invalid, or whose audio file is missing, is **skipped with a warning** in the
   console. Song select also notes how many were skipped. The game never crashes on one.

### How tempo detection works
`make_chart.py detect` finds the tempo over the whole song, not one average BPM:
1. **Onset strength:** librosa's when installed, else a built-in numpy spectral flux.
2. **Local tempo in overlapping windows** (`CHART_WINDOW_S` = 12 s, every `CHART_WINDOW_HOP_S` =
   3 s). The candidates are the autocorrelation peaks inside `CHART_BPM_RANGE` (90–185, so no
   half or double tempo). Each is refined with a comb filter, and the best fit wins, so a
   syncopated groove's 4:3 "tempo" loses to the real beat.
3. **Smoothing:** windows within `CHART_MERGE_BPM` of each other form one segment. A run shorter
   than `CHART_MIN_SEGMENT_S` (15 s) is detection jitter and joins its neighbour. Real tempo
   changes last longer, so they're kept. A segment whose windows mostly disagree with its tempo
   (fewer than `CHART_RUN_PURITY`, 80 %, agree) is a groove that fits two tempos, flipping back
   and forth (e.g. Theme for Harold's 3:2), not a tempo change. It joins its neighbour, the
   majority tempo wins, and detect prints the other tempo as a note.
4. **Steady grids:** each segment's tempo and beat phase are refitted over its whole span.
5. **Change points:** the change is placed where the *old* beat stops fitting the music (below
   `CHART_CHANGE_FIT` of its usual strength for two bars), on the beat the new grid meets most
   closely. Any small phase difference is kept as `shift_ms`.
6. **Downbeat:** the beat of the bar with the strongest bass (kick) onsets.
7. **Bar lines:** music changes tempo on a bar line. A change that landed mid-bar (often on a fill
   whose notes already fit the new tempo) moves to the nearest bar line within a bar where the
   old and new grids meet within `CHART_BARLINE_SNAP_MS` (20 ms). The new grid stays where it is;
   only the beat numbers after it shift, so its first beat is beat 1 of a bar.
7. **Start / end bar:** `CHART_INTRO_BARS` after the music gets loud (`CHART_LOUD_FRACTION` of the
   median bar loudness), and `CHART_OUTRO_BARS` before it fades out. Your last return must still
   be inside the song.

### Preview (check a chart by ear)
```powershell
..\..\my_env\Scripts\python.exe make_chart.py preview songs\Electrodoodle.chart.json
..\..\my_env\Scripts\python.exe make_chart.py preview songs\Electrodoodle.chart.json --at 2:00
```
The song plays with a **click on every chart beat** and a **louder click on beat 1 of each bar**,
plus the tempo-change ticks and LED.
- The window shows the bar and beat, the current tempo segment, your hits for this bar, and the
  next tempo jump.
- Clicks drifting away from the music means the tempo map is wrong. Clicks evenly early or late
  means the offset needs a nudge. The loud click on the wrong beat means the downbeat is wrong.

| Key | Action |
|---|---|
| ← / → | `offset_ms` −/+ 5 ms (Shift: 1 ms). Moves every beat |
| B | move every beat by half a beat (when the clicks sit on the off-beats) |
| ↑ / ↓ | `downbeat` +/− one beat (which beat is beat 1 of the bar) |
| [ / ] | `shift_ms` of the tempo segment playing now −/+ 5 ms (Shift: 1 ms). Moves that segment and everything after it |
| Space | pause / resume |
| PgUp / PgDn | 4 bars back / ahead (`PREVIEW_SEEK_BARS`) |
| J | jump to just before the next tempo jump |
| Home | restart |
| S | save (the old file is kept as `.chart.json.bak`) |
| Esc | quit (asks again if there are unsaved changes) |

Clicks are rescheduled as soon as you nudge, so you hear the change straight away. `start_bar`,
`end_bar` and `sections` are bar numbers, so after moving the downbeat check that they still fit
(edit them in the JSON).

To get the same click track as a file: `make_chart.py clicks <chart>` writes
`previews/<song>.clicks.wav`.

### Chart format
```json
{
  "song": "Electrodoodle.wav",
  "title": "Electrodoodle",
  "author": "Kevin MacLeod (Incompetech)",
  "attribution": {
    "title": "Electrodoodle",
    "artist": "Kevin MacLeod",
    "source": "incompetech.com",
    "license": "Creative Commons: By Attribution 4.0 License",
    "license_url": "http://creativecommons.org/licenses/by/4.0/",
    "credit": "\"Electrodoodle\" Kevin MacLeod (incompetech.com)\nLicensed under ..."
  },
  "offset_ms": 19.7,
  "tempo": [
    {"beat": 0, "bpm": 119.991},
    {"beat": 257, "bpm": 159.997, "shift_ms": -4.2}
  ],
  "downbeat": 0,
  "start_bar": 3,
  "end_bar": 88,
  "pattern": [4],
  "sections": [
    {"bars": [65, 73], "pattern": [2, 4]}
  ]
}
```
| Field | Meaning |
|---|---|
| `song` | audio file, relative to the chart |
| `title` | shown on song select (default: from the file name) |
| `author` | shown next to the title on song select |
| `attribution` | the full credit (title, artist, source, license, license URL, credit text) |
| `offset_ms` | time from the start of the file to **beat 0** |
| `tempo` | the tempo map: segments `{"beat": B, "bpm": X}`, the first on beat 0. Beats in a segment are evenly spaced. Optional `shift_ms` moves a segment's start (and every later beat) by a few ms when the new tempo doesn't begin exactly on the old grid; the beat just before it stretches or shrinks to fit |
| `downbeat` | which beat is **beat 1 of bar 1**. Bar *n*, beat *p* = `downbeat + 4 (n − 1) + (p − 1)` |
| `start_bar`, `end_bar` | the first and last bar with hits (inclusive) |
| `pattern` | the beats **you** hit in every bar: `[4]` (default) or `[2, 4]` |
| `sections` | optional bar ranges with another pattern: `{"bars": [first, last], "pattern": [2, 4]}`. Add `"warn": true` to play the warning cue before the section's first hit |

Why tempo segments and not a list of detected beat times? With segments:
- the grid is steady by construction, so there's no detection jitter;
- the chart is short and easy to edit by hand;
- fractional beats (upbeats, any flight length) convert exactly;
- a tempo jump is just a segment boundary, which the tempo-change indicator needs.

Example: bars 17–24 at [2, 4], the rest [4]:
```json
  "pattern": [4],
  "sections": [{"bars": [17, 24], "pattern": [2, 4]}]
```
Sections may not overlap. Extra fields, like `"notes"`, are kept when the tools re-save the file.

### Sound effects
- **Warning cue:** a section with `"warn": true` plays a two-note "ba-dum" `WARNING_LEAD_BEATS`
  (1 beat) before its first hit. An upbeat hit (future charts) would get a rising glide instead,
  and an amber halo on the ball.
- **Tempo tick:** see [Tempo change indicator](#tempo-change-indicator).
- `CUE_SOUNDS_ENABLED` and `CUE_VOLUME` control the warnings, separately from `MUSIC_VOLUME`.
- The soft hit "tock" has its own `HIT_SOUND_ENABLED` and `HIT_SOUND_VOLUME`. With the IMU, it
  plays when the hit registers, so it can sound slightly late.

## Tests
```powershell
cd "ME 193 AI Mobile Robotics\Midterm\rhythm"
..\..\my_env\Scripts\python.exe -m pytest tests              # no hardware needed (~35 s)
..\..\my_env\Scripts\python.exe -m pytest tests --hardware   # also checks the real audio clock
```
The tests use the real songs (read-only), mainly **Electrodoodle** because it has the tempo change
and a [2, 4] section. Charts that only need a file to exist use an empty placeholder file in a
temporary folder; no music is generated. The tests cover:
- **Tempo map:** conversions across tempo changes and shifts, the sudden-change rule (jumps up
  and down, exactly 5 BPM, drift both ways, close steps), blink times, the LED for both.
- **Charts:** bars, patterns and sections, `HIT_BEATS` validation and override, error messages,
  saving, skipping missing or broken songs, the eight Incompetech charts and their credits.
- **Scheduler:** [4] and [2, 4] with and without `HIT_BEATS`, switching patterns, equal flights
  each way, flights shrinking at a tempo increase, explicit flight times, bounces, net clearance,
  late-hit blending, misses.
- **Judging and scoring:** timing windows, the aim check (reach and the lag window), accuracy,
  streaks, best scores.
- **Calibration:** offset maths and saving.
- **Audio:** the clock, sample-exact mixing, seeking and cancelling, using a fake stream.
- **Motor:** peak-time swing detection and the haptic budget (on the real charts too).
- **Game flow:** a full keyboard play-through of Electrodoodle in every mode (100 % when every swing
  is on time, and in Aim mode at the ball). Also: its bars and tempo, "missed the ball" in Aim mode (contact at the ball's arrival, never the keyboard paddle while the webcam tracks you),
  position not mattering in Timing / Follow, the Follow ball reaching the paddle, M and per-mode
  best scores, the tempo ticks, and judging that is identical at 60 FPS and with heavy frame
  drops.
- **Chart helper:** tempo detection on the real songs (with and without librosa), mixed 3:2
  grooves, the detect command (also `--to-wav` from an MP3), the click track, preview editing and
  saving.
- **Rendering:** every screen at three window sizes, the play screen in every mode, and the LED
  colours.
- **No MQTT:** importing the game loads no MQTT module.

## Config (`config.py`)
Every tunable is in `config.py`, grouped and commented. Values marked `# DEFAULT` are starting
points chosen without hardware testing. The main groups:

| Group | Highlights |
|---|---|
| Songs | `HIT_BEATS` (`None`, `[4]` or `[2, 4]`), `BEATS_PER_BAR` (4), `SONGS_DIR`, `AUDIO_EXTENSIONS`, `SCORES_PATH` |
| Game modes | `GAME_MODE` (`aim`, `timing`, `follow`), `AIM_CONTACT_M`, `AIM_LOOKBACK_MS`, `AIM_LOOKAHEAD_MS`, `ARRIVAL_X_SPREAD_M` |
| Audio engine | `AUDIO_BACKEND`, `AUDIO_HOSTAPI` (WASAPI), `AUDIO_DEVICE`, `MUSIC_VOLUME`, `CUE_SOUNDS_ENABLED`, `CUE_VOLUME`, `TEMPO_TICK_ENABLED`, `TEMPO_TICK_VOLUME`, `HIT_SOUND_*`, `METRONOME_VOLUME`, `COUNTDOWN_CLICKS` |
| Judgment | `PERFECT_WINDOW_MS`, `GOOD_WINDOW_MS`, `MISS_GRACE_MS`, `GRADE_POINTS`, `STRAY_SWING_PENALTY`, `RANKS` |
| Latency | `INPUT_OFFSET_MS`, `VISUAL_OFFSET_MS` (overridden by `calibration.json`), `CALIB_*` |
| Swing | `RHYTHM_SWING_COOLDOWN_S`, `RHYTHM_SWING_PEAK_WINDOW_S`, `SWING_PEAK_DROP`, `RHYTHM_SWING_THRESHOLD_SCALE` (thresholds and axes come from `../config.py`) |
| Ball flight | `BOUNCE_LEAD_BEATS`, bounce spots, arc gravity / net clearance, `RETURN_TARGET_SPREAD_M`, miss / late-hit animation |
| Haptics | `RHYTHM_HAPTIC_ENABLED`, `RHYTHM_HAPTIC_PULSE_MS`, `RHYTHM_HAPTIC_SETTLE_MS`, `RHYTHM_HAPTIC_MIN_PULSE_MS`, `RHYTHM_HAPTIC_SAFETY_MS` (speed and motor from `../config.py`) |
| Cues | `WARNING_LEAD_BEATS`, `UPBEAT_GLOW_COLOR`, `ARRIVAL_HINT`, `ARRIVAL_HINT_BEATS` |
| Tempo change indicator | `TEMPO_JUMP_MIN_BPM`, `TEMPO_SUDDEN_WITHIN_BEATS`, `TEMPO_LED_*` (incl. `TEMPO_LED_SLOW_COLOR`) |
| Look | `TABLE_*`, `BALL_*` colours (white ball on a blue table), opponent paddle colours, `PLAIN_COLORS` (backboard, floor, panels), fonts |
| Paddle twist (visual) | `ROLL_DISPLAY_ENABLED`, `ROLL_DISPLAY_GAIN`, `ROLL_DISPLAY_DEADZONE_DEG`, `ROLL_DISPLAY_RECENTER_S`, `ROLL_DISPLAY_SWING_HOLD_S` |
| Chart helper | `CHART_BPM_RANGE`, `CHART_WINDOW_S`, `CHART_WINDOW_HOP_S`, `CHART_MIN_SEGMENT_S`, `CHART_MERGE_BPM`, `CHART_RUN_PURITY`, `CHART_CHANGE_FIT`, `CHART_BARLINE_SNAP_MS`, `CHART_INTRO_BARS`, `CHART_OUTRO_BARS`, `CHART_LOUD_FRACTION`, `PREVIEW_*` |

## Files
| File | What it does |
|---|---|
| `rhythm_main.py` | entry point: devices, window, key handling, frame loop, `HIT_BEATS` check |
| `rhythm_game.py` | game flow and play logic (no pygame): menus, game modes, songs, judging, tempo ticks, calibration |
| `audio_engine.py` | sounddevice mixer + song clock (with seeking); pygame and null fallbacks; audio loading |
| `tempo.py` | the tempo map (every beat ↔ time conversion), sudden-change detection, the LED state |
| `chart.py` | chart format: bars, patterns / sections, `HIT_BEATS`; load / validate / save / scan |
| `scheduler.py` | the timing rule (opponent halfway between your hits); ball / opponent position at any song time (bent to your paddle in Follow mode) |
| `judge.py` | timing (+ aim in Aim mode) → Perfect / Good / Miss |
| `scoring.py` | accuracy, streaks, best scores, the `emit()` hook |
| `calibration.py` | offset analysis, saving `calibration.json` |
| `rhythm_motor.py` | `RhythmMotor` / `RhythmSwingDetector` (subclasses of `../motor_imu.py`), cooldown and haptic checks |
| `sounds.py` | short synthesised effects: clicks, warning cues, tempo tick, hit sound (no music) |
| `plain.py` | the plain backdrop (backboard + floor) and the opponent's floating paddle |
| `rhythm_render.py` | draws the scene, HUD, tempo LED and every screen |
| `make_chart.py` | `detect` (tempo map + bars → chart, `--to-wav`), `preview` (live click track, nudge and save), `clicks` (click track to a file) |

## Credits
All eight songs are by Kevin MacLeod (incompetech.com) and are licensed under Creative Commons:
By Attribution 4.0 License (http://creativecommons.org/licenses/by/4.0/). The same credit is
stored in each chart's `attribution` field.

- "Electrodoodle" Kevin MacLeod (incompetech.com)
  Licensed under Creative Commons: By Attribution 4.0 License
  http://creativecommons.org/licenses/by/4.0/
- "Latin Industries" Kevin MacLeod (incompetech.com)
  Licensed under Creative Commons: By Attribution 4.0 License
  http://creativecommons.org/licenses/by/4.0/
- "Spazzmatica Polka" Kevin MacLeod (incompetech.com)
  Licensed under Creative Commons: By Attribution 4.0 License
  http://creativecommons.org/licenses/by/4.0/
- "Cut and Dry" Kevin MacLeod (incompetech.com)
  Licensed under Creative Commons: By Attribution 4.0 License
  http://creativecommons.org/licenses/by/4.0/
- "Itty Bitty 8 Bit" Kevin MacLeod (incompetech.com)
  Licensed under Creative Commons: By Attribution 4.0 License
  http://creativecommons.org/licenses/by/4.0/
- "Kick Shock" Kevin MacLeod (incompetech.com)
  Licensed under Creative Commons: By Attribution 4.0 License
  http://creativecommons.org/licenses/by/4.0/
- "Theme for Harold var 3" Kevin MacLeod (incompetech.com)
  Licensed under Creative Commons: By Attribution 4.0 License
  http://creativecommons.org/licenses/by/4.0/
- "Pinball Spring 160" Kevin MacLeod (incompetech.com)
  Licensed under Creative Commons: By Attribution 4.0 License
  http://creativecommons.org/licenses/by/4.0/

## MQTT later
When you want scores on MQTT, add a listener where the game creates its `ScoreKeeper` (in
`RhythmGame.start_song`):
```python
from mqtt_client import ScorePublisher           # ../mqtt_client.py (uses ../../mqttlib.py)
publisher = ScorePublisher()                     # create and start() it once, e.g. in rhythm_main.py
publisher.start()
...
self.score.listeners.append(lambda ev: publisher.publish_score(ev.accuracy))
```
Each event has `kind` ("cue" / "song_end"), `accuracy`, `streak`, `max_streak`, `counts`, `grade`
and `error_ms`.
