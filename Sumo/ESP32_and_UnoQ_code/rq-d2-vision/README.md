# 👁️ RQ-D2 Vision

Ultrasonic "eyes" and sad-song "voice" for the RQ-D2 sumo robot: measures the distance to the nearest object in front of the robot with an HC-SR04, publishes it over MQTT for the laptop's behavior tree, shows the robot's status on the UNO Q's LED matrix, and plays a sad song when told to.

This app only handles sensing/status/sound. Driving the LEGO motors, reading the color sensor, the controller, and the behavior tree itself all live in the **separate laptop program** - this app and that program only ever talk over MQTT, using the contract below.

## Wiring

| HC-SR04 pin | Goes to |
|---|---|
| VCC | 5V |
| GND | GND (shared with the UNO Q) |
| Trig | D9 |
| Echo | **See "Echo voltage protection" below** |

| Buzzer pin | Goes to |
|---|---|
| S (signal) | D5 |
| + | 3.3V rail (confirmed with Ryan) |
| − | GND |

Sensor is mounted low, ~3-5cm above the floor, facing forward (confirmed with Ryan) - see "Known blind spots" below for what that implies.

### Echo voltage protection

The HC-SR04's Echo pin outputs ~5V. The UNO Q's digital pins run 3.3V logic. **I could not independently confirm from the official datasheet whether D8 is 5V-tolerant**, because this build was done without internet access (web lookups were turned off for this task on your instruction). Rather than guess on a question that could damage the board, this build assumes D8 is **not** confirmed safe and the wiring instructions below include the voltage divider as a precaution:

```
Echo ──┬── 1 kΩ ──┬── D8
       │          │
      (5V         2 kΩ
     pulses)       │
                  GND
```

That is: Echo → 1kΩ resistor → D8, and D8 → 2kΩ resistor → GND. This divides the ~5V pulse down to ~3.3V before it reaches D8, at the cost of two cheap resistors and five minutes of breadboarding.

**Please check the official UNO Q datasheet/pinout yourself before your competition** - if D8 turns out to be genuinely 5V-tolerant, you can wire Echo straight to D8 with no divider (the simpler arduinogetstarted.com-style wiring you mentioned). Nothing in the code needs to change either way; the divider only affects the voltage level on the wire, not the sketch.

## How detection works

All of the measuring and decision-making happens in the **sketch** (`sketch/sketch.ino`), because it needs microsecond-accurate timing that Python can't reliably provide:

1. **Ping**: every 60ms (~16.7 Hz, within the assignment's 15-20Hz target and respecting the HC-SR04's minimum 60ms-between-pings requirement), the sketch sends a 10µs pulse on Trig and times the Echo pulse by hand with `micros()` + `digitalRead()`, with a 25ms timeout (~4.3m round trip) checked on every polling iteration. I tried `pulseIn(ECHO_PIN, HIGH, 25000)` first - it compiles fine on this board's `arduino:zephyr` core - but testing with no sensor connected (Echo permanently LOW) showed it does **not** reliably honor its timeout: Bridge calls started timing out after 10s and ultrasonic updates stopped arriving entirely, meaning it was blocking far longer than 25ms waiting for a rising edge that never came. Since "Echo never responds" is exactly the disconnected/broken-sensor case this app needs to detect rather than hang on, I switched to the manual `micros()` polling loop instead, which bounds the worst case strictly by hand.
2. **Filter**: each valid reading is pushed into a 5-slot rolling buffer; the published distance is the **median** of whatever's in that buffer, which rejects the odd wild spike a raw HC-SR04 reading occasionally produces (including interference from other robots' ultrasonic sensors).
3. **Detect** ("is something in the ring ahead of us?"):
   - Below `RING_DIAMETER_CM` (default 100cm, the ring's diameter) -> candidate for "detected".
   - Once detected, it has to retreat past `RING_DIAMETER_CM + HYSTERESIS_MARGIN_CM` (default 115cm) before being called "lost" again - this hysteresis gap stops an opponent sitting right at the ring's edge from flickering in and out of "detected" on every small sensor wobble.
   - Either transition only actually happens after `DEBOUNCE_COUNT` (default 3) consecutive readings agree, which is one more layer of anti-flicker on top of the median filter.
   - Readings closer than `MIN_VALID_DISTANCE_CM` (default 5cm) are discarded as sensor noise (the HC-SR04's own blind zone, or an echo bouncing off the robot's own chassis), not treated as a real object.
4. **Confidence**: the fraction of the readings currently in the 5-slot buffer that land close to the median (within 15%, or 10cm, whichever's bigger). A steady reading gives confidence close to 1.0; a noisy/borderline one gives something lower.
5. **`x`**: always `0.0` when detected, `null` when not - a single ultrasonic sensor can only tell you *that* something's ahead, never *which side* it's on.

The sketch pushes each filtered reading to Python over the Bridge (`Bridge.notify`, not Python polling) rather than Python asking for it - this is simpler than polling. The ping/filter/detect bookkeeping above always runs at the full ~16.7Hz ping rate; **how often that gets sent to Python is throttled** to a ~10Hz heartbeat, except that a change in `visible` is always sent immediately regardless of the heartbeat - see "Known limitation" below for why.

### Bug found and fixed during testing: Bridge calls from the wrong thread

While bench-testing this build (no HC-SR04/buzzer wired yet), LED icon changes started occasionally failing with `TimeoutError("... timed out after 10s")`, and `vision/opponent` would stall for several seconds and then arrive in a sudden burst. The root cause: `_on_vision_update`, the function that receives the sketch's ultrasonic updates, runs on the Bridge library's own dispatch thread (visible in logs as `Bridge.read_loop`) - and it was calling `Bridge.call(...)` directly to redraw the LED icon whenever something changed. That self-deadlocks: the thread making the call is the same thread needed to receive that call's response, so it can only ever time out, never succeed, and every other Bridge message queues up behind it until it gives up.

**Fix:** all LED icon `Bridge.call()`s now happen only from `loop()` (the main thread), never from inside a `Bridge.provide()` callback. `_on_vision_update`, `_on_song_finished`, `_on_robot_state`, and `_on_laptop_status` now only update shared state and publish to MQTT; `loop()` re-evaluates the LED icon itself on every ~0.1s tick (cheap, since it only actually calls the sketch when the icon changes). The one exception is `_on_event_lost`, which runs on paho's MQTT thread (not the Bridge thread) and calls Bridge directly so the sad face appears instantly - that thread doesn't have this problem. See the comments in `python/main.py` for the full explanation.

As a second, smaller precaution, the sketch also throttles its `vision_update` notifications to a steady ~10Hz heartbeat instead of firing on every ~60ms ping (a detection change is always sent immediately regardless) - this isn't the fix for the deadlock above, just extra headroom against flooding the Bridge with more traffic than it needs.

**Please still watch for this once the real HC-SR04 and buzzer are wired up**, as part of the verification checklist below: change the robot's state a few times in a row via `fake_robot.py` while watching `vision/opponent` in `mqtt_monitor.py`, and confirm updates stay smooth with no multi-second stalls. I'm confident in the root cause and the fix, but I'd rather you confirm it on the real hardware than take my word for it untested.

### Known blind spots

- **Low/angled opponents**: the sensor is mounted low (~3-5cm) and fires a narrow forward cone. A very low, sloped, or angled opponent front can deflect the ping away instead of bouncing it straight back, so the robot may not "see" every opponent shape equally well even at close range.
- **No left/right information**: by design (a single ultrasonic sensor), `x` is never anything but `0.0` or `null`. If the opponent is centered on the ring but off to one side of the robot, this app will still report it as "detected straight ahead."
- **Sensor-vs-"nothing in range" ambiguity**: a genuinely disconnected/broken sensor and a sensor that's pointed at open space beyond ~4m look identical on the wire (both time out). This app only declares the sensor itself "not responding" (the X icon, `camera: "missing"`) after ~100 consecutive timeouts (~6 seconds) specifically so a brief clear view doesn't falsely trigger it - but it means a genuine hardware fault takes a few seconds to be flagged, not milliseconds.
- **Other robots' sensors**: the median filter helps reject occasional ultrasonic cross-talk between robots' sensors, but can't fully eliminate it if it's frequent.

## Startup: running with no computer connected

This app needs **"Run at startup"** turned on (the toggle next to the Run button in App Lab) so it starts on its own as soon as the UNO Q powers up, with no laptop needed.

**Only one App Lab app runs at startup at a time.** If you ever need to switch which app starts automatically (e.g. back to "Motor Test" for a wiring check), open that other app in App Lab and turn its own "Run at startup" toggle on - this automatically turns this app's toggle off, since the setting is exclusive to whichever app you enabled it on most recently.

## MQTT contract

Broker: `test.mosquitto.org:1883`. All topics below start with `ME193/Rogers/RQ-D2/`. **Never compare timestamps across devices** - the receiver judges freshness using its own clock plus the `frame_age_ms`/similar age field.

| Topic | Direction | QoS / retain | Payload (JSON) |
|---|---|---|---|
| `vision/opponent` | UNO Q → laptop | 0 | `{"seq": int, "visible": bool, "x": 0.0 or null, "distance_cm": float or null, "confidence": 0..1, "frame_age_ms": int, "source": "ultrasonic"}` |
| `unoq/status` | UNO Q → laptop | *see note below* | `{"state": "online", "camera": "ok"/"missing", "fps": float}` - `camera` reports whether the ultrasonic sensor is responding; `fps` is the measurement rate |
| `laptop/status` | laptop → UNO Q | *see note below* | `{"state": "online"}` |
| `robot/state` | laptop → UNO Q | 1, sent on change | `{"phase": "armed"/"running"/"lost", "branch": "reverse"/"chase"/"teleop"/"idle"/null}` |
| `event/lost` | laptop → UNO Q | 1 | `{"reason": str}` |
| `event/song_done` | UNO Q → laptop | 1 | `{}` |

**Note on QoS/retain/Last Will:** the spec calls for `unoq/status` and `laptop/status` to be retained with a Last Will of `offline`. `mqttlib.py` doesn't support any of QoS, retain, or Last Will, **and per your instruction it was kept unmodified** - so this app's own `unoq/status` publishes go out as plain, non-retained messages (published once a second instead, so the laptop sees a steady heartbeat while this app is alive). Reading `laptop/status` works fine regardless - subscribing doesn't require any special support, so if the laptop's own MQTT client does publish it retained/with QoS1/LWT, this app will receive it exactly the same as any other message.

If you ever want real retain/QoS/LWT support, the minimal change (discussed with Ryan, not applied because he preferred a different approach) would be: optional `qos=0`/`retain=False` keyword arguments on `publish()`, an optional `qos=0` keyword on `subscribe()`, and an optional `will=(topic, payload, qos, retain)` keyword on `MQTTClient.__init__` that calls `client.will_set(...)` before connecting - all backward-compatible with every existing call.

## Tuning the maximum range (and everything else)

Every tunable number lives in **`python/config.py`**. The ones that matter most for competition day:

- `RING_DIAMETER_CM` - the ring's diameter in cm. Set to `100.0` (confirmed with Ryan). Change this if you compete in a different-sized ring.
- `MIN_VALID_DISTANCE_CM` - readings closer than this are discarded as noise. Set to `5.0` (confirmed with Ryan, matches the sensor's mounting).
- `HYSTERESIS_MARGIN_CM` / `DEBOUNCE_COUNT` - how twitchy vs. steady the detected/not-detected flip is. Raise `DEBOUNCE_COUNT` if the LED/behavior tree seem to flicker; lower it if reactions feel sluggish.

After editing `config.py`, **restart the app** (Stop then Run in App Lab) - the new values are pushed into the sketch once, at startup.

Pin numbers (`PIN_TRIG`, `PIN_ECHO`, `PIN_BUZZER`) and the ping rate (`PING_INTERVAL_MS`) are also listed in `config.py` for visibility, but they're compiled into `sketch/sketch.ino` - changing them means editing the `.ino` file itself and re-uploading, not just editing `config.py`.

## Debug tools (work without the laptop program running)

Both live in `tools/` and use the exact same `mqttlib.py`/`config.py` as the app - run them from any computer with internet access (they don't need App Lab or the board):

- **`python tools/mqtt_monitor.py`** - subscribes to every topic in the contract and prints every message as it arrives, with a timestamp. Useful for confirming the robot is actually publishing (and at what rate).
- **`python tools/fake_robot.py`** - an interactive menu that publishes fake `robot/state` and `event/lost` messages, so you can test every LED icon and the full lost → song → song_done sequence without the laptop's behavior tree running at all.

## LED matrix icons

| Situation | Icon |
|---|---|
| Ultrasonic sensor not responding | X |
| No internet / can't reach the MQTT broker | circle with a slash |
| Laptop not connected (or `laptop/status` says offline) | "..." (three dots) |
| `robot/state.phase == "armed"` | hollow diamond |
| `robot/state.branch == "reverse"` | left arrow |
| `robot/state.branch == "chase"` | centered box, grows as the opponent gets closer |
| `robot/state.branch == "teleop"` | controller silhouette |
| `robot/state.branch == "idle"` (or no branch yet) | single centered dot |
| `robot/state.phase == "lost"`, or an `event/lost` just arrived | sad face (shown immediately; held through the song) |

Checked in priority order top-to-bottom - e.g. a dead sensor always shows the X, even if everything else is fine.

## Verifying everything works

1. **Hand test**: wave a hand slowly toward and away from the sensor (within ~0-150cm) and watch `python tools/mqtt_monitor.py`'s `vision/opponent` output - `distance_cm` should track the hand smoothly (no huge spikes), and `visible` should flip to `true` somewhere around 100cm and back to `false` past ~115cm, without rapid flickering right at that boundary.
2. **Nothing in range**: with a clear view and nothing within ~4m, confirm `visible: false`, `distance_cm` is either `null` or a large number depending on what the sensor sees (a distant wall is still a valid reading), and the LED matrix isn't stuck on the sensor-error X.
3. **Each LED icon**: use `python tools/fake_robot.py` to walk through every menu option one at a time and confirm the matrix shows the right icon for each (see the table above). Try the "chase" option at a few different hand distances to see the box grow/shrink.
4. **Lost → song → song_done**: from `fake_robot.py`, trigger "event/lost" and confirm: the sad face appears immediately, the buzzer plays all 5 descending notes (~5 seconds total), and `mqtt_monitor.py` shows `event/song_done` arriving right after the last note. While the song plays, also watch `vision/opponent` in `mqtt_monitor.py` (wave a hand in front of the sensor throughout) - see "Bug found and fixed during testing" above; confirm updates stay smooth throughout with no multi-second stalls.
5. **Power cycle with no computer connected**: unplug everything except power to the UNO Q (no laptop, no USB data connection), power it on, and confirm it comes up on its own showing the "waiting for laptop" icon (and starts publishing `vision/opponent` once you wave a hand in front of it) - this checks that "Run at startup" is actually on.
