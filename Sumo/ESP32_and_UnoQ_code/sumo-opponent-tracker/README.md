# 🤖 Sumo Opponent Tracker

This app watches the camera and looks for the other robot. When it finds it, the
lights on the UNO Q show where it is and how close it is.

## Step 1: Get on the right Wi-Fi

The camera and the UNO Q both need to be on the **same Wi-Fi network**, or they
can't talk to each other.

- If the camera is on your normal Wi-Fi (home Wi-Fi, a hotspot) - great, nothing
  to do. Your computer can stay on the internet, and you can still chat with
  Claude Code while you work.
- If the camera made its own Wi-Fi called **SUMOCAM** (password `sumobot123`) -
  you need to connect the UNO Q to SUMOCAM too. **SUMOCAM has no internet.** If
  your computer also joins SUMOCAM (so you can see the tuning page below), your
  computer loses internet too, and **Claude Code chat stops working** until you
  switch back. So: do all your tuning in one sitting while on SUMOCAM, then
  switch back to normal Wi-Fi afterward if you want to chat again.

**Important: every time you change Wi-Fi networks, turn this app off and on
again** (the Stop and Run buttons in App Lab). Otherwise the website address
below will be wrong.

## Step 2: Find the website address

This app has its own website where you can watch the camera and tune things.
The address is different every time you're on a different Wi-Fi network.

To find it: look at App Lab's screen for this app, or look at the app's log
text for a line like this:

```
Network URL: http://192.168.4.2:7000
```

Type that address into a web browser, **on a device connected to the same
Wi-Fi as the UNO Q**. If you're on SUMOCAM and can't check the log, try
`http://192.168.4.2:7000` - that's almost always the right one on SUMOCAM.

Typing `localhost:7000` never works from your own computer - that always means
"look on this computer," not the UNO Q.

If the page just says "loading" forever, your computer and the UNO Q are
probably not on the same Wi-Fi network.

## Step 3: Tune the colors (the fast way!)

There are no sliders to fiddle with for this. The app looks for 4 colors:
**azure** (light blue), **pink**, **white**, and **wheel** (dark gray, for the
wheels). It also needs to know the **floor** color and the **tape** color, so
it can ignore those.

**The fast way to set a color:**
1. Hold the LEGO piece (or point the camera at the floor, or the tape) in
   front of the camera.
2. On the website, **click that spot on the live picture**. A small orange box
   jumps there.
3. Click the matching button below the picture, like **"Set azure here"**.
4. Done! Look at the black-and-white picture next to it (called the **mask**) -
   the spot you picked should now turn white there.

Do this once for each color: azure, pink, white, wheel, floor, and tape.

**If one click isn't quite enough:** click a *different* spot on the same
color (like the shadowed side of the same LEGO piece) and press its button
again. It doesn't start over - it *widens* what it already learned to also
cover the new spot. A couple of clicks per color usually nails it.

**Picked a bad spot by accident?** Click **"Reset learned colors"** to wipe
everything back to the starting point and try again.

**A tip:** your floor is light/white colored, so the "white" opponent color
will always look a bit messy, because it overlaps with the floor. That's okay -
the azure, pink, and wheel colors (plus the tape-rejecting and floor-rejecting
rules) do most of the work.

**If the mask picture looks like a mess of white everywhere** and you can't
tell what's going on: use the dropdown above the mask picture to pick one
single color instead of "combined" - that isolates exactly which one color is
lighting up too much, so you know which one to re-click/reset.

## Step 3b: Set the horizon and own-robot lines

Same click-then-press idea:
1. Click the spot on the live picture where the horizon is (where the wall
   meets the floor, or wherever you want "ignore everything above here" to
   start).
2. Click **"Set horizon line here."**
3. Do the same for the bottom of the view if your own robot pokes into the
   picture, using **"Set own-robot line here."**

## Step 4: Check it's working

- Point the camera at just the blue tape (no robot). The picture should say
  **visible=False** - it should ignore the tape.
- Bring a robot (or LEGO pieces) into view. It should say **visible=True**,
  with a green box drawn around it.

## Step 5: Tell it how far away things are (distance calibration)

Do this after the camera is mounted in its final spot - if you move the
camera later, do this step again.

1. Put a robot (or LEGO pieces) at a measured distance from the camera, like
   10 cm.
2. Type `10` into the box on the website and click **"Capture point at this
   distance."**
3. Move it farther away - like 20 cm, then 30, then 50, then 75 - and click
   Capture each time.

More distances = a more accurate guess later. Until you've captured at least
2 points, the distance will show as "?" - that's normal, not broken.

## Step 6: Save your work

Click **"Save settings to config.json"**. If you don't click this, your colors
and tuning are lost the next time the app restarts.

## Other things on the website

- **Record current frame**: saves a picture from the camera to a folder
  called `recordings`.
- **Replay**: next to each saved picture, this runs the detector on that old
  picture again - handy for testing without the camera even being on.
- **Advanced settings** (collapsed by default): cleanup and rejection
  settings with sensible starting values. You shouldn't need to open this for
  normal tuning - it's there in case something still needs a nudge after
  using the click-based tools above.

## What the colors on the picture mean

- 🟩 **Green box** = the robot it currently sees.
- 🟨 **Yellow line** = the horizon. Anything above this line is ignored
  (walls, people, ceiling).
- 🟪 **Magenta line** = anything below this line is ignored (that's where your
  own robot might peek into the bottom of the camera's view).
- 🟧 **Orange box** = the spot that will be used when you click a "Set ___
  here" button.

## Talking to the laptop (MQTT)

This app is the robot's "eyes and voice" - it watches the camera and shows
status, but **driving the motors, the color sensor, the controller, and the
actual behavior tree all live in the separate laptop program.** The two only
ever talk to each other over MQTT, through the public broker
`test.mosquitto.org:1883`, using the messages below. Never compare a
timestamp from one device against another device's clock - everything here
uses `frame_age_ms` plus the *receiving* device's own clock to judge
freshness instead.

All the topic names and MQTT settings live together near the top of
`python/main.py`, under `# ===================== TUNABLE: MQTT contract`.

### Buzzer wiring

| Buzzer pin | Goes to |
|---|---|
| S (signal) | D5 |
| + | 5V |
| − | GND |

The buzzer code (non-blocking 5-note melody, `tone()`/`noTone()`) is adapted
from the "Buzzer Melody" app in this same App Lab install - see that app for
the original standalone version this was built on. It plays a descending
G4-F4-E4-D4-C4 scale, one note per second, and **never plays on startup** -
only when `event/lost` arrives.

### MQTT contract

| Topic | Direction | Payload (JSON) |
|---|---|---|
| `ME193/Rogers/RQ-D2/vision/opponent` | UNO Q → laptop | `{"seq": int, "visible": bool, "x": float -1..1 or null, "distance_cm": float or null, "confidence": 0..1, "frame_age_ms": int}` (plus an extra `"source": "camera"` field beyond the strict contract, harmless to ignore) |
| `ME193/Rogers/RQ-D2/unoq/status` | UNO Q → laptop | `{"state": "online", "camera": "ok"/"missing", "fps": float}` |
| `ME193/Rogers/RQ-D2/laptop/status` | laptop → UNO Q | `{"state": "online"}` |
| `ME193/Rogers/RQ-D2/robot/state` | laptop → UNO Q | `{"phase": "armed"/"running"/"lost", "branch": "reverse"/"chase"/"teleop"/"idle"/null}` |
| `ME193/Rogers/RQ-D2/event/lost` | laptop → UNO Q | `{"reason": str}` |
| `ME193/Rogers/RQ-D2/event/song_done` | UNO Q → laptop | `{}` |

`vision/opponent` is published for **every** processed camera frame (roughly
12/sec - see `STREAM_FPS`). `unoq/status` is published once a second.

**A note on `mqttlib.py`:** it was provided exactly as specified and is not
modified - it doesn't support QoS, retain, or a Last Will. That means this
app's own publishes go out as plain, non-retained messages (the once-a-second
`unoq/status` heartbeat is what lets the laptop tell "alive" from "not," in
place of a Last Will). Reading `laptop/status` works fine regardless of
whatever QoS/retain/LWT settings the laptop side uses to publish it. Like
RQ-D2 Vision, this app treats the laptop as "online" whenever the most
recent `laptop/status` message said `"online"` (no timeout).

**A heads-up:** a different app on this same board, "RQ-D2 Vision" (an
earlier ultrasonic-sensor-based version of this exact same robot), publishes
to and subscribes from these *exact same topic names*. Since only one App
Lab app runs at a time this isn't an active conflict, but make sure **this**
app (Sumo Opponent Tracker) is the one with "Run at startup" turned on, not
that one, going forward.

## Debug tools (work without the laptop program running)

Both live in `tools/` and use this app's own `mqttlib.py` - run them from any
computer with internet access (they don't need App Lab or the board):

- **`python tools/mqtt_monitor.py`** - subscribes to every topic in the
  contract and prints every message as it arrives, with a timestamp. Useful
  for confirming the robot is actually publishing (and at what rate).
- **`python tools/fake_robot.py`** - an interactive menu that publishes fake
  `robot/state` and `event/lost` messages, so you can test every LED icon and
  the full lost → song → song_done sequence without the laptop's behavior
  tree running at all.

Both need `paho-mqtt` installed on whatever computer runs them (`pip install
paho-mqtt`).

## LED matrix icons

Checked in priority order, top to bottom - e.g. a dead camera always shows
the X, even if everything else about the robot's state is fine.

| Situation | Icon |
|---|---|
| ESP32-CAM stream missing/stale | X |
| Can't reach the MQTT broker | circle with a slash |
| Laptop not connected (`laptop/status` missing, stale, or not "online") | "..." (three dots) |
| `robot/state.phase == "lost"`, or `event/lost` just arrived | sad face (shown immediately; held through the whole song) |
| `robot/state.phase == "armed"` | hollow diamond |
| `robot/state.branch == "reverse"` | left arrow |
| `robot/state.branch == "chase"` | the camera's own live display: a box that grows as the opponent gets closer and moves left/right with it, or the searching sweep if nothing's currently visible |
| `robot/state.branch == "teleop"` | controller silhouette |
| `robot/state.branch == "idle"` (or no branch yet) | single centered dot |

## What I already verified myself

Before handing this off:
- The sketch compiles and the app starts cleanly with no errors (checked via
  the board's own logs after every change).
- MQTT actually connects to the real public broker (`test.mosquitto.org`) and
  logs a real measured round-trip time through it every 5 seconds.
- `unoq/status` publishes correctly and the console status line reflects
  `mqtt_connected`/`laptop_online` accurately with no laptop running.
- Camera discovery/reconnect, the debug web page, and all the detection
  tuning tools from before still work exactly as before - none of that was
  touched by this change.

**What I could not fully verify myself** (I don't have a way to run an MQTT
client to simulate the laptop from where I work, and I can't see the
physical LED matrix or hear the buzzer): whether `robot/state` correctly
drives each LED icon, and the full `event/lost` → sad face → song → buzzer →
`song_done` sequence. The checklist below covers exactly those - please run
through it once.

## Final verification checklist

1. **Camera/detection still works**: redo the Step 4 check above (tape alone
   = not visible, robot in view = visible with a green box). This confirms
   nothing in this MQTT change broke the existing detector.
2. **MQTT monitor sees vision data**: run `python tools/mqtt_monitor.py` on a
   computer with internet access and confirm `vision/opponent` messages are
   arriving at roughly the expected rate, and `unoq/status` once a second.
3. **Every LED icon**: run `python tools/fake_robot.py` and walk through each
   menu option one at a time, confirming the matrix shows the right icon for
   each (see the table above). Try "Running / chase" at a few different
   distances to see the box grow/shrink.
4. **Lost → song → song_done**: from `fake_robot.py`, trigger `event/lost`
   and confirm: the sad face appears immediately, the buzzer plays all 5
   descending notes (~5 seconds total, G4-F4-E4-D4-C4), and
   `mqtt_monitor.py` shows `event/song_done` arriving right after the last
   note.
5. **Laptop presence**: with `fake_robot.py` connected, send `laptop/status`
   → online and confirm the matrix leaves the "..." icon; send `laptop/status`
   → offline and confirm it returns to "...".
6. **No internet**: temporarily disconnect the UNO Q's Wi-Fi (or block
   internet while staying on the local network, if you can) and confirm the
   matrix shows the circle-with-a-slash icon, then reconnect and confirm it
   recovers on its own.
7. **Power cycle with no laptop connected**: make sure "Run at startup" is
   turned on for *this* app (not RQ-D2 Vision - see the heads-up above).
   Unplug everything except power to the UNO Q (no laptop, no USB data
   connection), power it on, and confirm it comes up on its own: camera
   discovery runs, the matrix eventually settles on the "..." (waiting for
   laptop) icon once MQTT connects, and `vision/opponent` starts publishing
   once the camera sees something - all with nobody's laptop involved.
