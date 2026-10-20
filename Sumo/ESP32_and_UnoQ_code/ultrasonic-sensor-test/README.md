# 📏 Ultrasonic Sensor Test

Standalone test: shows live HC-SR04 distance as a growing/shrinking box on the LED matrix and logs it to the console - no MQTT, no laptop, no behavior tree needed. Confirms sensor wiring and lets you see the detection range live.

Same wiring as "RQ-D2 Vision": Trig -> D9, Echo -> D8 (through the voltage divider - see that app's README), VCC -> 5V, GND -> GND.

## What it shows

- **Single dot**: nothing within 100cm (the same ring-diameter default as the real app).
- **Growing box**: something's within 100cm - the box gets bigger as it gets closer, same scheme as "RQ-D2 Vision"'s "chasing" icon.
- **X**: the sensor itself has stopped responding (~6 seconds of no valid echo at all).

Distance/detected/sensor-health are also logged to the console once a second (visible in App Lab's log view for this app), in case you'd rather read numbers than watch the matrix.

## Using it

Starting this app stops whatever else is running (only one App Lab app runs at a time) - when you're done testing, switch back to "RQ-D2 Vision" the same way, and re-enable "Run at startup" on it if you'd set that before.

All the detection numbers (100cm range, 5cm minimum, etc.) are hardcoded constants at the top of `sketch/sketch.ino`, matching "RQ-D2 Vision"'s defaults - this is a quick bench-test tool, not meant to be tuned from Python, so edit the `.ino` directly and re-upload if you want to try different numbers.

