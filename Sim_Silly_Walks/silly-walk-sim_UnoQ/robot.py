"""Control the walking robot from the UNO Q's own terminal (no web browser needed).

The app must be running first:  arduino-app-cli app start ~/ArduinoApps/silly-walk-sim

    python3 ~/ArduinoApps/silly-walk-sim/robot.py start            # PWM scale 0.5, 10 s
    python3 ~/ArduinoApps/silly-walk-sim/robot.py start 1.0 10     # full power (= sim), 10 s
    python3 ~/ArduinoApps/silly-walk-sim/robot.py stop
    python3 ~/ArduinoApps/silly-walk-sim/robot.py status

`start` waits and prints the status until the run is over and the log is saved (Ctrl+C = STOP).
It talks to the app's small web server on this board (localhost:7000), the same one the web page uses.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:7000/api"


def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=8) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return json.load(e)
    except urllib.error.URLError:
        sys.exit("Can't reach the app. Is it running?  arduino-app-cli app start ~/ArduinoApps/silly-walk-sim")


def start(scale=0.5, seconds=10):
    before = get("/status").get("last_result", "")
    res = get(f"/start?scale={scale}&timeout={seconds}")
    if not res.get("ok"):
        sys.exit(f"Could not start: {res.get('error')}")
    print(f"Started: PWM scale {scale}, {seconds} s. KEEP THE ROBOT STILL for 1 s. Ctrl+C = STOP.")
    last = None
    try:
        while True:
            s = get("/status")
            if s["state"] != last:
                print("  state:", s["state"])
                last = s["state"]
            if s["last_result"] and s["last_result"] != before and s["state"] == "idle":
                print(s["last_result"])
                print("Log file: ~/ArduinoApps/silly-walk-sim/logs/" + s["log_file"])
                return
            time.sleep(0.3)
    except KeyboardInterrupt:
        get("/stop")
        print("\nSTOP sent.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "start":
        start(float(sys.argv[2]) if len(sys.argv) > 2 else 0.5, int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    elif cmd == "stop":
        print(get("/stop"))
    else:
        print(get("/status"))
