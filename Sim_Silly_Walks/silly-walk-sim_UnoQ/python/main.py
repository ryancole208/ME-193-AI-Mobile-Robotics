# Silly Walk Sim - Linux (MPU) side: web page with Start / STOP + saving the run logs.
#
# The policy itself runs on the STM32 (sketch). This side only:
#   * serves the control page (assets/index.html) at http://<board-ip>:7000
#   * forwards Start (with PWM scale + timeout) and STOP to the sketch over the Bridge
#   * receives the log rows after each run and writes them to logs/run_<date>_<time>.csv
#     (= ~/ArduinoApps/silly-walk-sim/logs/ on the board; download them from the page or copy them off)
import os
import re
import threading
import time

from fastapi.responses import FileResponse, JSONResponse

from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge, Logger

logger = Logger("silly-walk")
LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)

CSV_HEADER = ("step,t_s,obs_wx,obs_wy,obs_wz,obs_sin_yaw,obs_cos_yaw,obs_prev_act_L,obs_prev_act_R,"
              "act_L,act_R,duty_L,duty_R,roll,pitch,tilt,yaw,compute_us")

lock = threading.Lock()
status = {"state": "starting", "last_result": "", "log_file": ""}
current = {"file": None, "name": "", "rows": 0}


# ---- messages from the sketch ----
def on_state(text):
    with lock:
        status["state"] = text
    logger.info(f"sketch state: {text}")


def on_log_line(line):
    with lock:
        if current["file"]:
            current["file"].write(line + "\n")
            current["rows"] += 1


def on_run_done(reason, rows):
    with lock:
        f, name, got = current["file"], current["name"], current["rows"]
        if f:
            f.close()
        current["file"] = None
        msg = f"run ended: {reason} - {got} of {rows} rows saved to {name}"
        if got != rows:
            msg += " (WARNING: rows missing)"
        status["last_result"] = msg
        status["log_file"] = name
    logger.info(msg)


Bridge.provide("state", on_state)
Bridge.provide("log_line", on_log_line)
Bridge.provide("run_done", on_run_done)


# ---- web API ----
def api_start(scale: float = 0.5, timeout: int = 10):
    scale_pct = int(round(max(0.05, min(1.0, scale)) * 100))
    timeout = max(1, min(30, int(timeout)))
    name = time.strftime("run_%Y%m%d_%H%M%S.csv")
    with lock:
        if current["file"]:
            return JSONResponse({"ok": False, "error": "a run is still being saved"}, status_code=409)
        f = open(os.path.join(LOG_DIR, name), "w")
        f.write(f"# pwm_scale={scale_pct / 100:.2f} timeout_s={timeout} started={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(CSV_HEADER + "\n")
        current.update(file=f, name=name, rows=0)
    try:
        ok = Bridge.call("start", scale_pct, timeout, timeout=5)
    except Exception as e:   # sketch not answering
        ok, err = False, str(e)
    else:
        err = "" if ok else "sketch refused (busy, or IMU not found)"
    if not ok:
        with lock:
            current["file"].close()
            os.remove(os.path.join(LOG_DIR, name))
            current["file"] = None
        logger.warning(f"start failed: {err}")
        return JSONResponse({"ok": False, "error": err}, status_code=409)
    logger.info(f"start: scale {scale_pct} %, timeout {timeout} s -> {name}")
    return {"ok": True, "log_file": name}


def api_stop():
    try:
        Bridge.call("stop", timeout=2)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True}


def api_status():
    with lock:
        return dict(status)


def api_logs():
    files = sorted((f for f in os.listdir(LOG_DIR) if f.endswith(".csv")), reverse=True)
    return {"files": files}


def api_log_file(name: str):
    if not re.fullmatch(r"run_\d{8}_\d{6}\.csv", name) or not os.path.exists(os.path.join(LOG_DIR, name)):
        return JSONResponse({"error": "no such log"}, status_code=404)
    return FileResponse(os.path.join(LOG_DIR, name), media_type="text/csv", filename=name)


web_ui = WebUI()
web_ui.expose_api("GET", "/api/start", api_start)
web_ui.expose_api("GET", "/api/stop", api_stop)
web_ui.expose_api("GET", "/api/status", api_status)
web_ui.expose_api("GET", "/api/logs", api_logs)
web_ui.expose_api("GET", "/api/logs/{name}", api_log_file)

App.run()
