# main.py  -  ESP32-CAM video streamer (version 2: multiple viewers)
#
# Saved on the board as main.py, this runs automatically at power-on.
# Up to 3 viewers can watch at once (e.g. the UNO Q AND your browser).
#
#   http://<camera-ip>/          -> web page showing the live video
#   http://<camera-ip>/stream    -> MJPEG video stream (what the UNO Q reads)
#   http://<camera-ip>/capture   -> one single JPEG photo
#   http://<camera-ip>/status    -> frames sent, FPS, viewers, free memory
#
# The camera's IP address is printed in Thonny's Shell when it starts.
# To stop it and get back to the >>> prompt in Thonny: press Ctrl+C.

import network
import socket
import time
import gc
from machine import Pin
from camera import Camera, PixelFormat, FrameSize, GrabMode

# ======================= SETTINGS YOU CAN CHANGE =======================

# "AP"  = always create the camera's own SUMOCAM network (camera at 192.168.4.1)
# "STA" = join an existing Wi-Fi network; if that fails, fall back to "AP"
WIFI_MODE = "STA"

STA_SSID = "Wifi"        # school Wi-Fi or phone hotspot (2.4 GHz)
STA_PASSWORD = "password"
STA_TIMEOUT_S = 20               # seconds per attempt
STA_ATTEMPTS = 3                 # attempts before falling back to AP

AP_SSID = "SUMOCAM"
AP_PASSWORD = "sumobot123"       # must be at least 8 characters
AP_CHANNEL = 6                   # 1, 6 or 11

HOSTNAME = "sumocam"             # may also be reachable as sumocam.local

FRAME_SIZE = "QVGA"    # QQVGA=160x120 (fastest), QVGA=320x240, VGA=640x480
JPEG_QUALITY = 60      # 0-100, higher = better picture but bigger/slower
VFLIP = False          # True if the camera is mounted upside down
HMIRROR = False        # True to mirror the image left/right

# Lighting
FLASH_MODE = "dim"     # front white LED: "off", "on" (full brightness) or "dim"
FLASH_DIM_PERCENT = 20 # LED brightness in "dim" mode, 1-100
BRIGHTNESS = 0         # image brightness, -2 (darker) to 2 (brighter)
AE_LEVEL = 0           # auto-exposure target, -2 to 2; higher = brighter picture

# Lock brightness/colour after a short warm-up so the picture doesn't change
# when an object enters. This makes "different from the empty scene"
# detection on the UNO Q much more reliable.
LOCK_EXPOSURE = True
WARMUP_S = 3

MAX_VIEWERS = 3

# =======================================================================

RED_LED = Pin(33, Pin.OUT)      # small red LED on the back, ON when value is 0
RED_LED.value(1)
FLASH_LED = Pin(4, Pin.OUT)     # keep the bright flash LED off
FLASH_LED.value(0)

AI_THINKER_PINS = dict(
    data_pins=[5, 18, 19, 21, 36, 39, 34, 35],
    vsync_pin=25, href_pin=23, sda_pin=26, scl_pin=27,
    pclk_pin=22, xclk_pin=0, powerdown_pin=32, reset_pin=-1,
)

stats = {"frames": 0, "fps": 0.0, "wifi": "", "ip": ""}


def led(on):
    RED_LED.value(0 if on else 1)


def blink(times, ms=120):
    for _ in range(times):
        led(True); time.sleep_ms(ms); led(False); time.sleep_ms(ms)


# ----------------------------- Wi-Fi -----------------------------------

def start_ap():
    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    sec = getattr(network.WLAN, "SEC_WPA2", None)
    if sec is None:
        sec = getattr(network, "AUTH_WPA2_PSK", 3)
    try:
        ap.config(ssid=AP_SSID, key=AP_PASSWORD, channel=AP_CHANNEL, security=sec)
    except Exception:
        ap.config(essid=AP_SSID, password=AP_PASSWORD, channel=AP_CHANNEL, authmode=sec)
    while not ap.active():
        time.sleep_ms(100)
    stats["wifi"] = "AP " + AP_SSID
    print("Created Wi-Fi network '{}' (password: {})".format(AP_SSID, AP_PASSWORD))
    return ap.ifconfig()[0]


def wifi_status_name(status):
    for attr in dir(network):
        if attr.startswith("STAT_") and getattr(network, attr) == status:
            return attr
    return str(status)


def start_sta():
    try:
        network.hostname(HOSTNAME)
    except Exception:
        pass
    network.WLAN(network.AP_IF).active(False)   # make sure SUMOCAM is off
    sta = network.WLAN(network.STA_IF)
    sta.active(True)
    sta.disconnect()                            # clear any half-finished attempt
    time.sleep_ms(500)

    for attempt in range(1, STA_ATTEMPTS + 1):
        print("Joining Wi-Fi '{}' (attempt {} of {}) ...".format(STA_SSID, attempt, STA_ATTEMPTS))
        sta.connect(STA_SSID, STA_PASSWORD)
        start = time.ticks_ms()
        last = None
        while not sta.isconnected():
            status = sta.status()
            if status != last:
                print("  status:", wifi_status_name(status))
                last = status
            if time.ticks_diff(time.ticks_ms(), start) > STA_TIMEOUT_S * 1000:
                break
            blink(1, 100)
        if sta.isconnected():
            try:
                sta.config(pm=sta.PM_NONE)      # no power saving = lower latency
            except Exception:
                pass
            secs = time.ticks_diff(time.ticks_ms(), start) / 1000
            print("Connected in {:.1f} s".format(secs))
            stats["wifi"] = "STA " + STA_SSID
            return sta.ifconfig()[0]
        print("  gave up on this attempt, last status:", wifi_status_name(last))
        sta.disconnect()
        time.sleep_ms(1000)

    sta.active(False)
    return None


def start_wifi():
    if WIFI_MODE == "STA":
        ip = start_sta()
        if ip:
            return ip
        print("Could not join '{}'. Falling back to own network.".format(STA_SSID))
    return start_ap()


# ----------------------------- Camera ----------------------------------

def set_prop(cam, name, value, quiet=False):
    try:
        setattr(cam, name, value)                # newer firmware: properties
        return True
    except Exception:
        try:
            getattr(cam, "set_" + name)(value)   # older firmware: set_ methods
            return True
        except Exception as e:
            if not quiet:
                print("Could not set", name, ":", e)
            return False


def start_camera():
    base = dict(
        pixel_format=PixelFormat.JPEG,
        frame_size=getattr(FrameSize, FRAME_SIZE),
        jpeg_quality=JPEG_QUALITY,
    )
    attempts = (
        dict(fb_count=2, grab_mode=GrabMode.LATEST),
        dict(fb_count=1, grab_mode=GrabMode.LATEST),
        dict(fb_count=1, grab_mode=GrabMode.WHEN_EMPTY),
    )
    last_error = None
    for use_pins in (False, True):
        for extra in attempts:
            args = dict(base)
            args.update(extra)
            if use_pins:
                args.update(AI_THINKER_PINS)
            try:
                cam = Camera(**args)
                set_prop(cam, "vflip", VFLIP)
                set_prop(cam, "hmirror", HMIRROR)
                print("Camera ready:", FRAME_SIZE, extra)
                return cam
            except Exception as e:
                last_error = e
                gc.collect()
    raise RuntimeError("Camera failed to start: {}".format(last_error))


def setup_lighting(cam):
    """Turn on the front LED and apply brightness settings before warm-up,
    so the exposure lock happens with the light already on."""
    if FLASH_MODE == "on":
        FLASH_LED.value(1)
        print("Flash LED: on (full)")
    elif FLASH_MODE == "dim":
        try:
            from machine import PWM
            pct = max(1, min(100, FLASH_DIM_PERCENT))
            PWM(FLASH_LED, freq=1000, duty_u16=int(65535 * pct / 100))
            print("Flash LED: dim ({}%)".format(pct))
        except Exception as e:
            print("Could not dim the flash LED ({}); using full brightness".format(e))
            FLASH_LED.value(1)
    else:
        FLASH_LED.value(0)
        print("Flash LED: off")

    set_prop(cam, "brightness", BRIGHTNESS)
    set_prop(cam, "ae_level", AE_LEVEL)

    # Make sure the camera still works with the LED settings applied
    time.sleep_ms(200)
    img = cam.capture()
    if img:
        release(cam)
    else:
        print("WARNING: camera stopped giving pictures after the LED was set up.")
        print("If the video doesn't work, set FLASH_MODE = \"on\" instead of \"dim\".")


def warm_up_and_lock(cam):
    if not LOCK_EXPOSURE:
        return
    print("Warming up camera for {} s (keep the scene as it normally looks)...".format(WARMUP_S))
    end = time.ticks_add(time.ticks_ms(), WARMUP_S * 1000)
    while time.ticks_diff(end, time.ticks_ms()) > 0:
        if cam.capture():
            release(cam)
    ok = 0
    for name in ("exposure_ctrl", "gain_ctrl", "whitebal"):
        if set_prop(cam, name, False, quiet=True):
            ok += 1
    print("Exposure/gain/white balance locked ({} of 3 settings applied)".format(ok))


def release(cam):
    try:
        cam.free_buffer()
    except Exception:
        pass


# ----------------------------- Web server ------------------------------

PAGE = """<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width">
<title>SumoCam</title><style>body{background:#111;color:#eee;font-family:sans-serif;text-align:center}
img{width:100%;max-width:640px;image-rendering:pixelated}</style></head>
<body><h2>SumoCam live</h2><img src="/stream"><p><a href="/status" style="color:#8cf">status</a></p></body></html>"""

STREAM_HEADER = (b"HTTP/1.1 200 OK\r\n"
                 b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                 b"Cache-Control: no-cache, no-store\r\n"
                 b"Connection: close\r\n\r\n")


def send_text(cl, code, ctype, body):
    head = "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n"
    cl.sendall(head.format(code, ctype, len(body)).encode())
    cl.sendall(body.encode() if isinstance(body, str) else body)


def handle_new(cl, cam, viewers):
    """Answer a new request. Returns True if the socket became a stream viewer."""
    cl.settimeout(2)
    request = cl.recv(1024)
    if not request:
        return False
    try:
        path = request.split(b" ")[1].decode()
    except Exception:
        path = "/"

    if path.startswith("/stream"):
        if len(viewers) >= MAX_VIEWERS:
            send_text(cl, "503 Service Unavailable", "text/plain", "too many viewers")
            return False
        cl.sendall(STREAM_HEADER)
        print("Viewer connected ({} total)".format(len(viewers) + 1))
        return True
    if path.startswith("/capture"):
        img = cam.capture()
        if img:
            send_text(cl, "200 OK", "image/jpeg", img)
            release(cam)
        else:
            send_text(cl, "503 Service Unavailable", "text/plain", "capture failed")
    elif path.startswith("/status"):
        body = "wifi={} ip={} frames={} fps={:.1f} viewers={} free_ram={}".format(
            stats["wifi"], stats["ip"], stats["frames"], stats["fps"], len(viewers), gc.mem_free())
        send_text(cl, "200 OK", "text/plain", body)
    elif path == "/" or path.startswith("/index"):
        send_text(cl, "200 OK", "text/html", PAGE)
    else:
        send_text(cl, "404 Not Found", "text/plain", "not found")
    return False


def send_frame(cam, viewers):
    img = cam.capture()
    if not img:
        return False
    part = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n".format(len(img)).encode()
    for v in viewers[:]:
        try:
            v.sendall(part)
            v.sendall(img)
            v.sendall(b"\r\n")
        except OSError:
            viewers.remove(v)
            v.close()
            print("Viewer disconnected ({} left)".format(len(viewers)))
    release(cam)
    return True


def main():
    print("=== SumoCam v2 starting ===")
    ip = start_wifi()          # Wi-Fi first: the camera running can disturb connecting
    stats["ip"] = ip
    cam = start_camera()
    setup_lighting(cam)
    warm_up_and_lock(cam)
    print("Camera address: http://{}/   (stream: http://{}/stream)".format(ip, ip))

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", 80))
    srv.listen(4)
    srv.setblocking(False)

    blink(3)   # three quick blinks = ready
    viewers = []
    count = 0
    t0 = time.ticks_ms()

    while True:
        # 1) accept any new connection without waiting
        try:
            cl, addr = srv.accept()
        except OSError:
            cl = None
        if cl:
            try:
                if handle_new(cl, cam, viewers):
                    cl.settimeout(2)   # a frozen viewer is dropped after 2 s
                    viewers.append(cl)
                else:
                    cl.close()
            except OSError as e:
                print("Request error:", e)
                cl.close()

        # 2) send the next frame to everyone watching
        if viewers:
            led(True)
            if send_frame(cam, viewers):
                count += 1
                stats["frames"] += 1
        else:
            led(False)
            time.sleep_ms(10)

        # 3) print stats every 5 seconds
        dt = time.ticks_diff(time.ticks_ms(), t0)
        if dt >= 5000:
            stats["fps"] = count * 1000 / dt
            if viewers:
                print("{:.1f} fps to {} viewer(s), free RAM {}".format(
                    stats["fps"], len(viewers), gc.mem_free()))
            count = 0
            t0 = time.ticks_ms()
            gc.collect()


main()
