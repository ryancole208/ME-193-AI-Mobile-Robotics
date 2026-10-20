"""Background reader for the ESP32-CAM's MJPEG stream.

Runs in its own thread so nothing else (the main loop, later the detector)
ever blocks on network I/O. Only the newest frame is kept - as soon as a new
one arrives the old one is discarded, since detection only ever needs "what
does the camera see right now," not a backlog.

Reconnects automatically if the stream drops (camera reboot, Wi-Fi hiccup,
wrong network, camera not powered) and reports when the frame we're holding
has gone stale, instead of raising and crashing the app.
"""

import http.client
import threading
import time
from collections import deque
from urllib.parse import urlsplit


def _parse_boundary(content_type):
    """Pulls the multipart boundary string out of a Content-Type header like
    'multipart/x-mixed-replace; boundary=frame'. Returns None if not found."""
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            return part.split("=", 1)[1].strip().strip('"')
    return None


def _read_exact(fp, length):
    """Reads exactly `length` bytes from a file-like object, raising if the
    connection closes early (fp.read() can return short reads)."""
    chunks = []
    remaining = length
    while remaining > 0:
        chunk = fp.read(remaining)
        if not chunk:
            raise RuntimeError("camera stream closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class MjpegStreamReader:
    """Reads a multipart/x-mixed-replace MJPEG stream (as served by the
    ESP32-CAM's /stream endpoint) in a background thread.

    Usage:
        reader = MjpegStreamReader("http://192.168.4.1/stream")
        reader.start()
        ...
        jpeg_bytes, received_at = reader.get_latest_frame()
        reader.stop()
    """

    def __init__(self, url, socket_timeout=3.0, retry_delay=1.0):
        self._url = url
        self._socket_timeout = socket_timeout  # also doubles as the connect timeout
        self._retry_delay = retry_delay  # how long to wait before reconnecting after an error

        self._lock = threading.Lock()
        self._latest_frame = None  # bytes of the most recent JPEG, or None if never received
        self._latest_frame_time = None  # time.monotonic() timestamp of that frame
        self._connected = False  # True while the HTTP stream connection is open
        self._frame_times = deque(maxlen=30)  # recent frame timestamps, used for fps

        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_latest_frame(self):
        """Returns (jpeg_bytes, received_at_monotonic) for the newest frame,
        or (None, None) if no frame has ever been received."""
        with self._lock:
            return self._latest_frame, self._latest_frame_time

    def get_frame_age(self):
        """Seconds since the last frame arrived, or None if we've never
        received one at all."""
        with self._lock:
            if self._latest_frame_time is None:
                return None
            return time.monotonic() - self._latest_frame_time

    def is_connected(self):
        with self._lock:
            return self._connected

    def get_fps(self):
        """Approximate frames-per-second, measured across the last up-to-30
        received frames. Returns 0.0 until at least 2 frames have arrived."""
        with self._lock:
            times = list(self._frame_times)
        if len(times) < 2:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def _set_connected(self, connected):
        with self._lock:
            self._connected = connected

    def _store_frame(self, data):
        now = time.monotonic()
        with self._lock:
            self._latest_frame = data
            self._latest_frame_time = now
            self._frame_times.append(now)

    def _run(self):
        """Background thread body: keep (re)connecting to the stream until
        stop() is called. Every connection attempt that fails just logs and
        retries after _retry_delay - this thread must never let an exception
        escape and kill the app."""
        while not self._stop_event.is_set():
            try:
                self._read_stream_once()
            except Exception as exc:
                print(f"[camera_stream] stream error: {exc!r}; retrying in {self._retry_delay}s")
            self._set_connected(False)
            self._stop_event.wait(self._retry_delay)

    def _read_stream_once(self):
        """Opens the MJPEG stream and reads frames from it until the
        connection drops or an error occurs. Raises on any failure - the
        caller (_run) catches it and reconnects."""
        parts = urlsplit(self._url)
        conn = http.client.HTTPConnection(
            parts.hostname, parts.port or 80, timeout=self._socket_timeout
        )
        try:
            conn.request("GET", parts.path or "/")
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} from camera")

            boundary = _parse_boundary(response.getheader("Content-Type", ""))
            if boundary is None:
                raise RuntimeError(
                    f"expected a multipart MJPEG stream, got Content-Type="
                    f"{response.getheader('Content-Type')!r}"
                )
            boundary_marker = ("--" + boundary).encode("ascii", errors="ignore")

            self._set_connected(True)

            while not self._stop_event.is_set():
                line = response.readline()
                if not line:
                    raise RuntimeError("camera stream closed")
                line = line.strip()
                if not line or not line.startswith(boundary_marker):
                    continue  # blank line between parts, or something we don't care about

                content_length = None
                while True:
                    header_line = response.readline()
                    if not header_line:
                        raise RuntimeError("camera stream closed while reading headers")
                    header_line = header_line.strip()
                    if not header_line:
                        break  # blank line = end of this part's headers
                    if header_line.lower().startswith(b"content-length:"):
                        content_length = int(header_line.split(b":", 1)[1].strip())

                if content_length is None:
                    raise RuntimeError("MJPEG part had no Content-Length header")

                self._store_frame(_read_exact(response, content_length))
        finally:
            conn.close()
