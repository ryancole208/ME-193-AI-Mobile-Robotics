"""Client for the UNO Q HTTP API (contract 8.1).

Every request carries X-Token. Connection errors and timeouts are retried with
a short back-off (Wi-Fi on a walking person drops often); a 401 is never
retried, because a wrong token won't fix itself.
"""
import time
from pathlib import Path

import requests


class UnoQError(Exception):
    pass


class UnoQ:
    def __init__(self, url, token, timeout=3.0, retries=2, backoff=0.5):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.http = requests.Session()

    # ------------------------------------------------------------ core

    def _request(self, method, path, headers=None, timeout=None, **kw):
        h = {"X-Token": self.token}
        h.update(headers or {})
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = self.http.request(method, self.url + path, headers=h,
                                      timeout=timeout or self.timeout, **kw)
            except (requests.ConnectionError, requests.Timeout) as e:
                last = e
                if attempt < self.retries:
                    time.sleep(self.backoff * (attempt + 1))
                continue
            if r.status_code == 401:
                raise UnoQError("401 Unauthorized: the UNO Q rejected the token (check config.yaml `token`)")
            if r.status_code >= 500 and attempt < self.retries:
                last = UnoQError(f"HTTP {r.status_code} from {path}")
                time.sleep(self.backoff * (attempt + 1))
                continue
            return r
        raise UnoQError(f"UNO Q unreachable at {self.url}{path}: {last}")

    def get_json(self, path, params=None, **kw):
        r = self._request("GET", path, params=params, **kw)
        if not r.ok:
            raise UnoQError(f"GET {path} -> HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def post_json(self, path, body=None, **kw):
        r = self._request("POST", path, json=body if body is not None else {}, **kw)
        if not r.ok:
            raise UnoQError(f"POST {path} -> HTTP {r.status_code}: {r.text[:200]}")
        return r.json() if r.content else {}

    def post_bytes(self, path, data, **kw):
        r = self._request("POST", path, data=data,
                          headers={"Content-Type": "application/octet-stream"}, **kw)
        try:
            body = r.json()
        except ValueError:
            body = {"ok": False, "error": r.text[:500]}
        if not r.ok and "ok" not in body:
            body["ok"] = False
        return body

    def download(self, path, dest, chunk=1 << 16):
        """Download to dest, resuming a previous dest.part if one exists.

        We ask for the missing tail with an HTTP Range header. If the server
        answers 206 we append; if it ignores Range (200) we start over. Returns
        False if the file doesn't exist on the UNO Q (404).
        """
        dest = Path(dest)
        part = dest.with_name(dest.name + ".part")
        for attempt in range(self.retries + 1):
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            try:
                r = self._request("GET", path, headers=headers, stream=True, timeout=max(self.timeout, 10))
                if r.status_code == 404:
                    return False
                if r.status_code == 416:          # we already have every byte
                    part.replace(dest)
                    return True
                if not r.ok:
                    raise UnoQError(f"GET {path} -> HTTP {r.status_code}")
                mode = "ab" if r.status_code == 206 else "wb"
                with open(part, mode) as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
                part.replace(dest)
                return True
            except (requests.RequestException, UnoQError) as e:
                if attempt >= self.retries:
                    raise UnoQError(f"download {path} failed: {e} (partial kept, rerun to resume)")
                time.sleep(self.backoff * (attempt + 1))
        return False

    # ------------------------------------------------------------ endpoints

    def status(self, **kw):
        return self.get_json("/api/status", **kw)

    def events(self, since, **kw):
        return self.get_json("/api/events", params={"since": since}, **kw)

    def stream(self, since, **kw):
        return self.get_json("/api/stream", params={"since": since}, **kw)

    def files(self):
        return self.get_json("/api/files")

    def truth(self, session_id, steps, distance_in=None, is_calib_walk=False, notes=""):
        return self.post_json("/api/session/truth", {
            "session_id": session_id, "steps": int(steps), "distance_in": distance_in,
            "is_calib_walk": bool(is_calib_walk), "notes": notes or ""})


def cfg_client(cfg, **overrides):
    """UnoQ client from config.yaml; overrides (timeout=, retries=) win over the config."""
    t = cfg.get("timeouts", {}) or {}
    kw = dict(timeout=t.get("request_s", 3.0), retries=t.get("retries", 2))
    kw.update(overrides)
    return UnoQ(cfg["unoq_url"], cfg["token"], **kw)
