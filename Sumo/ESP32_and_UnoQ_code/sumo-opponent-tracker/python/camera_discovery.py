"""Finds the ESP32-CAM's current address, with its actual IP never
hardcoded anywhere (that changes every time DHCP hands out a new lease).
Tries, in order:

  1. The last address that worked (cached in a small file next to this one).
     Routers usually hand the same device the same IP back, so after the
     first successful discovery on a given network this is instant.
  2. sumocam.local (the camera's mDNS hostname).
  3. 192.168.4.1 (the camera's fixed address when it falls back to its own
     SUMOCAM access point).
  4. A parallel scan of KNOWN_SUBNET_PREFIXES (below) for a host that
     answers /status with a body containing "wifi=".

A note on why step 4 needs a configured prefix instead of detecting it:
this Python app runs inside its own network-isolated container - it can
make outbound connections to any address on the LAN (which is how steps 1
and 3 work), but it has no network interface of its own on the LAN, so it
has no way to ask "what subnet am I on" the way a normal Linux machine
could. The actual camera IP within that prefix is still auto-discovered
by the scan; only the stable /24 shape of the network needs to be told to
it once. (This also means step 2, mDNS, can never actually succeed here -
mDNS relies on receiving multicast replies on the LAN segment, which this
same container isolation blocks - but it's kept, and cheap, in case a
future App Lab version gives the app real network access.)
"""

import concurrent.futures
import http.client
import os
import socket

# ===================== TUNABLE =====================
STATUS_PATH = "/status"
PROBE_TIMEOUT = 0.3  # seconds to wait for one /status response while probing
MDNS_TIMEOUT = 2.0  # seconds to wait for sumocam.local to resolve
SCAN_WORKERS = 64  # how many hosts to probe at once during the subnet scan
CACHE_PATH = os.path.join(os.path.dirname(__file__), ".camera_ip_cache")
SUMOCAM_FALLBACK_IP = "192.168.4.1"
MDNS_HOSTNAME = "sumocam.local"

# The /24 prefix (first three numbers) of every network the camera and UNO Q
# might both be on. Add a new entry here whenever you join a new network -
# e.g. your phone hotspot's prefix, once you know it (check the hotspot's
# settings screen, or the camera's own /status page while connected to it).
# This is NOT the camera's IP - that part (the last number) is still found
# automatically by scanning every possibility in the prefix.
KNOWN_SUBNET_PREFIXES = [
    "192.168.1",  # home Wi-Fi
    "172.20.10",  # phone hotspot ("Ryan Phone") - iPhone hotspots use 172.20.10.x
]
# =====================================================


def _probe(ip, timeout=PROBE_TIMEOUT, verbose=False):
    """True if http://ip/status responds with "wifi=" in the body. That
    string is specific to the ESP32-CAM's own /status page, so it tells a
    real camera apart from any other device that happens to answer on
    port 80. Pass verbose=True (used for the direct, non-scan checks) to
    print *why* a probe failed - useful when a known-good IP isn't found."""
    try:
        conn = http.client.HTTPConnection(ip, 80, timeout=timeout)
        try:
            conn.request("GET", STATUS_PATH)
            response = conn.getresponse()
            body = response.read(512)
            ok = response.status == 200 and b"wifi=" in body
            if verbose and not ok:
                print(
                    f"[camera_discovery] {ip}{STATUS_PATH} -> HTTP {response.status}, "
                    f"body starts with {body[:80]!r}"
                )
            return ok
        finally:
            conn.close()
    except OSError as exc:
        if verbose:
            print(f"[camera_discovery] {ip}{STATUS_PATH} -> {exc!r}")
        return False


def _read_cached_ip():
    try:
        with open(CACHE_PATH, "r") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _write_cached_ip(ip):
    try:
        with open(CACHE_PATH, "w") as f:
            f.write(ip)
    except OSError as exc:
        print(f"[camera_discovery] couldn't write IP cache: {exc!r}")


def _resolve_mdns(hostname, timeout=MDNS_TIMEOUT):
    """socket.gethostbyname() has no built-in timeout and can hang for a
    while if mDNS isn't supported, so it runs on a worker thread that we
    just stop waiting on if it's too slow."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(socket.gethostbyname, hostname)
        try:
            return future.result(timeout=timeout)
        except Exception:
            return None


def _scan_known_subnets():
    """Probes every host in every configured KNOWN_SUBNET_PREFIXES range, in
    parallel. Only practical because home routers and phone hotspots are
    almost always /24 subnets (254 hosts), so even scanning several of them
    at once finishes in well under a second."""
    candidates = [
        f"{prefix}.{host}" for prefix in KNOWN_SUBNET_PREFIXES for host in range(1, 255)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=SCAN_WORKERS) as pool:
        future_to_ip = {pool.submit(_probe, ip): ip for ip in candidates}
        for future in concurrent.futures.as_completed(future_to_ip):
            if future.result():
                return future_to_ip[future]
    return None


def discover_camera():
    """Returns the camera's base URL, e.g. "http://192.168.1.42", or None if
    it can't be found anywhere right now. Successful lookups are cached so
    the next call (e.g. after a brief Wi-Fi hiccup) is instant."""
    cached_ip = _read_cached_ip()
    if cached_ip:
        print(f"[camera_discovery] trying cached address {cached_ip}")
        if _probe(cached_ip, verbose=True):
            return f"http://{cached_ip}"

    print(f"[camera_discovery] trying {MDNS_HOSTNAME}")
    mdns_ip = _resolve_mdns(MDNS_HOSTNAME)
    if mdns_ip is None:
        print(f"[camera_discovery] {MDNS_HOSTNAME} did not resolve")
    elif _probe(mdns_ip, verbose=True):
        _write_cached_ip(mdns_ip)
        return f"http://{mdns_ip}"

    print(f"[camera_discovery] trying SUMOCAM fallback {SUMOCAM_FALLBACK_IP}")
    if _probe(SUMOCAM_FALLBACK_IP, verbose=True):
        _write_cached_ip(SUMOCAM_FALLBACK_IP)
        return f"http://{SUMOCAM_FALLBACK_IP}"

    print(f"[camera_discovery] scanning known subnets {KNOWN_SUBNET_PREFIXES}")
    scanned_ip = _scan_known_subnets()
    if scanned_ip:
        print(f"[camera_discovery] found camera at {scanned_ip}")
        _write_cached_ip(scanned_ip)
        return f"http://{scanned_ip}"

    print("[camera_discovery] subnet scan found nothing")
    return None
