"""
UDP helpers for the bridge: one JSON object per packet, UTF-8, compact separators.
UdpSender sends to Godot; UdpReceiver listens for Godot's commands.
"""

import json
import select
import socket


def encode(msg):
    return json.dumps(msg, separators=(",", ":")).encode("utf-8")


class UdpSender:
    """Fire-and-forget sender to one (host, port). Nobody listening is not an error for UDP."""

    def __init__(self, host, port):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0

    def send(self, msg):
        """Send a dict; returns the bytes sent, or None if the OS refused the send."""
        data = encode(msg)
        try:
            self.sock.sendto(data, self.addr)
        except OSError as e:
            print(f"[udp] send to {self.addr[0]}:{self.addr[1]} failed: {e}")
            return None
        self.sent += 1
        return data

    def close(self):
        self.sock.close()


class UdpReceiver:
    """Non-blocking listener on (host, port). wait() blocks until a packet arrives or the timeout
    passes -- use it as the main loop's sleep, so a packet is handled the moment it lands."""

    def __init__(self, host, port, max_bytes=4096):
        self.addr = (host, port)
        self.max_bytes = max_bytes
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.addr)          # OSError if the port is taken
        self.sock.setblocking(False)
        self.received = 0

    def wait(self, timeout):
        """True if a packet is waiting (returns early), False after timeout seconds."""
        ready, _, _ = select.select([self.sock], [], [], max(0.0, timeout))
        return bool(ready)

    def drain(self, limit=64):
        """Every packet waiting now (at most limit), as bytes. Never blocks."""
        packets = []
        for _ in range(limit):
            try:
                data, _ = self.sock.recvfrom(self.max_bytes)
            except (BlockingIOError, InterruptedError):
                break
            except ConnectionResetError:   # Windows reports an earlier ICMP "port unreachable" here
                continue
            except OSError as e:           # e.g. a packet bigger than max_bytes (WinError 10040): skip it
                print(f"[udp] receive on port {self.addr[1]} failed: {e}")
                continue
            self.received += 1
            packets.append(data)
        return packets

    def close(self):
        self.sock.close()
