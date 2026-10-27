# ble_link: Bluetooth LE link to the GaitNode ankle node

This is a local brick for the Gait Activity Classifier. It connects to the Pico W ankle node
over BLE and streams its notifications to the app.

## Why there are two parts

App Lab runs the app's Python in a container that can't reach Bluetooth. That container has
no D-Bus socket, and Linux only allows Bluetooth sockets in the host network namespace. So
`brick_compose.yaml` starts one more small container, **`ble-relay`**:

- **Image:** the same one as the app (`python-apps-base`), using the same venv that App Lab
  builds, with `bleak` installed from this folder's `requirements.txt`.
- **Mounts:** `/run/dbus` read-only, so `bleak` can talk to BlueZ. BlueZ's policy already lets
  the `arduino` user do this - the same D-Bus connection is used both for `bleak`'s own
  scan/connect/disconnect and for the relay's direct BlueZ queries below, so if one works the
  other does too.
- **Runs:** `relay.py`, which scans for `GaitNode`, connects and reads the Info
  characteristic. After **every** (re)connection it subscribes to IMU and battery
  notifications, and it reconnects forever.
- **Output:** forwards everything as JSON lines on the Unix socket
  `/app/.cache/ble_link/relay.sock`.

### Staying connected and reconnecting cleanly

- **Keepalive:** a `PING` is written to the Control characteristic every 2 s while connected
  (`PING_INTERVAL_S`), independent of anything the app sends - the Pico W drops the link after
  6 s without one.
- **Clean shutdown:** `SIGTERM` (what `docker stop` / App Lab's "stop app" sends) and `SIGINT`
  are handled explicitly: the relay stops notifications and disconnects from BlueZ before
  exiting, every time a connection ends (not just on shutdown). Without this, killing the
  process mid-connection leaves BlueZ thinking the Pico W is still connected - the Pico W then
  never resumes advertising, and the next start's scan hangs forever.
- **Reconnecting after an unclean exit (e.g. an old build, or a crash):** the relay remembers
  the GaitNode's address in `last_address` next to the socket. Before scanning, it asks BlueZ
  (over D-Bus) whether that address is still marked connected. If so, it connects to it directly
  instead of scanning (BlueZ's `Connect()` on an already-connected device just succeeds); if that
  direct connect fails, it explicitly disconnects the stale link via BlueZ so a normal scan can
  find the now-advertising Pico W.

`__init__.py` holds the app side: the `BleLink` brick (or `RelayClient` outside App Lab). It
reads that socket and calls `on_event(LinkEvent)` for each message.

## Usage

```python
from ble_link import BleLink

link = BleLink(on_event=my_queue.put)   # started/stopped automatically by App.run()
link.write_control("PING")              # -> Control characteristic
```

## Socket messages

| From the relay | Meaning |
|---|---|
| `hello` | relay version and the BlueZ check (`{"ok": true, "adapters": {"hci0": true}}`) |
| `status` | `scanning`, `connecting`, `connected`, `disconnected` or `error`, with a message |
| `conn` | address, RSSI from the advertisement, MTU and how it was obtained, Info characteristic (hex) |
| `imu` / `bat` | raw notification payloads (hex); parsing happens in `python/gait/protocol.py` |
| `mtu` | the MTU changed after connecting |
| `log` | relay warnings worth showing in the app log |

| From the app | Meaning |
|---|---|
| `{"t": "write", "d": "PING"}` | write ASCII to the Control characteristic |

The client also produces `relay_up` and `relay_down` events for the socket itself.

## Variables

| Variable | Default | Meaning |
|---|---|---|
| `GAITNODE_NAME` | `GaitNode` | advertised name to connect to |
| `BLE_RELAY_SOCKET` | `/app/.cache/ble_link/relay.sock` | socket shared by the two containers |
| `BLE_RELAY_ADDRESS_FILE` | next to `BLE_RELAY_SOCKET`, named `last_address` | remembers the GaitNode's address across relay restarts |
