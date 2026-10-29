"""ble_link brick (copied from unoq_app): the BLE relay CONTAINER.

App Lab picks up brick_compose.yaml, which starts the `ble-relay` container
running relay.py (bleak + BlueZ over D-Bus) and serving GaitNode notifications
on the Unix socket .cache/ble_link/relay.sock. requirements.txt (bleak) is
installed into the app's venv, which the relay container reuses.

The app-side client lives in python/ble_client.py (RelayClient + AnkleLink);
nothing imports this package.
"""
