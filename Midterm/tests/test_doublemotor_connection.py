"""Double Motor connection logic with a fake device (no Bluetooth needed).

The fake mimics the parts of legoeducation.DoubleMotor that motor_imu uses:
search(), connect(device, device_notification_delay=), connected,
set_notification_callback(), imu_device, motor_run_for_time(), disconnect().
"""

import threading
import time
from types import SimpleNamespace

import pytest

import motor_imu
from motor_imu import RSSI_BY_ADDRESS, MotorIMU, choose_nearest


def dev(name, address):
    return SimpleNamespace(name=name, address=address)


class FakeDoubleMotor:
    def __init__(self, scans, rssi):
        self.scans = list(scans)        # list of device lists, one per search() call
        self.rssi = rssi
        self.connected = False
        self.callback = None
        self.connect_calls = []
        self.search_kwargs = []
        self.pulses = []
        self.disconnected = False
        self.imu_device = SimpleNamespace(accelerometerX=0, accelerometerY=0, accelerometerZ=1000,
                                          gyroscopeX=0, gyroscopeY=0, gyroscopeZ=0)

    def search(self, timeout, **kw):
        self.search_kwargs.append(kw)
        devices = self.scans.pop(0) if self.scans else []
        for d in devices:  # what RssiRecordingBLE.on_scan would record
            RSSI_BY_ADDRESS[d.address] = self.rssi[d.address]
        return devices or None

    def set_notification_callback(self, cb):
        self.callback = cb

    def connect(self, device, device_notification_delay=100):
        self.connect_calls.append((device, device_notification_delay))
        self.connected = True

    def notify(self, n=1, **imu):
        for k, v in imu.items():
            setattr(self.imu_device, k, v)
        for _ in range(n):
            self.callback(object())

    def drop(self):
        self.connected = False

    def motor_run_for_time(self, ms, *, motor, speed, blocking):
        self.pulses.append((ms, motor, speed, blocking))

    def disconnect(self):
        self.connected = False
        self.disconnected = True


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture(autouse=True)
def fast_reconnect(monkeypatch):
    monkeypatch.setattr(motor_imu.config, "BLE_RECONNECT_INTERVAL_S", 0.01)
    RSSI_BY_ADDRESS.clear()


def test_rssi_transport_is_registered_with_library():
    from legoeducation.ble_transport import get_transport
    assert motor_imu.LEGO_AVAILABLE
    assert get_transport() is motor_imu.RssiRecordingBLE


def test_rssi_transport_records_matching_devices_only():
    from legoeducation.basic_ble import LEGO_COMPANY_ID, SERVICE_UUID
    t = motor_imu.RssiRecordingBLE()
    t._filters = t._normalize_filters({"product_id": 513})  # 513 = Double Motor
    lego_adv = SimpleNamespace(connectable=True, service_uuids=[SERVICE_UUID.lower()], rssi=-48,
                               manufacturer_data={LEGO_COMPANY_ID: bytes([0x02, 0x01, 0, 0, 0, 0])})
    other_adv = SimpleNamespace(connectable=True, service_uuids=[], rssi=-30, manufacturer_data={})
    single_motor_adv = SimpleNamespace(connectable=True, service_uuids=[SERVICE_UUID.lower()], rssi=-20,
                                       manufacturer_data={LEGO_COMPANY_ID: bytes([0x02, 0x00, 0, 0, 0, 0])})
    t.on_scan(dev("Double Motor", "AA"), lego_adv)
    t.on_scan(dev("Headphones", "BB"), other_adv)
    t.on_scan(dev("Single Motor", "CC"), single_motor_adv)
    assert [d.address for d in t.device_list] == ["AA"]
    assert RSSI_BY_ADDRESS == {"AA": -48}


def test_choose_nearest_picks_strongest_rssi():
    a, b, c = dev("A", "1"), dev("B", "2"), dev("C", "3")
    assert choose_nearest([a, b, c], {"1": -80, "2": -45, "3": -60}) is b
    assert choose_nearest([a, b], {"1": -70}) is a  # unknown RSSI ranks last
    assert choose_nearest([], {}) is None


def test_connects_to_nearest_and_streams_imu():
    far, near = dev("Double Motor far", "F"), dev("Double Motor near", "N")
    fake = FakeDoubleMotor(scans=[[far, near]], rssi={"F": -85, "N": -40})
    swings = []
    m = MotorIMU(on_swing=swings.append, device_factory=lambda: fake)
    m.start()
    try:
        assert wait_for(lambda: fake.connect_calls)
        device, delay = fake.connect_calls[0]
        assert device is near
        assert delay == motor_imu.config.BLE_NOTIFICATION_MS
        assert m.device_name == "Double Motor near"

        # calibration at rest -> status connected
        fake.notify(motor_imu.config.IMU_CALIBRATION_SAMPLES)
        assert m.detector is not None
        assert m.detector.raw_per_g == pytest.approx(1000)
        assert m.status.startswith("connected")
        assert m.connected
        assert m.samples_received == motor_imu.config.IMU_CALIBRATION_SAMPLES
    finally:
        m.stop()
    assert fake.disconnected


def test_swing_from_streamed_imu_reaches_callback():
    fake = FakeDoubleMotor(scans=[[dev("DM", "X")]], rssi={"X": -50})
    swings = []
    m = MotorIMU(on_swing=swings.append, device_factory=lambda: fake)
    m.start()
    try:
        assert wait_for(lambda: fake.connected)
        fake.notify(motor_imu.config.IMU_CALIBRATION_SAMPLES)
        fake.notify(5)
        fake.notify(1, gyroscopeZ=int(motor_imu.config.SWING_GYRO_THRESHOLD_RAW * 2))
        deadline = time.monotonic() + motor_imu.config.SWING_PEAK_WINDOW_S + 0.1
        while time.monotonic() < deadline and not swings:
            fake.notify(1)
            time.sleep(0.01)
        assert len(swings) == 1
        assert swings[0].strength >= 2.0
    finally:
        m.stop()


def test_reconnects_after_drop():
    fake = FakeDoubleMotor(scans=[[dev("DM", "X")], [dev("DM", "X")]], rssi={"X": -50})
    m = MotorIMU(on_swing=lambda e: None, device_factory=lambda: fake)
    m.start()
    try:
        assert wait_for(lambda: len(fake.connect_calls) == 1)
        fake.drop()
        assert wait_for(lambda: len(fake.connect_calls) == 2)
        assert fake.connected
    finally:
        m.stop()


def test_retries_when_no_motor_found():
    fake = FakeDoubleMotor(scans=[[], [], [dev("DM", "X")]], rssi={"X": -50})
    m = MotorIMU(on_swing=lambda e: None, device_factory=lambda: fake)
    m.start()
    try:
        assert wait_for(lambda: fake.connected)
        assert len(fake.search_kwargs) == 3
    finally:
        m.stop()


def test_scan_passes_card_filters(monkeypatch):
    monkeypatch.setattr(motor_imu.config, "BLE_CARD_COLOR", 9)
    monkeypatch.setattr(motor_imu.config, "BLE_CARD_SERIAL", "0049")
    fake = FakeDoubleMotor(scans=[[dev("DM", "X")]], rssi={"X": -50})
    m = MotorIMU(on_swing=lambda e: None, device_factory=lambda: fake)
    m._dm = fake
    assert m.connect_once()
    assert fake.search_kwargs[0]["card_color"] == 9
    assert fake.search_kwargs[0]["card_serial"] == "0049"


def test_haptic_pulse_sent_from_motor_thread(monkeypatch):
    monkeypatch.setattr(motor_imu.config, "HAPTIC_ENABLED", True)
    fake = FakeDoubleMotor(scans=[[dev("DM", "X")]], rssi={"X": -50})
    m = MotorIMU(on_swing=lambda e: None, device_factory=lambda: fake)
    m.start()
    try:
        assert wait_for(lambda: fake.connected)
        m.request_haptic()
        assert wait_for(lambda: fake.pulses)
        ms, motor, speed, blocking = fake.pulses[0]
        assert ms == motor_imu.config.HAPTIC_PULSE_MS and blocking is False
    finally:
        m.stop()


def test_haptic_pulse_never_registers_as_swing_or_twist(monkeypatch):
    monkeypatch.setattr(motor_imu.config, "HAPTIC_ENABLED", True)
    fake = FakeDoubleMotor(scans=[[dev("DM", "X")]], rssi={"X": -50})
    swings = []
    m = MotorIMU(on_swing=swings.append, device_factory=lambda: fake)
    m.start()
    try:
        assert wait_for(lambda: fake.connected)
        fake.notify(motor_imu.config.IMU_CALIBRATION_SAMPLES + 5)
        assert wait_for(lambda: m.roll is not None)
        m.calibrate_neutral()
        m.request_haptic()
        assert wait_for(lambda: fake.pulses)
        assert m.quiet
        # the pulse shakes the handle: big accel + gyro on every axis while quiet
        shake = int(motor_imu.config.SWING_GYRO_THRESHOLD_RAW * 3)
        fake.notify(5, accelerometerX=3000, accelerometerY=2000, gyroscopeX=shake, gyroscopeY=shake,
                    gyroscopeZ=shake)
        fake.notify(5, accelerometerX=0, accelerometerY=0, gyroscopeX=0, gyroscopeY=0, gyroscopeZ=0)
        assert swings == []
        assert m.roll.frozen
    finally:
        m.stop()


def test_haptic_disabled_flag(monkeypatch):
    monkeypatch.setattr(motor_imu.config, "HAPTIC_ENABLED", False)
    m = MotorIMU(on_swing=lambda e: None, device_factory=lambda: None)
    m.request_haptic()
    assert not m._haptic.is_set()


def test_factory_failure_reports_unavailable():
    def boom():
        raise RuntimeError("no bluetooth adapter")
    m = MotorIMU(on_swing=lambda e: None, device_factory=boom)
    m.start()
    assert wait_for(lambda: m.status.startswith("unavailable"))
    m.stop()


@pytest.mark.hardware
def test_real_double_motor_connects_and_streams_imu():
    """Needs a powered-on Double Motor nearby and Bluetooth enabled."""
    events = []
    m = MotorIMU(on_swing=events.append)
    m.start()
    try:
        assert wait_for(lambda: m.status.startswith("connected"), timeout=30), m.status
        n0 = m.samples_received
        time.sleep(1.0)
        rate = m.samples_received - n0
        assert rate > 10, f"only {rate} IMU samples/s"
        print(f"\nconnected to {m.device_name} [{m.device_address}], {rate} samples/s, "
              f"1 g = {m.detector.raw_per_g:.0f} raw")
    finally:
        m.stop()
