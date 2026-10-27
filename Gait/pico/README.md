# GaitNode: Pico W ankle sensor

A Raspberry Pi Pico W worn on the ankle. It reads an MPU-6050 IMU at exactly 100 Hz and streams the raw readings over Bluetooth Low Energy (BLE) to an Arduino UNO Q on the waist, which does all the classification. This node only measures and transmits. It has no features, no models and no calibration.

Everything is MicroPython using only built-in modules (`machine`, `bluetooth`, `struct`, `time`, `micropython`, `gc`). There's nothing to install through Thonny's package manager.

> **Build status:** all code written (steps 1–5), not yet tested on hardware. Test in the order of the build steps below; each step only needs the files listed for it.

## Files

| File | On the Pico W? | Contents |
|---|---|---|
| `config.py` | yes | All settings: pins, ranges, packet size, `DEBUG`, `WATCHDOG_ENABLED` |
| `mpu6050_fifo.py` | yes | Small MPU-6050 driver built around the sensor's FIFO |
| `ble_node.py` | yes | BLE peripheral: advertising, GATT service, notifications |
| `main.py` | yes, last | Startup delay, main loop, packets, LED, battery |
| `tests/t1_imu.py` … `t3_ble_counter.py` | no, run from Thonny | One check per build step |
| `tools/ble_check.py` | no, runs on the laptop | Laptop-side BLE checker (Python + `bleak`) |

## Wiring

All pin numbers are checked against the official [Pico W pinout](https://datasheets.raspberrypi.com/picow/PicoW-A4-Pinout.pdf) and the [Pico W datasheet](https://datasheets.raspberrypi.com/picow/pico-w-datasheet.pdf).

| From | To Pico W | Notes |
|---|---|---|
| MPU-6050 VCC | 3V3(OUT), pin 36 | regulated 3.3 V, keep total load under 300 mA |
| MPU-6050 GND | GND, e.g. pin 38 | any GND pin works |
| MPU-6050 SDA | GP4, pin 6 | I2C0 SDA |
| MPU-6050 SCL | GP5, pin 7 | I2C0 SCL |
| Battery + → switch → | VSYS, pin 39 | allowed range is 1.8 to 5.5 V |
| Battery − | GND, e.g. pin 38 | |

Leave the MPU-6050's AD0, INT, XDA, XCL pins unconnected. AD0 floating or low gives address 0x68.

**⚠ Switch the battery OFF whenever USB is plugged in.** USB power reaches VSYS through a diode (VSYS ≈ 4.7 V). With the battery wired straight to VSYS, that voltage is pushed backwards into the AAA cells, which can make alkaline cells leak or swell. The datasheet's fix is a second Schottky diode in series with the battery. We're not adding parts, so use the switch instead.

**Why 2 × AAA works.** Fresh alkaline cells give about 3.0 to 3.2 V, falling to about 1.8 V when flat. The Pico W's buck-boost regulator makes a steady 3.3 V from anything between 1.8 and 5.5 V, so the sensor gets a steady 3.3 V the whole time.

## Axis convention

The sensor sends its own raw axes. Any remapping happens on the UNO Q. To mount it:

- **+Y** points up the shin toward the knee. The pod lid's arrow marks +Y.
- **+X** points forward.
- **+Z** points outward from the leg. On a GY-521 board, +Z comes out of the chip side of the board.

X, Y and Z must form a right-handed set (X × Y = Z). "Y up, Z outward, X forward" is only possible on the **right** ankle. On the left ankle, with Y up and Z outward, X points backward. Either wear the pod on the right ankle or tell the UNO Q side about the flip.

## Installing MicroPython (once per board)

You need [Thonny](https://thonny.org) (version 4 or newer).

1. Unplug the Pico W. **Hold the white BOOTSEL button** and plug the USB cable into the computer, then let go. A drive called `RPI-RP2` appears. This is normal.
2. In Thonny: **Tools → Options → Interpreter**.
3. In the first drop-down choose **MicroPython (Raspberry Pi Pico)**.
4. Click **Install or update MicroPython** (bottom right).
5. In the installer:
   - Target volume: `RPI-RP2`
   - MicroPython family: `RP2`
   - Variant: **Raspberry Pi • Pico W / Pico WH**. Choose the **W** variant. The plain "Pico" firmware has no Wi-Fi or Bluetooth.
   - Version: the newest one offered
6. Click **Install**, wait for "Done!", and close the installer.
7. Back in Options, set **Port** to the board (or "Try to detect port automatically") and click OK.
8. The Shell at the bottom should show something like `MicroPython v1.xx on 20xx-xx-xx; Raspberry Pi Pico W with RP2040` and a `>>>` prompt. Check that it says **Pico W**.

## Uploading files

For each file that goes on the Pico W:

1. In Thonny, **File → Open… → This computer**, and open the file from this `pico/` folder.
2. **File → Save as… → Raspberry Pi Pico**, and type **exactly the same name** (e.g. `config.py`), then OK.

Alternatively, open **View → Files**. Browse to this folder in the top pane, right-click a file and choose **Upload to /**.

Upload in this order. Library files go up before anything that imports them.

| Order | File | When |
|---|---|---|
| 1 | `config.py` | before step 1 |
| 2 | `mpu6050_fifo.py` | before step 1 |
| 3 | `ble_node.py` | before step 3 |
| 4 | `main.py` | before step 4, **last**, because it starts running on every power-up |

You can upload the first three at once; they do nothing on their own. Hold off on `main.py` until steps 1–3 pass. Once it's on the board it grabs the sensor and Bluetooth at every power-up, so you'd have to stop it with Ctrl+C before each test script.

**Test scripts don't need uploading.** Open `tests/t1_imu.py` (etc.) from this computer and press **F5** (Run). Thonny sends it to the Pico W and runs it there, using the `config.py` and `mpu6050_fifo.py` already on the board.

After editing `config.py` on the laptop, upload it again. The board only sees the copy saved on it.

## Stopping `main.py`

`main.py` runs by itself every time the board powers up. It waits 3 s first and prints `starting in 3 s, press Ctrl+C to stop`.

- **Normal way:** click in Thonny's Shell and press **Ctrl+C**, or click the red **Stop** button. You get a `>>>` prompt and can upload files.
- **If Thonny can't connect:** unplug, plug back in, and press Stop / Ctrl+C during the 3 s window.
- **If the watchdog is on** (`WATCHDOG_ENABLED = True`), stopping the program doesn't stop the watchdog, and the board reboots about 8 s later. Press Ctrl+C during the 3 s startup window, before the watchdog starts. Then upload a `config.py` with `WATCHDOG_ENABLED = False`.
- **Last resort:** in **View → Files**, right-click `main.py` on the Pico and **Rename** it (e.g. to `main_off.py`). Then it won't auto-start.

## Build steps

### Step 1: IMU wiring and axes (`tests/t1_imu.py`)

Upload `config.py` and `mpu6050_fifo.py`, then open `tests/t1_imu.py` and press F5. Expected output, with the sensor lying flat, chip side up:

```
t1_imu: MPU-6050 wiring and axis check
I2C scan found: ['0x68']
WHO_AM_I = 0x68
Configured: +-16 g, +-2000 deg/s, 100 Hz, DLPF_CFG=3
At rest, the axis pointing UP reads about +1.00 g and |a| is about 1.00.

acc g  x +0.01 y -0.02 z +1.00  |a| 1.00   gyro dps  x    -1.2 y    +0.6 z    -0.4   up: +Z
```

Tilt checks:

| Hold the pod with… | Expect |
|---|---|
| lid arrow (+Y) pointing at the ceiling | `y ≈ +1.00`, `up: +Y` |
| outward face (+Z) at the ceiling | `z ≈ +1.00`, `up: +Z` |
| forward edge (+X) at the ceiling | `x ≈ +1.00`, `up: +X` |
| still | gyro values within a few deg/s of 0 (a small constant offset is normal) |
| spun about an axis, counter-clockwise as seen from that axis's + end | that gyro axis goes positive |

### Step 2: FIFO timing (`tests/t2_fifo.py`)

Needs `config.py` and `mpu6050_fifo.py` on the board. Open `tests/t2_fifo.py`, press F5 and wait about 62 s. Expected ending:

```
Part 1 result
  samples collected : 6000 in 60.0 s (expect about 6000)
  effective rate    : 100.00 Hz (expect 100 +- 1; the MPU-6050 clock is +-1%)
  overflows         : 0 (expect 0)
  max FIFO fill     : 156 of 1024 bytes (expect about 120-250)
  raw min / max (counts, and converted):
    ax    -120 ..    150   (   -0.06 ..    +0.07 g)
    ...
Part 2: not reading for 1.2 s to force an overflow...
  FIFO_COUNT = 1024 (expect 1024, i.e. full)
  overflow detected = True (expect True)

t2_fifo: PASS
```

Anything from 5940 to 6060 samples is fine. The sensor's own clock is only accurate to about ±1 %, and the receiver works in sensor time anyway.

### Step 3: Bluetooth counter (`tests/t3_ble_counter.py`)

Needs `config.py` and `ble_node.py` on the board. No sensor is needed. Open `tests/t3_ble_counter.py` and press F5:

```
t3_ble_counter: BLE advertising + counter test
[ble] advertising as GaitNode
Advertising as 'GaitNode'. Connect with nRF Connect or tools/ble_check.py --counter.
```

**Phone check (nRF Connect, free on iOS and Android):**

1. In the Scanner tab, find `GaitNode` and tap **Connect**.
2. Open the service starting `6E400000-…`.
3. On characteristic `6E400001-…`, tap the triple-arrow (subscribe) icon. The value should count up 10 times a second. It's shown as 4 little-endian hex bytes: `01-00-00-00`, `02-00-00-00`, and so on.
4. Thonny should show `CONNECTED`, `central SUBSCRIBED to IMU notifications (MTU …)` and `sent counter 50 …`.
5. Read `6E400004-…`. Expect `01-10-D0-07-64-00`, meaning protocol v1, 16 g, 2000 °/s, 100 Hz.
6. Disconnect in the app. Thonny prints `DISCONNECTED (advertising again)`.

**Laptop check:** disconnect the phone first, because the node takes only one connection at a time. Then, in a terminal on the laptop:

```
pip install bleak
cd Gait/pico/tools
python ble_check.py --counter --seconds 30
```

Expect `connected, MTU …` and lines like `counter 10 (received 10, missed 0)`, ending with `done: received ~300, missed 0`.

### Step 4: full firmware (`main.py`)

1. Stop any running script. Upload `main.py`, keeping the name.
2. Press **Ctrl+D** in the Shell (soft reboot), or unplug and replug. Expected:
   ```
   starting in 3 s, press Ctrl+C to stop
   [ble] advertising as GaitNode
   heartbeat ON: dropping a central after 6000 ms without PING
   IMU running
   status: conn=False stream=False mtu=23 n=1 pkts=49 sent=0 ... pings=0 hb_drops=0 ovf=0 i2c_err=0 ...
   ```
   The LED blinks slowly while advertising. `pkts` climbs by about 10 a second even with nobody connected; those packets fill the resend buffer.
3. On the laptop, run `python ble_check.py`. It runs for 5 minutes; add `--seconds 60` for a quick check.
   - On connect, the LED blinks 3 times quickly, then goes off.
   - Thonny shows `MTU exchanged: 247 …` (or whatever the laptop allows), then `streaming STARTED (MTU …, n = 10)`.
   - `ble_check.py` writes `PING` every 2 s as the heartbeat, so `pings=` in the status line climbs by about 2–3 every 5 s and `hb_drops` stays 0.
   - `ble_check.py` prints a line every second with the latest sample in g and °/s, then this summary:
   ```
   packets lost (seq)  : 0  = 0.00 %
   samples lost (t0)   : 0  = 0.00 %  (target < 2 %)
   effective rate      : 100.0x Hz received
   ```
4. Once everything works, set `DEBUG = False` in `config.py` and upload it again, so there are no prints on battery.

If the MTU stays at 23, `n` drops to 1 and the node sends 100 one-sample packets a second instead of 10 ten-sample packets. That still works, but less efficiently. See Troubleshooting.

### Step 5: robustness and power

**Pull SDA.** With `main.py` running and `ble_check.py` connected, pull out the SDA jumper for a few seconds, then push it back in. Thonny should show, in order:

1. `IMU error #1: [Errno 5] EIO -> reinitialising`
2. `IMU init failed … Retrying in 1 s`, once a second while the wire is out
3. `IMU running` and `resync: about N samples lost, t0 moved forward`

`ble_check.py` receives no packets during the gap, then prints `t0 jump: about N sample(s) missing`. N should be about 100 × the number of seconds the wire was out.

**Out of range.** Walk away with the node, or wrap it in foil, until the link drops. Thonny shows `[ble] central disconnected`, and the LED goes back to slow blinking. When you come back, the UNO Q (or a restarted `ble_check.py`) reconnects and the LED gives 3 quick blinks. `seq` jumps on reconnect; that's expected.

**Central restart (heartbeat).** With `ble_check.py` or the UNO Q app streaming, kill the app abruptly (close the terminal, or stop the UNO Q app) without letting it disconnect cleanly. If the link really closes, Thonny shows `[ble] central disconnected` at once. If the host's Bluetooth stack (BlueZ on the UNO Q) keeps the link open, Thonny shows this about 6 s later instead:

```
[ble] no PING for 6000 ms: disconnecting handle 64 and advertising again (drop #1)
[ble] advertising as GaitNode
```

The LED goes back to slow blinking, and the restarted app reconnects without a power cycle. A later `[ble] old connection 64 closed` line is the stack confirming the drop; it's harmless.

**Current measurement.** See [Current and battery life](#current-and-battery-life).

## Design notes

- **The sensor keeps time, not the Pico W.** MicroPython pauses at random for garbage collection and Bluetooth work. The MPU-6050 samples itself (1 kHz gyro rate ÷ (1 + 9) = 100 Hz) into its 1024-byte FIFO, and the Pico W empties it about every 100 ms. The FIFO holds 85 samples (0.85 s), so even a long pause loses nothing.
- **Why `DLPF_CFG = 3`.** Turning on the digital low-pass filter switches the gyro output rate from 8 kHz to 1 kHz, which makes the exact ÷10 to 100 Hz possible. Setting 3 gives about 44 Hz (accel) and 42 Hz (gyro) bandwidth with about 5 ms delay. That's just under the 50 Hz Nyquist limit for 100 Hz sampling, so it also acts as the anti-aliasing filter, while keeping fast heel-strike transients.
- **Gyro X as clock source.** The register map recommends a gyro-referenced PLL over the internal 8 MHz oscillator, for better stability.
- **No allocation in the loop.** Every buffer, including one pre-sliced `memoryview` per possible sample count, is created once at startup. Little garbage is made, so garbage-collection pauses stay short.
- **`t0_us` is kept as two 16-bit halves.** On the RP2040, MicroPython stores integers below 2^30 without allocating memory, but a 32-bit timestamp passes that after about 18 minutes. Packing the halves as `<HH` gives exactly the same bytes as one little-endian `uint32`.
- **Packets are built straight into the resend ring.** There are 20 preallocated buffers, each with ready-made `memoryview`s for every n from 1 to 10. Sending or re-sending a packet never allocates. The node keeps building packets while nobody is connected, which keeps the FIFO empty; those packets only land in the ring.
- **Subscription check.** MicroPython's `gatts_notify` sends whether or not the central subscribed, and it raises no event when the central subscribes. `ble_node.py` reads the subscription descriptor (CCCD), which MicroPython places at `value_handle + 1`, and clears it on every connect and disconnect. That way the UNO Q must subscribe again after each reconnect, as the protocol says.
- **Heartbeat.** When the UNO Q app restarts, BlueZ can keep the BLE connection open with nothing behind it. The Pico W never gets a disconnect event, so it never advertises again, and the new app instance can't find it. The UNO Q therefore writes `PING` every 2 s. If a central is connected and no `PING` arrives for `PING_TIMEOUT_MS` (6 s, three missed PINGs), the node calls `gap_disconnect` itself, clears its connection and subscription state at once instead of waiting for a disconnect event that may never come, and restarts advertising from the main loop. The timer starts at connect, so a central has 6 s to send its first `PING`. PINGs are stamped in the BLE event handler, so a `RESEND` arriving in the same 100 ms loop can't hide one. Each PING still gets a battery notification in reply, but with the latest 10 s reading rather than a fresh one, so the shared VSYS pin isn't read every 2 s. Set `PING_TIMEOUT_MS = 0` to turn the heartbeat off, for example when poking at the node with nRF Connect.
- **Recovery, not reboot.** An I2C error or a failed health check (WHO_AM_I and FIFO settings, checked once a second) reinitialises the sensor, retrying once a second. Samples lost in the gap are estimated from the Pico W's clock and added to `t0_us`.
- **Settings are read back.** `configure()` reads every register it writes. A flaky SDA wire shows up as a clear error at startup rather than as silent wrong ranges.

## Shared wireless protocol

*Identical in the Pico W and UNO Q briefs. Don't change one side without the other.*

**Link:** BLE. The Pico W is the peripheral, the UNO Q is the central. **Advertised name:** `GaitNode`. **Service UUID:** `6e400000-7a1b-4c2d-9e10-a5c0f0e1d200`

| Characteristic | UUID | Properties | Payload |
|---|---|---|---|
| IMU data | `6e400001-7a1b-4c2d-9e10-a5c0f0e1d200` | notify | packet below |
| Battery | `6e400002-7a1b-4c2d-9e10-a5c0f0e1d200` | notify + read | `uint16 vsys_mV`, every 10 s; `0` when battery reading is disabled |
| Control | `6e400003-7a1b-4c2d-9e10-a5c0f0e1d200` | write | ASCII `PING` (one battery notification in reply; also the heartbeat, see below), `RESEND <seq>` (optional) |
| Info | `6e400004-7a1b-4c2d-9e10-a5c0f0e1d200` | read | `uint8 protocol_version=1; uint8 accel_range_g=16; uint16 gyro_range_dps=2000; uint16 sample_rate_hz=100` |

**IMU packet** (little-endian, packed):

```
uint16 seq         // +1 per packet, wraps at 65535
uint32 t0_us       // time of the first sample in the packet
uint8  n           // samples in this packet (normally 10)
int16  raw[n][6]   // ax, ay, az, gx, gy, gz, raw MPU-6050 counts
```

- The header is 7 bytes and each sample is 12, so a packet is 7 + 12n bytes: 127 bytes at n = 10, 10 packets per second.
- **Units (converted on the UNO Q):** g = raw / 2048, °/s = raw / 16.4.
- **Timing:** samples are exactly 10 000 µs apart. `t0_us` = (sample count before this packet) × 10 000, wrapped to 32 bits. This is sensor time, not the Pico W's clock.
- **After a FIFO overflow:** the Pico W resets the FIFO and estimates the samples lost from `ticks_us` (time since the last good read ÷ 10 000 µs). It adds them to the sample count, so the next packet's `t0_us` jumps forward by the length of the gap. The UNO Q sees both a `seq` gap and a `t0_us` jump.
- **Small MTU:** n is lowered so that 7 + 12n ≤ MTU − 3. A full 10-sample packet needs an MTU of at least 130. The UNO Q must handle any n from 1 to 10.
- **Resend:** the last 20 packets (2 s) are kept. `RESEND <seq>` re-sends one if it's still there.
- **Heartbeat:** the UNO Q writes `PING` to the control characteristic every 2 s while connected, starting right after it connects. If the Pico W gets no `PING` for 6 s, it disconnects and advertises again. The UNO Q must then reconnect and subscribe again, as after any disconnect.
- The MPU-6050's FIFO is big-endian, so the Pico W byte-swaps every value into the little-endian packet.

## LED patterns

| Pattern | Meaning |
|---|---|
| slow blink (0.5 s on, 0.5 s off) | advertising, waiting for the UNO Q |
| 3 quick blinks | UNO Q just connected |
| off | connected (streaming once the UNO Q subscribes); off saves power |
| stuck on or stuck off, not blinking, with no UNO Q connected | `main.py` crashed or was stopped. Connect Thonny and read the error. |

## Current and battery life

*Fill in the measured values after step 5.*

| State | Current (mA) |
|---|---|
| advertising (LED blinking) | |
| connected and streaming | |

**How to measure:**

1. Set the multimeter to DC **mA**: move the red lead to the mA socket and pick the 200 mA range.
2. Unplug USB and turn the battery switch **off**.
3. Connect the meter **in series** across the switch: red probe on the battery side, black probe on the side that goes to VSYS. With the switch off, all the current now flows through the meter.
4. Watch each state for about 30 s and note a typical value. The reading flickers because the radio transmits in bursts.
5. Afterwards, move the red lead back to the V socket. If you forget, the next voltage reading shorts through the meter.

**Estimating battery life.** At a 50–100 mA drain, alkaline AAA cells give roughly 800–1000 mAh, less than the nominal ~1200 mAh they give at low drain. Two in series give the same mAh at twice the voltage. As the battery voltage falls, the Pico W's regulator draws more current, so plan on about 70 % of the rating:

    life (hours) ≈ 0.7 × 900 mAh ÷ measured streaming current (mA)

For example, 45 mA gives 0.7 × 900 ÷ 45 ≈ 14 hours. You can watch VSYS on the battery characteristic. Swap the cells when it falls below about 2.2 V (2200 mV). The regulator stops working at 1.8 V.

## Troubleshooting

| You see | Likely cause / fix |
|---|---|
| `I2C scan found: []` | No sensor on the bus. Check VCC is on pin 36 (not 37, 3V3_EN, which is right next to it), GND, and SDA/SCL not swapped. Check the GY-521's power LED is on. |
| `OSError: [Errno 5] EIO` or `[Errno 110] ETIMEDOUT` | I2C wire loose mid-run. Reseat the jumpers. Long or thin wires to the ankle may need `I2C_FREQ = 100_000` in `config.py`. |
| `WHO_AM_I mismatch: read 0x70` (or 0x72, 0x98…) | Clone chip, not a genuine MPU-6050. If the readings look right, add the value to `WHO_AM_I_OK` in `config.py`. |
| `register 0x1B reads 0x00, wrote 0x18` | A write didn't stick, usually a bad connection. Reseat the wires and run again. |
| `ImportError: no module named 'config'` | `config.py` isn't saved on the Pico, or has a different name. Check **View → Files**. |
| `|a|` far from 1.00 at rest | Normal is about 0.95 to 1.05 (uncalibrated). A value far off, like 0.5 or 2.0, suggests wrong `ACCEL_LSB_PER_G` or a clone with different scaling. |
| Thonny: "Device is busy or does not respond" | `main.py` is running. Press Ctrl+C or Stop (see **Stopping `main.py`**). |
| No `bluetooth` module | Wrong firmware. Reinstall choosing the **Pico W** variant. |
| `ImportError: no module named 'ble_node'` | `ble_node.py` isn't on the Pico. Upload it. |
| t2: samples far below 6000, or `overflows` > 0 | Something stalls the loop for more than 0.85 s. Make sure no other script, such as `main.py`, is running. |
| t2: `overflow detected = False` | The overflow flag wasn't set. Paste the full output. |
| `ble_check.py`: `NOT FOUND` | The node isn't advertising (LED not slow-blinking), or the phone is still connected, so close nRF Connect. On Windows, check that Bluetooth is on in Settings. |
| `ble_check.py` connects but gets no packets | Thonny should show `streaming STARTED`. If it doesn't, the node isn't seeing the subscription. Paste the `[ble]` lines. |
| `mtu=23 n=1` and many `notify_fail` | The laptop or UNO Q refused a larger MTU. Data still flows, but in 1-sample packets. On the UNO Q side, accept the MTU request. |
| `notify_fail` climbing steadily | The BLE send queue is full because the link is weak or slow. Move closer. If the battery reading might be to blame, set `BATTERY_READ_ENABLED = False`. |
| `no PING for 6000 ms: disconnecting …` while the central is running fine | The central isn't sending `PING` every 2 s. Check the UNO Q code (or that you're running the updated `ble_check.py`). nRF Connect doesn't send PINGs, so it gets dropped after 6 s: set `PING_TIMEOUT_MS = 0` in `config.py` while testing with it. |
| Node stuck "connected" (LED off, `conn=True`) after the UNO Q app restarts, and the app can't find it | Heartbeat is off (`PING_TIMEOUT_MS = 0`) or `config.py` on the Pico is out of date. Set it to 6000 and upload `config.py`, `ble_node.py` and `main.py` again. |
| Battery always `0 mV` | Expected if `BATTERY_READ_ENABLED = False`. Otherwise, paste the output. |
| Bluetooth drops about every 10 s | The VSYS reading may be disturbing the radio. Set `BATTERY_READ_ENABLED = False` in `config.py`. |
| `main.py crashed:`, then it reboots every few seconds | Read the error printed before `rebooting in 5 s`. To keep it stopped, press Ctrl+C during the 3 s startup window, set `REBOOT_ON_CRASH = False`, and paste the error. |
| **Battery warm, leaking or swollen** | USB was plugged in with the battery switch ON. Remove those cells now. Always switch the battery off before plugging in USB. |

## Sources

- InvenSense, *MPU-6000 and MPU-6050 Register Map and Descriptions*, RM-MPU-6000A-00 Rev 4.0 (2012), via [SparkFun](https://cdn.sparkfun.com/datasheets/Sensors/Accelerometers/RM-MPU-6000A.pdf). Used for every register, bit, scale factor and the FIFO byte order.
- Raspberry Pi, [*Raspberry Pi Pico W Datasheet*](https://datasheets.raspberrypi.com/picow/pico-w-datasheet.pdf) (release 7, build date 03/07/2026). Used for the power pins, VSYS range and VSYS/3 sensing on ADC3 shared with the wireless SPI clock.
- Raspberry Pi, [*Pico W pinout*](https://datasheets.raspberrypi.com/picow/PicoW-A4-Pinout.pdf).
- MicroPython docs, [`bluetooth` module](https://docs.micropython.org/en/latest/library/bluetooth.html) (checked 2026-10-22). Used for the IRQ event codes, characteristic flags, the shape of `gatts_register_services`' return value, `gatts_notify` (sends whether or not the central subscribed), `gatts_set_buffer` (default 20 bytes), and `config(mtu=…)` / `gattc_exchange_mtu`.
