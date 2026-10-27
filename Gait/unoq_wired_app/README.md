# Gait Classifier, wired IMU test version

This is the same app as `unoq_app` (the Bluetooth version), but the MPU-6050 is wired straight
to the UNO Q. There is no Pico W and no Bluetooth. It has the same one-time setup, both kNN models
(A "1D" and B "Full IMU"), live mode with smoothing, tap gestures, matrix animations, logging,
replay, evaluate and `diag`. The battery screens are gone, because there is no battery.

The two apps share no files, so either one can be changed or deleted on its own. Only one App
runs at a time: starting this one stops the Bluetooth app, and vice versa.

## Quick start

1. **Wire it** (power off) as in the table below.
2. **Deploy:** `arduino-app-cli app start ~/ArduinoApps/unoq_wired_app`. The matrix flashes **W**.
3. **Run diag** with the IMU lying flat and still: `python3 ~/ArduinoApps/unoq_wired_app/tools/wired.py diag`.
   It ends with a PASS/FAIL table in the log and in `reports/diag.md`.
4. **Set up:** restart normally (`arduino-app-cli app restart ~/ArduinoApps/unoq_wired_app`).
   With no `config_wired.json`, the matrix shows **CFG ?** and setup starts. Follow the screens
   exactly as in the Bluetooth app's README: stand still, then S, gesture check, W, R, B.
5. **Use it.** The stick figure shows the activity. 4 heel taps switch between `1D` and `6D`.

## Wiring

Use the GY-521 / MPU-6050 breakout, connected to the UNO Q's **3.3 V** header pins.

| MPU-6050 | UNO Q | MCU pin |
|---|---|---|
| VCC | 3.3V (power header) | – |
| GND | GND | – |
| SDA | SDA, the pin next to AREF (D20) | PB11 |
| SCL | SCL, the pin next to SDA (D21) | PB10 |
| AD0, INT, XDA, XCL | leave unconnected | – |

- These header pins are the sketch's **`Wire`** bus (I2C2). The choice is the constant
  `IMU_WIRE` at the top of `sketch/sketch.ino`.
- To use the **Qwiic** connector instead (3.3 V only), set `IMU_WIRE` to `Wire1`.
- A4/A5 are `Wire2`, which needs pull-up resistors to 3.3 V.
- The LED matrix uses PF0–PF10, so none of these buses conflicts with it.
- All UNO Q MCU headers are 3.3 V logic. Never feed the breakout 5 V when it's wired to them.

## Cable length

The bus runs at **100 kHz** by default (`IMU_BUS_HZ` in the sketch), so a 0.5–1 m cable to the
ankle works. On a long cable:
- keep it as short as you can;
- twist SDA with a GND wire and SCL with another GND wire, or use a 4-core cable with shielding to GND;
- tape it along the leg so it doesn't flex at the connector;
- don't raise the speed to 400 kHz unless the wires are under about 20 cm.

If the cable is disturbed, I2C errors are counted and the sensor is re-initialised
automatically within about a second, without rebooting. The gap shows up in the stream and
in the logs.

## Gestures

Gestures are heel (or toe) taps with the instrumented foot, standing still before and after.

| Gesture | How | Effect |
|---|---|---|
| Switch | 4 taps, 150–450 ms apart | switches model A / B (the matrix shows `1D` or `6D`); remembered after a reboot |
| Redo setup | 8 taps within 30 s of the IMU data first arriving, then **4 more within 5 s** of `SET?` | backs up `config_wired.json` and records setup again; anything else cancels |
| (8 taps later) | – | no action (the Bluetooth app shows the battery here) |

A blinking dot at the bottom of column 11 means the redo-setup window is open. Every gesture
and near-miss goes to `logs/gestures-*.csv`.

## Matrix screens

| Screen | Meaning |
|---|---|
| `W` for 1 s at start | the wired app is running |
| stick figure: standing / walking / running / biking | the classified activity |
| column 12, top 2 pixels / bottom 2 pixels | model A (`1D`) / model B (`6D`) is shown |
| `IMU`, then a big `X`, repeating | no IMU samples for more than 2 s; check the wiring |
| big `?` | not enough data, gaps, or "unsure" |
| `CFG` then `?` | no usable `config_wired.json` (missing, invalid, or not from a wired setup), so setup starts |
| `LEG` then `?` | `setup.yaml` names a different leg than setup used |
| letter + countdown or bar, `G` + dots, spinner, ✓, scrolling `A 87%` / `B 91%` | setup screens |
| `D` + number, then ✓ or `X` | diag running, then passed or failed |
| scrolling `SET?` | confirm redo-setup with 4 taps |

## Other commands

```bash
python3 tools/wired.py mode setup        # redo setup on purpose
python3 tools/wired.py mode holdout      # record test data into data/holdout/
python3 tools/wired.py evaluate          # reports/evaluate.md for both models
python3 tools/wired.py replay logs/raw-20261022-101500.csv
python3 tools/wired.py mode "replay logs/raw-20261022-101500.csv"   # replay on the real matrix
```

Other ways to redo setup: delete or rename `config_wired.json` and restart.

- **Logs:** `logs/run-*.csv` has both models' predictions per window. Only the newest 20
  files are kept.
- **Setup recordings:** stored in this app's own `data/setup/`.

## Assumptions

1. **Bus:** the header SDA/SCL pins are `Wire` = I2C2 (PB11/PB10), as given in the UNO Q user
   manual and the board's devicetree. The Qwiic connector is `Wire1` (I2C4).
2. **Pull-ups:** the GY-521 breakout's own pull-up resistors (to its regulator output) are
   used. No extra resistors are added.
3. **Sensor setup**, checked against the MPU-6000/6050 Register Map rev 4.0:
   - `PWR_MGMT_1` = 0x80 (reset), then 0x01 (PLL with X-gyro clock);
   - `SMPLRT_DIV` = 9 and `CONFIG` = 0x03 (DLPF_CFG 3), giving 1 kHz / 10 = 100 Hz;
   - `GYRO_CONFIG` = 0x18 (±2000 °/s) and `ACCEL_CONFIG` = 0x18 (±16 g);
   - `FIFO_EN` = 0x78, so accel and gyro make 12 bytes per sample;
   - `USER_CTRL` sets FIFO_EN, and FIFO_RESET is only used while the FIFO is off.

   Every write is read back to verify it.
4. **Overflow handling:** a FIFO overflow (`INT_STATUS` bit 4), a nearly full FIFO, or a byte
   count that isn't whole samples makes the sketch reset the FIFO. The samples lost since the
   last read (elapsed time / 10 ms) are added to the sample count, so `t0_us` jumps while `seq`
   stays continuous, exactly as the Pico W does. Recovering from an I2C error is handled the
   same way.
5. **Bridge transfer:** each packet goes to Python as one binary argument
   (`std::vector<uint8_t>` becomes a MsgPack bin, which arrives as Python `bytes`). It is sent
   with `Bridge.notify("imu_packet", ...)` at about 10 per second, the same way the Bridge
   library sends its own TCP/UDP payloads. `imu_status` is notified once per second with the
   sensor state, WHO_AM_I, I2C error count, re-inits, FIFO overflows and bus speed.
6. **A non-0x68 WHO_AM_I** (a clone chip) still streams, but diag check 1 fails and the log warns.
7. **Startup "W" and no-data screen:** the "W" is drawn by the sketch itself, so it appears even
   before Python starts. "IMU X" is also the screen until the first samples arrive.
8. **The 8-tap long gesture** does nothing outside the redo-setup window, because there is no
   battery to show.
9. **The redo-setup window** counts from the first IMU data after the app starts (the wired
   equivalent of "the node first connects").
10. **No tests or simulators are kept,** per the brief. The code was checked by reading it, by
    compiling the sketch, and with a one-off simulated run that was deleted afterwards.
11. **Each one-shot mode** (diag, setup, holdout, replay) is selected with a `RUN_MODE` file,
    because App Lab apps take no command-line arguments. `tools/wired.py` writes it and
    restarts the app.
