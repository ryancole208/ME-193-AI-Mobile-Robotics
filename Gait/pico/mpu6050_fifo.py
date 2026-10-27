# mpu6050_fifo.py - small MPU-6050 driver built around the sensor's FIFO.
#
# Why the FIFO: MicroPython can pause for several milliseconds at random
# moments (garbage collection, Bluetooth work). If the Pico W timed each
# sample, those pauses would make the samples uneven. Instead the MPU-6050
# samples itself at exactly 100 Hz and stores each sample in its 1024-byte
# FIFO; the Pico W just empties the FIFO every ~100 ms.
#
# Source for every register address and bit below:
#   InvenSense, "MPU-6000 and MPU-6050 Register Map and Descriptions",
#   document RM-MPU-6000A-00, Revision 4.0, 03/09/2012.
# "reg N" in the comments is the register's decimal number in that document.

import struct
import time
from micropython import const

# --- Register addresses ----------------------------------------------------
_SMPLRT_DIV = const(0x19)    # reg 25: sample rate = gyro rate / (1 + value)
_CONFIG = const(0x1A)        # reg 26: DLPF_CFG in bits 2:0
_GYRO_CONFIG = const(0x1B)   # reg 27: FS_SEL in bits 4:3
_ACCEL_CONFIG = const(0x1C)  # reg 28: AFS_SEL in bits 4:3
_FIFO_EN = const(0x23)       # reg 35: which sensors go into the FIFO
_INT_ENABLE = const(0x38)    # reg 56
_INT_STATUS = const(0x3A)    # reg 58: bits clear when this register is read
_ACCEL_XOUT_H = const(0x3B)  # reg 59: first of 14 data bytes (regs 59-72)
_USER_CTRL = const(0x6A)     # reg 106
_PWR_MGMT_1 = const(0x6B)    # reg 107
_FIFO_COUNTH = const(0x72)   # reg 114: must be read before FIFO_COUNTL
_FIFO_R_W = const(0x74)      # reg 116: read this to pop bytes off the FIFO
_WHO_AM_I = const(0x75)      # reg 117: 0x68 on a genuine MPU-6050

# --- Bit values ------------------------------------------------------------
# FIFO_EN (reg 35): bit7 TEMP, bit6 XG, bit5 YG, bit4 ZG, bit3 ACCEL.
# Gyro X/Y/Z + accel, no temperature -> 0b0111_1000.
_FIFO_GYRO_ACCEL = const(0x78)
# USER_CTRL (reg 106): bit6 FIFO_EN, bit2 FIFO_RESET. FIFO_RESET only works
# while FIFO_EN is 0, and clears itself.
_USER_FIFO_EN = const(0x40)
_USER_FIFO_RESET = const(0x04)
# PWR_MGMT_1 (reg 107): bit7 DEVICE_RESET, bit6 SLEEP, bits 2:0 CLKSEL.
# CLKSEL = 1 is "PLL with X axis gyroscope reference"; SLEEP = 0 wakes it.
_PWR_DEVICE_RESET = const(0x80)
_PWR_WAKE_CLK_XGYRO = const(0x01)
# INT_ENABLE / INT_STATUS (regs 56/58): bit4 FIFO_OFLOW.
_INT_FIFO_OFLOW = const(0x10)

FIFO_SIZE = const(1024)       # bytes
BYTES_PER_SAMPLE = const(12)  # 6 values x 2 bytes

# Full-scale range -> 2-bit select value (reg 27 and reg 28 tables)
_ACCEL_FS_SEL = {2: 0, 4: 1, 8: 2, 16: 3}
_GYRO_FS_SEL = {250: 0, 500: 1, 1000: 2, 2000: 3}


class MPU6050:
    """MPU-6050 on an I2C bus.

    FIFO byte order (reg 116: "Data is written to the FIFO in order of
    register number"): with accel and gyro enabled and temperature disabled,
    each 12-byte sample is
        ACCEL_X ACCEL_Y ACCEL_Z GYRO_X GYRO_Y GYRO_Z
    as signed 16-bit BIG-endian values (high byte first).
    """

    def __init__(self, i2c, addr=0x68, max_samples=10, who_am_i_ok=(0x68,)):
        self.i2c = i2c
        self.addr = addr
        self.max_samples = max_samples
        self.who_am_i_ok = who_am_i_ok
        # All buffers are made once here, so reading never allocates memory.
        self._one = bytearray(1)
        self._two = bytearray(2)
        self._data = bytearray(14)
        buf = memoryview(bytearray(BYTES_PER_SAMPLE * max_samples))
        # _views[k] is exactly k samples long (slicing a memoryview inside
        # the loop would allocate, so all slices are prepared up front).
        self._views = [buf[:BYTES_PER_SAMPLE * k] for k in range(max_samples + 1)]

    # --- low-level register access ----------------------------------------

    def _write(self, reg, value):
        self._one[0] = value
        self.i2c.writeto_mem(self.addr, reg, self._one)

    def _read_u8(self, reg):
        self.i2c.readfrom_mem_into(self.addr, reg, self._one)
        return self._one[0]

    # --- setup -------------------------------------------------------------

    def who_am_i(self):
        return self._read_u8(_WHO_AM_I)

    def check_identity(self):
        """Raise RuntimeError if WHO_AM_I is not an expected value."""
        wid = self.who_am_i()
        if wid not in self.who_am_i_ok:
            raise RuntimeError(
                "MPU-6050 WHO_AM_I mismatch: read 0x%02X, expected one of %s. "
                "If this is a clone chip that otherwise works, add 0x%02X to "
                "WHO_AM_I_OK in config.py." % (wid, [hex(v) for v in self.who_am_i_ok], wid))
        return wid

    def configure(self, dlpf_cfg=3, smplrt_div=9, accel_range_g=16, gyro_range_dps=2000):
        """Reset and set up the sensor. Does NOT start the FIFO (see fifo_start).

        Raises OSError on I2C trouble and RuntimeError if the chip is not
        recognised or a setting does not read back correctly.
        """
        self.check_identity()
        self._write(_PWR_MGMT_1, _PWR_DEVICE_RESET)   # all registers to default
        time.sleep_ms(100)                            # let the reset finish
        self._write(_PWR_MGMT_1, _PWR_WAKE_CLK_XGYRO)  # wake + gyro X clock
        time.sleep_ms(50)                             # PLL settles

        settings = (
            (_CONFIG, dlpf_cfg & 0x07),
            (_SMPLRT_DIV, smplrt_div),
            (_GYRO_CONFIG, _GYRO_FS_SEL[gyro_range_dps] << 3),
            (_ACCEL_CONFIG, _ACCEL_FS_SEL[accel_range_g] << 3),
            (_INT_ENABLE, _INT_FIFO_OFLOW),  # so INT_STATUS reports overflow
        )
        for reg, value in settings:
            self._write(reg, value)
        # Read every setting back: catches a loose wire that drops writes.
        for reg, value in settings:
            got = self._read_u8(reg)
            if got != value:
                raise RuntimeError("MPU-6050 register 0x%02X reads 0x%02X, wrote 0x%02X"
                                   % (reg, got, value))

    # --- direct reading (used by tests/t1_imu.py) ---------------------------

    def read_raw(self):
        """Latest sample straight from the data registers (not the FIFO).

        Returns (ax, ay, az, temp, gx, gy, gz) in raw counts. Allocates a
        tuple, so use it for tests only, not in the streaming loop.
        """
        self.i2c.readfrom_mem_into(self.addr, _ACCEL_XOUT_H, self._data)
        return struct.unpack(">7h", self._data)

    # --- FIFO ----------------------------------------------------------------

    def fifo_start(self):
        """Empty the FIFO and (re)start filling it with accel + gyro samples.

        Also used to recover after an overflow, because an overflow drops the
        oldest bytes and can leave the FIFO out of step with 12-byte samples.
        """
        self._write(_USER_CTRL, 0)                 # FIFO off (needed for reset)
        self._write(_FIFO_EN, 0)
        self._write(_USER_CTRL, _USER_FIFO_RESET)  # empty it; bit self-clears
        time.sleep_ms(1)
        self._read_u8(_INT_STATUS)                 # clear any old overflow flag
        self._write(_FIFO_EN, _FIFO_GYRO_ACCEL)
        self._write(_USER_CTRL, _USER_FIFO_EN)

    def fifo_count(self):
        """Number of BYTES waiting in the FIFO (high byte read first)."""
        self.i2c.readfrom_mem_into(self.addr, _FIFO_COUNTH, self._two)
        return (self._two[0] << 8) | self._two[1]

    def fifo_overflowed(self):
        """True if the FIFO overflowed since the last call (reading clears it)."""
        return bool(self._read_u8(_INT_STATUS) & _INT_FIFO_OFLOW)

    def healthy(self):
        """True if the chip still answers and the FIFO is still running.

        A brown-out or a glitch on a loose wire can reset the MPU-6050 to its
        defaults (asleep, FIFO off) without any I2C error, so main.py calls
        this about once a second. Raises OSError if the bus itself fails.
        """
        if self._read_u8(_WHO_AM_I) not in self.who_am_i_ok:
            return False
        if not self._read_u8(_USER_CTRL) & _USER_FIFO_EN:
            return False
        if self._read_u8(_FIFO_EN) != _FIFO_GYRO_ACCEL:
            return False
        return True

    def fifo_read(self, k):
        """Pop exactly k whole samples (1..max_samples) off the FIFO.

        Returns a memoryview of k*12 bytes in the big-endian order described
        in the class docstring. The view is reused by the next call, so copy
        or pack the data before calling again.
        """
        view = self._views[k]
        # A burst read of FIFO_R_W keeps reading the FIFO (the address does
        # not advance), so one I2C transaction fetches all k samples.
        self.i2c.readfrom_mem_into(self.addr, _FIFO_R_W, view)
        return view
