"""Small data types shared between the input sources (IMU, keyboard) and the game."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SwingEvent:
    """One physical swing.

    t        -- time.monotonic() at swing onset (used for hit timing)
    strength -- peak strength normalised to the trigger threshold (1.0 = just tripped)
    source   -- "imu" or "keyboard"

    Spin inputs (see spin.spin_for_swing):
    roll_deg       -- filtered handle roll at onset, relative to the neutral grip (IMU)
    roll_rate_dps  -- signed twist rate with the largest magnitude during the swing window (IMU)
    vertical_g     -- mean up(+)/down(-) dynamic acceleration during the swing window (IMU)
    spin_side/top  -- explicit spin (keyboard); None = derive from the IMU fields
    """
    t: float
    strength: float
    source: str = "imu"
    accel_peak_g: float = 0.0
    gyro_peak_raw: float = 0.0
    roll_deg: float = 0.0
    roll_rate_dps: float = 0.0
    vertical_g: float = 0.0
    spin_side: float | None = None
    spin_top: float | None = None
