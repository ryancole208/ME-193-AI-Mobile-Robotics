"""
Q-learning "walker" for a LEGO Education Double Motor.

Each side of the Double Motor drives a continuously spinning leg. The robot
learns which left/right speed combination keeps it walking straight
forward, using the Double Motor's internal IMU yaw as feedback:

    state  = which yaw-error bin the robot is in (how far it has turned)
    action = a (left speed, right speed) pair - straight, or a steer
    reward = high when the yaw error is near zero, negative as it grows,
             plus a bonus for reducing the error

Learning loop (repeats until Ctrl+C or MAX_STEPS):
    1. Roll a random number. If it is below the exploration rate, pick a
       random action (explore); otherwise pick the action with the
       highest Q for the current state (exploit).
    2. Execute the action for STEP_TIME seconds.
    3. Read the new yaw, compute the reward, update Q.
    4. Multiply the exploration rate by EPSILON_DECAY (0.98).

Hardware: LEGO Education Double Motor, identified by its Azure connection
card, serial 1096.
"""

import json
import os
import random
import time

import legoeducation as le

# --- Q-learning constants ---
LEARNING_RATE = 0.3       # alpha: how much each new experience overwrites the old Q
DISCOUNT_FACTOR = 0.8     # gamma: how much future reward matters vs. immediate reward
EPSILON_START = 0.9       # initial exploration rate (90% random actions at first)
EPSILON_DECAY = 0.998      # exploration rate is multiplied by this after every step

# --- Reward constants ---
ON_TRACK_DEG = 3.0        # |yaw error| below this counts as walking straight
REWARD_ON_TRACK = 10.0    # reward for being on track
REWARD_PER_DEG = -0.5     # penalty per degree of |yaw error| when off track
REWARD_IMPROVE = 2.0      # bonus if |yaw error| shrank this step (penalty if it grew)

# --- State bins (yaw error, degrees) ---
# Bin edges split the error into 7 states:
#   0: < -20   1: -20..-10   2: -10..-3   3: -3..3 (on track)
#   4: 3..10   5: 10..20     6: > 20
STATE_EDGES = [-20.0, -10.0, -ON_TRACK_DEG, ON_TRACK_DEG, 10.0, 20.0]
NUM_STATES = len(STATE_EDGES) + 1

# --- Actions: (left leg speed %, right leg speed %) ---
BASE_SPEED = 50
SMALL_STEER = 15
BIG_STEER = 30
ACTIONS = [
    (BASE_SPEED, BASE_SPEED),                              # 0: straight
    (BASE_SPEED - SMALL_STEER, BASE_SPEED + SMALL_STEER),  # 1: slight left
    (BASE_SPEED + SMALL_STEER, BASE_SPEED - SMALL_STEER),  # 2: slight right
    (BASE_SPEED - BIG_STEER, BASE_SPEED + BIG_STEER),      # 3: hard left
    (BASE_SPEED + BIG_STEER, BASE_SPEED - BIG_STEER),      # 4: hard right
]
ACTION_NAMES = ["straight", "slight left", "slight right", "hard left", "hard right"]
NUM_ACTIONS = len(ACTIONS)

# --- Hardware identification (Azure card, serial 1096) ---
CARD_COLOR = le.LEGO_COLOR_AZURE
CARD_SERIAL = "1096"

# --- Hardware / timing ---
# The two motors usually face opposite directions on a walker, so one must
# spin the "other way" for both legs to move forward. If a leg walks
# backward, flip its sign here.
LEFT_FORWARD_SIGN = 1
RIGHT_FORWARD_SIGN = -1
# NOTE: if the yaw value doesn't change when you rotate the robot on the
# table, the Double Motor is mounted on a different face - change this.
YAW_FACE = le.DEVICE_FACE_TOP
# NOTE: if "steer left" makes the yaw error grow instead of shrink, the
# yaw sign is opposite to this build's steering - set this to -1.
YAW_SIGN = 1
STEP_TIME = 0.5           # seconds each action runs before measuring yaw
MAX_STEPS = 300           # stop after this many steps (None = run until Ctrl+C)
Q_TABLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.json")


def wrap_angle(deg):
    """Wrap an angle to [-180, 180) so crossing +/-180 doesn't jump."""
    return (deg + 180.0) % 360.0 - 180.0


def read_yaw_error(doublemotor):
    """Yaw error relative to the heading at start (0 deg). Waits for a
    valid (non-NaN) IMU notification."""
    yaw = doublemotor.imu_device.yaw
    while yaw != yaw:  # NaN guard: no IMU notification yet
        time.sleep(0.01)
        yaw = doublemotor.imu_device.yaw
    return YAW_SIGN * wrap_angle(yaw)


def get_state(yaw_error):
    """Map a yaw error (deg) to a state index 0..NUM_STATES-1."""
    for i, edge in enumerate(STATE_EDGES):
        if yaw_error < edge:
            return i
    return NUM_STATES - 1


def get_reward(prev_error, new_error):
    """Reward for landing at new_error after starting from prev_error."""
    if abs(new_error) < ON_TRACK_DEG:
        reward = REWARD_ON_TRACK
    else:
        reward = REWARD_PER_DEG * abs(new_error)

    if abs(new_error) < abs(prev_error):
        reward += REWARD_IMPROVE
    elif abs(new_error) > abs(prev_error):
        reward -= REWARD_IMPROVE
    return reward


def choose_action(q_table, state, epsilon):
    """Roll a random number: explore if below epsilon, else exploit."""
    roll = random.random()
    if roll < epsilon:
        return random.randrange(NUM_ACTIONS), "explore"
    row = q_table[state]
    best = max(row)
    # Break ties randomly so an all-zero row doesn't always pick action 0
    return random.choice([a for a, q in enumerate(row) if q == best]), "exploit"


def execute_action(doublemotor, action):
    left, right = ACTIONS[action]
    doublemotor.motor_set_speed(LEFT_FORWARD_SIGN * left, motor=le.MOTOR_LEFT)
    doublemotor.motor_set_speed(RIGHT_FORWARD_SIGN * right, motor=le.MOTOR_RIGHT)
    doublemotor.motor_run(motor=le.MOTOR_BOTH, blocking=False)
    time.sleep(STEP_TIME)


def update_q(q_table, state, action, reward, next_state):
    """Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))"""
    target = reward + DISCOUNT_FACTOR * max(q_table[next_state])
    q_table[state][action] += LEARNING_RATE * (target - q_table[state][action])


def load_q_table():
    if os.path.exists(Q_TABLE_FILE):
        with open(Q_TABLE_FILE) as f:
            q_table = json.load(f)
        if len(q_table) == NUM_STATES and all(len(r) == NUM_ACTIONS for r in q_table):
            print(f"Loaded Q-table from {Q_TABLE_FILE}")
            return q_table
        print("Saved Q-table shape doesn't match - starting fresh.")
    return [[0.0] * NUM_ACTIONS for _ in range(NUM_STATES)]


def save_q_table(q_table):
    with open(Q_TABLE_FILE, "w") as f:
        json.dump(q_table, f, indent=2)
    print(f"Saved Q-table to {Q_TABLE_FILE}")


def print_q_table(q_table):
    header = "state".ljust(8) + "".join(name.rjust(14) for name in ACTION_NAMES)
    print(header)
    for s, row in enumerate(q_table):
        print(str(s).ljust(8) + "".join(f"{q:14.2f}" for q in row))


def train(doublemotor, q_table):
    epsilon = EPSILON_START
    yaw_error = read_yaw_error(doublemotor)
    state = get_state(yaw_error)
    step = 0

    print("Training. Press Ctrl+C to stop.")
    while MAX_STEPS is None or step < MAX_STEPS:
        action, mode = choose_action(q_table, state, epsilon)
        execute_action(doublemotor, action)

        new_error = read_yaw_error(doublemotor)
        next_state = get_state(new_error)
        reward = get_reward(yaw_error, new_error)
        update_q(q_table, state, action, reward, next_state)

        print(f"step {step:4d} | eps {epsilon:.3f} {mode:7s} | "
              f"s {state} -> {next_state} | {ACTION_NAMES[action]:12s} | "
              f"yaw err {new_error:7.1f} | r {reward:6.1f}")

        epsilon *= EPSILON_DECAY
        yaw_error, state = new_error, next_state
        step += 1


def main():
    doublemotor = le.DoubleMotor()

    print("Connecting to Double Motor (Azure card, serial 1096)...")
    doublemotor.connect(
        card_color=CARD_COLOR,
        card_serial=CARD_SERIAL,
        device_notification_delay=50,  # IMU update every 50 ms
    )
    if not doublemotor.connected:
        print("Could not connect. Check the hub is powered on and the "
              "Azure card (1096) is attached.")
        return

    q_table = load_q_table()
    try:
        doublemotor.imu_set_yaw_face(YAW_FACE)
        print("Point the robot in the direction it should walk...")
        time.sleep(2.0)
        doublemotor.imu_reset_yaw_axis(0)  # current heading becomes the target
        time.sleep(0.2)

        train(doublemotor, q_table)
    except KeyboardInterrupt:
        pass
    finally:
        doublemotor.motor_stop(motor=le.MOTOR_BOTH)
        doublemotor.disconnect()
        save_q_table(q_table)
        print_q_table(q_table)


if __name__ == "__main__":
    main()
