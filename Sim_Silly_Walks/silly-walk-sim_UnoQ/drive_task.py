"""mjlab task: teach the Onshape two-wheel robot to drive in a straight line down +x.

The robot starts at the origin pointing a little off to one side (random yaw) and must drive forward
fast while staying on the line y = 0 and keeping its heading along +x. The policy sees its own
velocities, its heading error and its sideways offset, and outputs one wheel speed for each wheel.

Importing this file registers the task "Drive-Straight" (see train.py / play.py next to it).
Everything you will want to tune is in the constants block and the reward weights.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  base_ang_vel,
  base_lin_vel,
  bad_orientation,
  joint_vel_rel,
  last_action,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---- constants you may want to change ----------------------------------------------------------------
# the Onshape export: next to this file when run by runIt.py, one folder up in the repo
_HERE = Path(__file__).resolve().parent
ROBOT_XML = next(p for p in (_HERE / "onshape" / "robot.xml", _HERE.parent / "onshape" / "robot.xml") if p.exists())
LEFT, RIGHT = "left_wheel", "right_wheel"       # joint names in robot.xml
TARGET_SPEED = 1.0        # m/s the robot is rewarded for going forward
MAX_WHEEL_SPEED = 8.0      # rad/s; a policy output of 1.0 means this wheel speed
WHEEL_KD = 0.5             # how hard the motor pushes toward the wheel speed it was told (N*m per rad/s)
WHEEL_TORQUE_LIMIT = 3.0   # N*m
START_YAW = 0.0             # rad; the robot starts pointing up to this far off +x
MAX_OFF_LINE = 5.0          # m; the episode ends if the robot wanders this far from y = 0
EPISODE_S = 10.0
# -------------------------------------------------------------------------------------------------------

_ROBOT = SceneEntityCfg("robot")


def _get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
  for actuator in list(spec.actuators):     # mjlab adds its own wheel motors (below)
    spec.delete(actuator)
  chassis = spec.body("chassis")
  chassis.add_freejoint()                   # let the robot move around (same as drive.py)
  for g in chassis.geoms:                   # only the wheels should grip; the chassis slides (same as drive.py)
    g.friction[0] = 0.01
  return spec


def _robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.06), joint_pos={"left_wheel": 0.0, "right_wheel": 0}, joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # stiffness 0 = no position target, so the motor only chases the velocity we ask for
        BuiltinPdActuatorCfg(
          target_names_expr=(LEFT, RIGHT), stiffness=0.0, damping=WHEEL_KD, effort_limit=WHEEL_TORQUE_LIMIT
        ),
      ),
    ),
  )


# ---- observations and rewards: how the robot is doing relative to "straight down +x" -------------------


def _yaw(env: ManagerBasedRlEnv) -> torch.Tensor:
  w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sideways distance from this env's start line (its y axis), in metres."""
  robot = env.scene["robot"]
  return robot.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]


def heading_and_offset(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(yaw), cos(yaw), off-line distance]. Shape [num_envs, 3]."""
  yaw = _yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw), _off_line(env)], dim=-1)


#For IMU only build
def heading_only(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(yaw), cos(yaw)]: what an IMU can measure."""
  yaw = _yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


def forward_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for going along +x at TARGET_SPEED: 0 when standing still, 1 at the target, less beyond it, negative backwards.

  It slopes everywhere (a narrow bump around the target would give a robot that starts out still nothing to learn from).
  """
  vx = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
  return torch.clamp(1.0 - torch.square((vx - TARGET_SPEED) / TARGET_SPEED), min=-1.0)


def stay_on_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_off_line(env))


def face_forward(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_yaw(env))


def too_far_off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _off_line(env).abs() > MAX_OFF_LINE


# ---- environment ----------------------------------------------------------------------------------------


def drive_straight_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  #For IMU data only
  actor_terms = {                       # what the real robot can measure with just an IMU
    "ang_vel": ObservationTermCfg(func=base_ang_vel),
    "heading": ObservationTermCfg(func=heading_only),
    "last_action": ObservationTermCfg(func=last_action),
  }
  critic_terms = {                      # training-only helper still sees everything
    "lin_vel": ObservationTermCfg(func=base_lin_vel),
    "ang_vel": ObservationTermCfg(func=base_ang_vel),
    "heading_offset": ObservationTermCfg(func=heading_and_offset),
    "wheel_vel": ObservationTermCfg(func=joint_vel_rel),
    "last_action": ObservationTermCfg(func=last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg(critic_terms),
  }


  '''
  actor_terms = {
    "lin_vel": ObservationTermCfg(func=base_lin_vel),
    "ang_vel": ObservationTermCfg(func=base_ang_vel),
    "heading_offset": ObservationTermCfg(func=heading_and_offset),
    "wheel_vel": ObservationTermCfg(func=joint_vel_rel),
    "last_action": ObservationTermCfg(func=last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg({**actor_terms}),
  }
  '''
  actions: dict[str, ActionTermCfg] = {
    "wheels": JointVelocityActionCfg(
      entity_name="robot", actuator_names=(LEFT, RIGHT), scale=MAX_WHEEL_SPEED, use_default_offset=True
    ),
  }

  events = {
    "reset_pose": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {"yaw": (-START_YAW, START_YAW)}, "velocity_range": {}},
    ),
  }

  rewards = {
    "forward_speed": RewardTermCfg(func=forward_speed, weight=5.0),
    "stay_on_line": RewardTermCfg(func=stay_on_line, weight=0.5),
    "face_forward": RewardTermCfg(func=face_forward, weight=0.5),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.005),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "tipped_over": TerminationTermCfg(func=bad_orientation, params={"limit_angle": math.radians(60.0)}),
    "off_line": TerminationTermCfg(func=too_far_off_line),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": _robot_cfg()},
      num_envs=1,
      env_spacing=4.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="chassis",
      distance=1.2,
      elevation=-25.0,
      azimuth=140.0,
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005)),
    decimation=4,
    episode_length_s=1e10 if play else EPISODE_S,
  )


def drive_straight_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(64, 64), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="drive_straight",
    logger="tensorboard",           # the default, wandb, needs an account and a login
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=500,
  )


register_mjlab_task(
  task_id="Drive-Straight",
  env_cfg=drive_straight_env_cfg(),
  play_env_cfg=drive_straight_env_cfg(play=True),
  rl_cfg=drive_straight_ppo_runner_cfg(),
)
