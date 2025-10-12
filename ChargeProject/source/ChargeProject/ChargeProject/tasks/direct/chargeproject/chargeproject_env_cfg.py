# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from requests import patch
from sympy import prime
from trimesh import Trimesh
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # noqa isort: skip
from isaaclab_assets.robots.spot import SPOT_CFG  # noqa
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from ChargeProject.tasks.direct.chargeproject.environments import MySceneCfg, ROBOT_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from .spider_robot import SPIDER_CFG



@configclass
class ChargeprojectEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 60.0
    # - spaces definition
    #action_space = 12
    action_space = 24
    #observation_space = 51
    #observation_space = 87 # without height scanner
    # observation_space = 376 # with height scanner
    observation_space = 691 # with height scanner and lidar
    state_space = 0
    # simulation
    decimation = 2
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot: ArticulationCfg = SPIDER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    #robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    #robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Spider Robot (base, leg_hip_i, leg_middle_i, leg_lower_i, leg_foot_i)
    base_name = "body"
    foot_names = "leg_foot_.*"
    undesired_contact_body_names = "body|leg_upper_.*|leg_middle_.*|leg_lower_.*"

    # Unitree Go2
    #base_name = "base"
    #foot_names = ".*_foot"
    #undesired_contact_body_names = ".*_thigh"

    # Spot
    # base_name = "body"
    # foot_names = ".*_foot"
    # undesired_contact_body_names = ".*_uleg"

    # Anymal
    # base_name = "base"
    # foot_names = ".*FOOT"
    # undesired_contact_body_names = ".*THIGH"

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    height_scanner = RayCasterCfg(
        prim_path=f"/World/envs/env_.*/Robot/{base_name}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.6]),  # type: ignore
        debug_vis=False,
        mesh_prim_paths=["/World/terrain"],
    )

    lidar_sensor = RayCasterCfg(
        prim_path=f"/World/envs/env_.*/Robot/{base_name}",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0)),
        mesh_prim_paths=["/World/terrain"],
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=3, vertical_fov_range=(-30.0, 30.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=10.0
        )
    )
    
    # scene
    scene: InteractiveSceneCfg = MySceneCfg()
    point_max_distance = 10 #20 #6.0
    point_min_distance = 5 #10 #4.0
    success_tolerance = 0.25 #1  # meters
    time_out_per_target = 30.0  # seconds
    time_out_decrease_per_target = 0.075  # seconds

    marker_colors = 57

    # Final rewards
    action_scale = 0.15# 0.2
    
    progress_reward_scale = 1000.0 #/7.5
    #progress_target_divisor = 7.5
    velocity_alignment_reward_scale = 10.0 #2#6
    # Multiplied by targets hit reward
    reach_target_reward_scale = 500.0
    forward_vel_reward_scale = 0.0#1.2#/30
    life_time_reward_scale = 0.001
    time_penalty_scale = 0.0 #-5
    death_penalty_scale = -5000.0 # -500
    still_penalty_scale = -5.0 * 4.0
    speed_reward_scale = 0.5
    #lin_vel_reward_scale = 1.5
    #yaw_rate_reward_scale = 0.75
    z_vel_penalty_scale = -0.001
    jump_penalty_scale = -10.0
    ang_vel_reward_scale = -0.0375
    joint_torque_reward_scale = -5e-05
    joint_accel_reward_scale = -1.0e-7 # -1.5e-7
    dof_vel_reward_scale = 0
    action_rate_reward_scale = -0.003
    feet_air_time_reward_scale = 3.0
    undesired_contact_reward_scale = -0.75 * 4.0
    flat_orientation_reward_scale = -1.2
