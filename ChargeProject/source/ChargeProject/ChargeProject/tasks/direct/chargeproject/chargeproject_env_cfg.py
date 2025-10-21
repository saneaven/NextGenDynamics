# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from ChargeProject.tasks.direct.chargeproject.environments import MySceneCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass

from gymnasium import spaces


@configclass
class ChargeprojectEnvCfg(DirectRLEnvCfg):

    #always should be on
    log = True

    # env
    episode_length_s = 60.0
    # - spaces definition
    #action_space = 12
    action_space = 24
    #observation_space = 51
    #observation_space = 87 # without height scanner
    # observation_space = 376 # with height scanner
    observation_space = 722 # with height scanner and lidar
    
    observation_space = spaces.Dict({
        "observations": spaces.Box(-math.inf, math.inf, shape=(118,), dtype=float),
        "height_data": spaces.Box(-math.inf, math.inf, shape=(16, 16), dtype=float),
        "lidar_data": spaces.Box(-math.inf, math.inf, shape=(35, 12, 3), dtype=float)
    })
    state_space = 0 #idk why this is here
    # simulation
    decimation = 2
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation,
        physx=PhysxCfg(
            #gpu_collision_stack_size = 2**27,
            gpu_max_rigid_patch_count = 2**19
        )
    )
    
    # Spider Robot (base, leg_hip_i, leg_middle_i, leg_lower_i, leg_foot_i)
    base_name = "body"
    foot_names = "leg_foot_.*"
    undesired_contact_body_names = "body|leg_upper_.*|leg_middle_.*|leg_lower_.*"
    middle_leg_joint_names = "joint_leg_upper_leg_middle_.*"
    lower_leg_names = "leg_lower_.*"
    lower_leg_joint_names = "joint_leg_middle_leg_lower_.*"
    hip_joint_names = "joint_body_leg_hip_.*"

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

    
    # scene
    scene: MySceneCfg = MySceneCfg(
        num_envs= int(1024 * 1.5),  # 1024
        env_spacing=4.0,
        replicate_physics=True
    )
    
    point_max_distance = 10 #20 #6.0
    point_min_distance = 5 #10 #4.0
    success_tolerance = 1.0 #1  # meters
    time_out_per_target = 30.0  # seconds
    time_out_decrease_per_target = 0.075  # seconds
    base_on_ground_time = 1.0 #seconds before death if base is on ground

    log_targets_reached_max = 10
    log_targets_reached_step = 1

    marker_colors = 57

    # Final rewards
    action_scale = 1
    
    progress_reward_scale = 50.0 * 10.0 # linear version ish
    #progress_reward_scale = 50  * 5 * 5 # 1.5 version
    progress_pow = 1.3
    distance_lookback = 8
    #progress_target_divisor = 7.5
    velocity_alignment_reward_scale = 12.0 * 100.0 # 10.0 #2#6
    # Multiplied by targets hit reward
    reach_target_reward_scale = 750.0
    # forward_vel_reward_scale = 0.0 #1.2#/30
    life_time_reward_scale = 0.002
    # time_penalty_scale = 0.0 #-5
    death_penalty_scale = -1000.0 # -5000.0 # -500.0
    # still_penalty_scale = -5.0 * 4.0
    speed_reward_scale = 1.0 * 50.0
    #lin_vel_reward_scale = 1.5
    #yaw_rate_reward_scale = 0.75
    # z_vel_penalty_scale = -0.001
    jump_penalty_scale = -0.5 * 5.0
    # ang_vel_reward_scale = -0.0375
    joint_torque_reward_scale = -1.0e-07
    joint_accel_reward_scale = -1.0e-08 # idk this works # -1.5e-07
    # dof_vel_reward_scale = 0
    action_rate_reward_scale = -0.0002
    body_angular_velocity_penalty_scale = -1.5 * 20.0
    body_vertical_acceleration_penalty_scale = -0.1 * 20.0
    # feet_air_time_reward_scale = 1.0
    undesired_contact_reward_scale = -1.5 * 8.0 ## -0.75
    flat_orientation_reward_scale = -1.5 * 8.0
