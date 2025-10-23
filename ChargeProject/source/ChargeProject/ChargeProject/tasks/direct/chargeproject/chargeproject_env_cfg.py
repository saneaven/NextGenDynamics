# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from requests import patch
from sympy import prime
from trimesh import Trimesh
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # noqa isort: skip
from isaaclab_assets.robots.spot import SPOT_CFG  # noqa
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

#from ChargeProject.tasks.direct.chargeproject.environments import MySceneCfg, ROBOT_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from .spider_robot import SPIDER_CFG

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

from gymnasium import spaces



@configclass
class ChargeprojectEnvCfg(DirectRLEnvCfg):

    #always should be on
    log = True

    # env
    episode_length_s = 60.0
    # - spaces definition
    action_space = spaces.Box(-math.inf, math.inf, shape=(6, 4), dtype=float)
    observation_space = spaces.Dict({
        "base_obs": spaces.Box(-math.inf, math.inf, shape=(33, ), dtype=float),
        "leg_obs": spaces.Box(-math.inf, math.inf, shape=(6, 33), dtype=float),
        "height_data": spaces.Box(-math.inf, math.inf, shape=(16, 16), dtype=float)
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
    # robot(s)
    robot: ArticulationCfg = SPIDER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    #robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    #robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
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
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.5, 1.5]),  # type: ignore
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=int(1024*0.75),#0),
        env_spacing=4.0, 
        replicate_physics=True
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    point_max_distance = 10 #20 #6.0
    point_min_distance = 5 #10 #4.0
    success_tolerance = 0.5 # 0.25  # meters
    time_out_per_target = 10 #5.0  # seconds
    time_out_decrease_per_target = 0.075  # seconds
    base_on_ground_time = 1.0 #seconds before death if base is on ground

    log_targets_reached_max = 10
    log_targets_reached_step = 1

    marker_colors = 57

    # Final rewards
    action_scale = 1
    

    # Training stages:
    # Removed scaling progress/velocity rewards by number of targets reached
    # init, 2k steps: Learning rate 1e-4 (learning rate may be able to start at 1e-3)
    #                 joint_leg_middle_leg_lower_ = 125 (may not be needed)
    #                 Stopped training after velocity/progress became noticeable (2025-10-22_19-49-57_ppo_torch, v1.0.0)
    # init, 3k steps: Learning rate to 1e-3
    #                 Stopped training after height ~equal init height (2025-10-22_20-42-36_ppo_torch, v1.0.1)
    # start, 5.5k steps: After init comments below
    #                   joint_leg_middle_leg_lower_ = 160 
    #                   Stopped after it starts "jumping" (2025-10-23_15-28-12_ppo_torch, v1.1.0)
    # main, _k steps: Learning rate to 1e-4
    #                 After start comments below
    

    #  1e-4 then set to 1e-3 for faster learning
    progress_reward_scale = 2000# * 2 # Remove (*2) after init
    progress_pow = 1#.4
    distance_lookback = 10

    velocity_alignment_reward_scale = 120# * 4 # Remove (*4) after init
    # Multiplied by targets hit reward
    reach_target_reward_scale = 1000
    death_penalty_scale = -2000
    movement_reward_scale = 30
    z_vel_reward_scale = -120 * 3 # add (*3) after start
    ang_vel_reward_scale = -2.7

    joint_torque_reward_scale = -0.0015 
    joint_accel_reward_scale = -5.3e-07 / 6 # add /6 after start
    dof_vel_reward_scale = -0.006 / 8 # add /8 after start
    action_rate_reward_scale = -1.5 / 2 # add /2 after start

    feet_air_time_reward_scale = 26.6
    feet_air_time_target = 0.4 # set to 0.4 after start (was 0.7 but not sure if this matters)
    feet_ground_time_reward_scale = 40
    feet_ground_time_target = 0.4 # set to 0.4 after start (was 0.7 but not sure if this matters)
    
    undesired_contact_reward_scale = -100
    undesired_contact_time_reward_scale = -15
    desired_contact_reward_scale = 10 * 4 # add (*4) after start
    flat_orientation_reward_scale = -1200
    body_height_reward_scale = 114 / 2# * 4 # Remove (*4) after init # add (/2) after start
    lower_leg_reward_scale = 200 / 10 # add (/10) after start
    hip_penalty_scale = -30
    feet_under_body_penalty_scale = -72000 * 2 # add (*2) after start
    body_penalty_radius = 0.175

    # rewards positive joint velocity when time from contact
    step_reward_scale = 50 * 4 # Set 50 after init (from 0) # Add (*4) after start
    step_up_time_end = 0.2 # set to 0.2 after start (was 0.55)
    # linear scale of penalty if leg doesn't step in this time
    step_length_penalty_scale = -15
    step_penalty_start = 1.3
    step_penalty_cap = 2.0
    grounded_length_penalty_scale = -15
    grounded_penalty_start = 2.0
    grounded_penalty_cap = 2.0
    feet_up_step_time_penalty_scale = -60 # set -40 after start (from 0)
    feet_down_step_time_penalty_scale = -60 # set -40 after start (from 0)
    feet_step_time_multiplier = 2.0 # makes more going up than being on ground
    feet_step_time_target = 0.4
    feet_step_time_leeway = 0.6 # clamped out on positive

    joint_default_penalty = 0

