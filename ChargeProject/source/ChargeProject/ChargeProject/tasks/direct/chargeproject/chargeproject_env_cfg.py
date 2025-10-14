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
    observation_space = 376 # with height scanner
    state_space = 376
    # simulation
    decimation = 2
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation,
        physx=PhysxCfg(
            gpu_collision_stack_size = 2**27,
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
    lower_leg_names = "leg_lower_.*"

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
        mesh_prim_paths=["/World/ground"],
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=int(1024*8),
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
    success_tolerance = 1 # 0.25  # meters
    time_out_per_target = 25 #5.0  # seconds
    time_out_decrease_per_target = 0.075  # seconds
    death_velocity_threshold = 20000.0 # m/s

    log_targets_reached_max = 10
    log_targets_reached_step = 1

    marker_colors = 57

    # Final rewards
    action_scale = 0.15# 0.2
    
    progress_reward_scale = 50 *2 * 5 *2#* 4
    #progress_reward_scale = 50  * 5 * 5
    progress_pow = 1.2
    distance_lookback = 8
    #progress_target_divisor = 7.5
    velocity_alignment_reward_scale = 10 * 6 #2#6
    # Multiplied by targets hit reward
    reach_target_reward_scale = 500
    forward_vel_reward_scale = 0
    time_penalty_scale = 0
    death_penalty_scale = -200
    still_penalty_scale = -5    * 8
    #lin_vel_reward_scale = 1.5
    #yaw_rate_reward_scale = 0.75
    z_vel_reward_scale = 0# -30  * 4   / 6
    ang_vel_reward_scale = -0.0375 * 10    * 18
    joint_torque_reward_scale = -5e-05 * 50     / 2 * 2#   / 20
    joint_accel_reward_scale = -1.5e-7 / 3   * 1.5    / 4 * 4
    dof_vel_reward_scale = 0
    action_rate_reward_scale = -0.003 * 3   * 3    * 1.5      * 2.5 * 2
    feet_air_time_reward_scale = 1.5 / 1.4   * 10    * 8     * 5
    feet_air_time_target = 0.2
    feet_air_time_max = 0.2
    undesired_contact_reward_scale = -0.75   * 2    * 2     * 5
    flat_orientation_reward_scale = -1*15  * 3
    #lower_leg_penalty_scale = -450 * 4 * 2 / 10
