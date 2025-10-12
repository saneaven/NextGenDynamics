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
import isaaclab.terrains as terrain_gen

from ChargeProject.tasks.direct.chargeproject.double_noise_env import HfTwoScaleNoiseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


SMOOTH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg = terrain_gen.TerrainGeneratorCfg(
    size=(64.0, 64.0),
    # border_width=10.0,
    num_rows=1,
    num_cols=1,
    color_scheme="random",
    vertical_scale=0.000003,
    horizontal_scale=0.05,
    sub_terrains={
        "random_uniform": HfTwoScaleNoiseCfg(
            proportion=1.0,
            macro_noise_step=0.005,
            macro_noise_range=(-0.3, 0.3),
            macro_downsampled_scale=1.6,
            micro_noise_step=0.000003,
            micro_noise_range=(-0.2, 0.2),
            micro_downsampled_scale=0.05,
            flat_patch_sampling={
                "robot_spawn": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=2048, patch_radius=1.0, max_height_diff=0.5
                ),
            },
            size=(128., 128.)
        ),
        # "plane": terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.2,
        #     size=(2,2),
        #     flat_patch_sampling={
        #         "robot_spawn": terrain_gen.FlatPatchSamplingCfg(
        #             num_patches=100, patch_radius=1.0, max_height_diff=0.5
        #         ),
        #     },
        # ),
    },
)

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
    observation_space = 691 # with height scanner and lidar
    state_space = 0
    # simulation
    decimation = 2
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation,
        physx=PhysxCfg(
            #gpu_collision_stack_size = 2**27,
            #gpu_max_rigid_patch_count = 2**19
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, 
        env_spacing=4.0, 
        replicate_physics=True
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="generator",
        terrain_generator=SMOOTH_TERRAIN_CFG,
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
    success_tolerance = 1.0 #1  # meters
    time_out_per_target = 30.0  # seconds
    time_out_decrease_per_target = 0.075  # seconds

    log_targets_reached_max = 10
    log_targets_reached_step = 1

    marker_colors = 57

    # Final rewards
    action_scale = 0.15# 0.2
    
    progress_reward_scale = 50  * 5 * 5*4# linear version ish
    #progress_reward_scale = 50  * 5 * 5 # 1.5 version
    progress_pow = 1.3
    distance_lookback = 8
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
