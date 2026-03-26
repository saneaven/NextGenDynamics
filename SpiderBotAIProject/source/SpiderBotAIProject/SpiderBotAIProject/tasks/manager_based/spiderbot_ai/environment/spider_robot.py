# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from ..paths import SPIDER_USD_PATH


SPIDER_JOINT_INFO = {
    "default_pos": {
        "leg_.*_shoulder_yaw_joint": math.radians(0.0),
        "leg_.*_inner_pitch_joint": math.radians(-45.0),
        "leg_.*_middle_pitch_joint": math.radians(90.0),
        "leg_.*_outer_pitch_joint": math.radians(45.0),
    },
    "limit_min": {
        "leg_.*_shoulder_yaw_joint": math.radians(-90.0),
        "leg_.*_inner_pitch_joint": math.radians(-150.0),
        "leg_.*_middle_pitch_joint": math.radians(-170.0),
        "leg_.*_outer_pitch_joint": math.radians(-170.0),
    },
    "limit_max": {
        "leg_.*_shoulder_yaw_joint": math.radians(90.0),
        "leg_.*_inner_pitch_joint": math.radians(150.0),
        "leg_.*_middle_pitch_joint": math.radians(170.0),
        "leg_.*_outer_pitch_joint": math.radians(170.0),
    },
}

VEL_LIMIT = 6.0


SPIDER_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[
        "leg_.*_shoulder_yaw_joint",
        "leg_.*_inner_pitch_joint",
        "leg_.*_middle_pitch_joint",
        "leg_.*_outer_pitch_joint",
    ],
    effort_limit_sim={
        "leg_.*_shoulder_yaw_joint": 40.0,
        "leg_.*_inner_pitch_joint": 40.0,
        "leg_.*_middle_pitch_joint": 80.0,
        "leg_.*_outer_pitch_joint": 60.0,
    },
    stiffness={
        "leg_.*_shoulder_yaw_joint": 80.0,
        "leg_.*_inner_pitch_joint": 70.0,
        "leg_.*_middle_pitch_joint": 45.0,
        "leg_.*_outer_pitch_joint": 20.0,
    },
    velocity_limit_sim={
        "leg_.*_shoulder_yaw_joint": VEL_LIMIT,
        "leg_.*_inner_pitch_joint": VEL_LIMIT,
        "leg_.*_middle_pitch_joint": 0.75 * VEL_LIMIT,
        "leg_.*_outer_pitch_joint": 0.5 * VEL_LIMIT,
    },
    damping={
        "leg_.*_shoulder_yaw_joint": 1.5,
        "leg_.*_inner_pitch_joint": 1.5,
        "leg_.*_middle_pitch_joint": 1.0,
        "leg_.*_outer_pitch_joint": 1.0,
    },
    friction=0.05,
    armature=0.005,
)


SPIDER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SPIDER_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.01,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos=SPIDER_JOINT_INFO["default_pos"],
    ),
    actuators={"legs": SPIDER_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)

