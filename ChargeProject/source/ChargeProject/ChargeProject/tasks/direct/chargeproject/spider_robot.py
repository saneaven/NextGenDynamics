
import math

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

SPIDER_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[
        "joint_body_leg_hip_.*",
        "joint_leg_hip_leg_upper_.*",
        "joint_leg_upper_leg_middle_.*",
        "joint_leg_middle_leg_lower_.*",
    ],
    effort_limit={
        "joint_body_leg_hip_.*": 12.0,
        "joint_leg_hip_leg_upper_.*": 30.0,
        "joint_leg_upper_leg_middle_.*": 22.0,
        "joint_leg_middle_leg_lower_.*": 11.0,
    },
    velocity_limit={
        "joint_body_leg_hip_.*": 4.0,
        "joint_leg_hip_leg_upper_.*": 4.0,
        "joint_leg_upper_leg_middle_.*": 4.0,
        "joint_leg_middle_leg_lower_.*": 4.0,
    },
    stiffness={
        "joint_body_leg_hip_.*": 12.0,
        "joint_leg_hip_leg_upper_.*": 30.0,
        "joint_leg_upper_leg_middle_.*": 22.0,
        "joint_leg_middle_leg_lower_.*": 11.0,
    },
    damping={
        "joint_body_leg_hip_.*": 0.2,
        "joint_leg_hip_leg_upper_.*": 0.6,
        "joint_leg_upper_leg_middle_.*": 0.4,
        "joint_leg_middle_leg_lower_.*": 0.2,
    },
    armature = 0.001,
)


SPIDER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        #usd_path=f"../../../SpiderBot/spider/spider.usd", not tested go down
        usd_path=f"../SpiderBot/spider/spider.usd",
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
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.01,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),  # start above the ground
        # Default angles: body-hip=0°, hip-upper=-35°, upper-middle=70°, middle-lower=55°
        joint_pos={
            "joint_body_leg_hip_.*": math.radians(0.0),
            "joint_leg_hip_leg_upper_.*": math.radians(35.0),
            "joint_leg_upper_leg_middle_.*": math.radians(-70.0),
            "joint_leg_middle_leg_lower_.*": math.radians(-55.0),
        },
    ),
    actuators={
        "legs": SPIDER_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)