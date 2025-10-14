from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from .spider_robot import SPIDER_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.assets import AssetBaseCfg
from ChargeProject.tasks.direct.chargeproject.double_noise_env import HfTwoScaleNoiseCfg

from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

base_name = "body"

ROUGH_TERRAIN_CFG: terrain_gen.TerrainGeneratorCfg = terrain_gen.TerrainGeneratorCfg(
    size=(4.0, 4.0),
    border_width=2.0,
    num_rows=20,
    num_cols=20,
    # horizontal_scale=0.1,
    # vertical_scale=0.005,
    # slope_threshold=0.5,
    # use_cache=False,
    color_scheme="height",
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.02, 0.2),
            step_width=1.3,
            platform_width=1.1,
            border_width=1.2,
            flat_patch_sampling={
                "robot_spawn": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=10, patch_radius=1.0, max_height_diff=0.5
                ),
            },
        ),
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.1,
            flat_patch_sampling={
                "robot_spawn": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=10, patch_radius=1.0, max_height_diff=0.5
                ),
            },
        ),
        "pyramid_reverse": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.02, 0.2),
            step_width=0.3,
            platform_width=1.0,
            border_width=0.5,
        ),
        "random_uniform": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(-0.1, 0.1),
            noise_step=1.0,
            horizontal_scale=20.0,
            vertical_scale=0.2,
            size=(20.0, 20.0),
            flat_patch_sampling={
                "robot_spawn": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=10, patch_radius=1.0, max_height_diff=0.5
                ),
            },
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            num_obstacles=100,
            obstacle_width_range=(0.05, 5.0),
            obstacle_height_range=(0.05, 0.2),
            size=(20.0, 20.0)
        )
    },
)


SMOOTH_TERRAIN_CFG: terrain_gen.TerrainGeneratorCfg = terrain_gen.TerrainGeneratorCfg(
    size=(128.8, 128.0),
    # border_width=10.0,
    num_rows=1,
    num_cols=1,
    color_scheme="random",
    vertical_scale=0.000005,
    horizontal_scale=0.1,
    sub_terrains={
        "random_uniform": HfTwoScaleNoiseCfg(
            proportion=1.0,
            macro_noise_step=0.0005,
            macro_noise_range=(-0.25, 0.25),
            macro_downsampled_scale=0.8,
            micro_noise_step=0.000005,
            micro_noise_range=(-0.05, 0.05),
            micro_downsampled_scale=0.1,
            flat_patch_sampling={
                "robot_spawn": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=1024, patch_radius=1.2, max_height_diff=0.15
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
class MySceneCfg(InteractiveSceneCfg):
    num_envs = 1024
    env_spacing = 4.0
    replicate_physics = True

    terrain: terrain_gen.TerrainImporterCfg = terrain_gen.TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="generator",
        terrain_generator=SMOOTH_TERRAIN_CFG,
        # max_init_terrain_level=1,
        # collision_group=-1,
        # physics_material=sim_utils.RigidBodyMaterialCfg(
        #     friction_combine_mode="multiply",
        #     restitution_combine_mode="multiply",
        #     static_friction=1.0,
        #     dynamic_friction=1.0,
        #     restitution=0.0,
        # ),
    )

    # robot(s)
    robot: ArticulationCfg = SPIDER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    #robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    #robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    
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

    dome = AssetBaseCfg(
        prim_path="/World/Lights/Dome",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.8, 0.8, 0.8),
        ),
    )
