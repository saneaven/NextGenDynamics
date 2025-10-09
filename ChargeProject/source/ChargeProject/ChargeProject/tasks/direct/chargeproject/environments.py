from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

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


SMOOTH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg = terrain_gen.TerrainGeneratorCfg(
    size=(20.0, 20.0),
    # border_width=10.0,
    num_rows=1,
    num_cols=1,
    # horizontal_scale=0.1,
    # vertical_scale=0.005,
    # slope_threshold=0.5,
    # use_cache=False,
    color_scheme="random",
    sub_terrains={
        "random_uniform": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(-0.05, 0.05),
            noise_step= 0.005,
            downsampled_scale=0.3,
            flat_patch_sampling={
                "robot_spawn": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=10, patch_radius=1.0, max_height_diff=0.5
                ),
            },
            size=(100., 100.)
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

ROBOT_CFG = UNITREE_GO2_CFG.replace(  # type: ignore
    prim_path="/World/envs/env_.*/Robot",
    init_state=UNITREE_GO2_CFG.init_state.replace(  # type: ignore
        pos=(0.0, 0.0, 0.5),
    ),
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    num_envs = 786
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
