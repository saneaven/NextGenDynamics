# Combine two different scale noises to create a terrain with both large hills and fine surface roughness.
from isaaclab.utils import configclass
import numpy as np
from isaaclab.terrains.height_field import hf_terrains, hf_terrains_cfg
from isaaclab.terrains.height_field.hf_terrains import height_field_to_mesh


@height_field_to_mesh
def two_scale_noise_terrain(difficulty: float, cfg: "HfTwoScaleNoiseCfg") -> np.ndarray:
    base = cfg 

    def make_ru(noise_range, noise_step, down_s):

        sub = hf_terrains_cfg.HfRandomUniformTerrainCfg(
            size=base.size,
            horizontal_scale=base.horizontal_scale,
            vertical_scale=base.vertical_scale,
            downsampled_scale=down_s if down_s is not None else base.horizontal_scale,
            noise_range=noise_range,
            noise_step=noise_step,
            border_width=base.border_width,
            slope_threshold=base.slope_threshold,
        )
        
        return hf_terrains.random_uniform_terrain.__wrapped__(0.0, sub).astype(np.int32)
    
    def make_wave(amplitude_range):
        sub = hf_terrains_cfg.HfWaveTerrainCfg(
            size=base.size,
            horizontal_scale=base.horizontal_scale,
            vertical_scale=base.vertical_scale,
            amplitude_range=amplitude_range,
            num_waves=int(base.size[0]/6.0),
            border_width=base.border_width,
            slope_threshold=base.slope_threshold,
        )

        return hf_terrains.wave_terrain.__wrapped__(0.0, sub).astype(np.int32)

    # Generate two different scale noises and combine them
    # z_macro = make_ru(cfg.macro_noise_range, cfg.macro_noise_step, cfg.macro_downsampled_scale) # large hills
    z_macro = make_wave(cfg.macro_noise_range) # large hills
    
    z_micro = make_ru(cfg.micro_noise_range, cfg.micro_noise_step, cfg.micro_downsampled_scale) # fine surface roughness
    z = z_macro + z_micro
    return np.clip(z, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)

@configclass
class HfTwoScaleNoiseCfg(hf_terrains_cfg.HfTerrainBaseCfg):
    function = two_scale_noise_terrain
    # roughness of the surface (little noise)
    micro_noise_range: tuple[float, float] = (-0.02, 0.02)
    micro_noise_step: float = 0.000002
    micro_downsampled_scale: float | None = 0.05
    # hills (large noise)
    macro_noise_range: tuple[float, float] = (-0.3, 0.3)
    macro_noise_step: float = 0.01
    macro_downsampled_scale: float | None = 0.8

