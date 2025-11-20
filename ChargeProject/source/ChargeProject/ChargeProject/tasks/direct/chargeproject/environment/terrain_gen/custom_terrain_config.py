from pathlib import Path
from dataclasses import dataclass

CURRENT_DIR = Path(__file__).resolve().parent


@dataclass
class CustomTerrainCfg:
    size: tuple[float, float] = (196.0, 196.0) # Terrain size in meters (x, y)
    meter_per_grid: float = 0.15 # Grid resolution in meters
    SAVE_PATH: Path = CURRENT_DIR / "terrains" / "custom_terrain.usd" # Path to save the generated heightmap
    seed: int = 42  # Random seed for heightmap generation


    roughness: float = 0.05  # Roughness factor for heightmap generation
    hill_scale: float = 1024.0  # Scale factor for hill generation
    hill_height: float = 6.0  # Maximum height of hills
    hill_noise_lacunarity: float = 2.5  # Lacunarity of noise for hills
    hill_noise_persistence: float = 0.525  # Persistence of noise for hills
    hill_noise_octaves: int = 32  # Number of octaves for hill noise

    def __post_init__(self):
        self.grid_size = (
            int(self.size[1] / self.meter_per_grid),
            int(self.size[0] / self.meter_per_grid)
        )

