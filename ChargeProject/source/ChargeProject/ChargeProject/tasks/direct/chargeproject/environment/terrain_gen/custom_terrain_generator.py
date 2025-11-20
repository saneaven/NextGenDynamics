import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))  # Adjust the path as needed

from .height_map_generator import generate_height_map
from .height_map_to_usd import save_height_map_to_usd
from .custom_terrain_config import CustomTerrainCfg


class CustomTerrainGenerator:
    def __init__(self, config: CustomTerrainCfg):
        self.config = config
        self.height_map = None
                
    def initialize(self):
        if self.config is None:
            raise ValueError("CustomTerrainGenerator not initialized.")

        if self.height_map is not None:
            return str(self.config.SAVE_PATH)

        print("Generating height_map...")
        self._generate_height_map()
        print("Saving height_map to USD...")
        self._save_height_map()

        print("Map generation complete.")

    def _generate_height_map(self):
        """Generates a flat height_map based on the configuration."""
        if self.config is None:
            raise ValueError("CustomTerrainGenerator not initialized. Call initialize() first.")
        
        height_map = generate_height_map(self.config)
        self.height_map = height_map

    def _save_height_map(self):
        """Saves the generated height_map to a file."""
        if self.height_map is not None:
            save_height_map_to_usd(self.height_map, str(self.config.SAVE_PATH))
        else:
            raise ValueError("height_map not generated yet. Call _generate_height_map() first.")

    def get_terrain_path(self) -> str:
        """Returns the path to the height_map file.
        Returns:
            str: Path to the generated height_map file.
        """
        return str(self.config.SAVE_PATH)


terrain_config = CustomTerrainCfg()
terrain_generator = CustomTerrainGenerator(config=terrain_config)

terrain_generator.initialize()