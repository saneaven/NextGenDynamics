"""Convert spider_robot URDF (via xacro) to USD for Isaac Lab.

Usage:
    # Step 1 only (xacro -> URDF) - works in any Python env with xacro installed:
        python scripts/convert_urdf.py --skip-usd

    # Full pipeline (xacro -> URDF -> USD) - requires Isaac Lab runtime:
        <ISAACLAB_PATH>/isaaclab.sh -p scripts/convert_urdf.py        # Linux / Docker
        <ISAACLAB_PATH>\\isaaclab.bat -p scripts/convert_urdf.py       # Windows

    # USD only (skip xacro, use existing .urdf):
        <ISAACLAB_PATH>/isaaclab.sh -p scripts/convert_urdf.py --skip-xacro
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ASSET_DIR = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "SpiderBotAIProject"
    / "SpiderBotAIProject"
    / "assets"
    / "spider_robot"
)

XACRO_INPUT = ASSET_DIR / "urdf" / "spider_robot.urdf.xacro"
URDF_OUTPUT = ASSET_DIR / "urdf" / "spider_robot.urdf"
USD_OUTPUT = ASSET_DIR / "urdf" / "spider_robot" / "spider_robot.usd"


def generate_urdf() -> None:
    """Run xacro to expand .xacro -> .urdf."""
    mesh_root = str(ASSET_DIR / "meshes")
    cmd = [
        "xacro",
        str(XACRO_INPUT),
        f"mesh_root:={mesh_root}",
        "-o", str(URDF_OUTPUT),
    ]
    print(f"[1/2] Generating URDF: {URDF_OUTPUT.name}")
    subprocess.check_call(cmd)
    print(f"      -> {URDF_OUTPUT}")


def convert_to_usd() -> None:
    """Convert URDF to USD using Isaac Lab's UrdfConverter."""
    # Isaac Sim / Lab imports only available inside the runtime
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    cfg = UrdfConverterCfg(
        asset_path=str(URDF_OUTPUT),
        usd_dir=str(USD_OUTPUT.parent),
        usd_file_name=USD_OUTPUT.name,
        fix_base=False,
        merge_fixed_joints=False,
        force_usd_conversion=True,
        make_instanceable=False,
    )

    print(f"[2/2] Converting URDF -> USD: {USD_OUTPUT.name}")
    converter = UrdfConverter(cfg)
    print(f"      -> {converter.usd_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert spider URDF to USD")
    parser.add_argument("--skip-xacro", action="store_true", help="Skip xacro step (use existing .urdf)")
    parser.add_argument("--skip-usd", action="store_true", help="Skip USD conversion (only generate URDF)")
    args = parser.parse_args()

    if not args.skip_xacro:
        generate_urdf()
    else:
        print("[1/2] Skipping xacro (--skip-xacro)")

    if not args.skip_usd:
        convert_to_usd()
    else:
        print("[2/2] Skipping USD conversion (--skip-usd)")

    print("Done!")


if __name__ == "__main__":
    main()
