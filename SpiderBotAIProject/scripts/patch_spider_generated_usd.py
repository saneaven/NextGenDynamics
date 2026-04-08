"""Patch generated SpiderBot USD files after asset export.

This script removes the invalid ``root_joint`` emitted by the current USD generation
pipeline. The joint binds ``/spider_robot`` (not a rigid body) to ``base_link``,
which causes PhysX clone warnings such as:

    Cloning joints ... /Physics/root_joint without a body rel ...

Usage:
    python scripts/patch_spider_generated_usd.py

    python scripts/patch_spider_generated_usd.py \
        --usd-root source/SpiderBotAIProject/SpiderBotAIProject/assets/spider_robot/urdf/spider_robot
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_USD_ROOT = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "SpiderBotAIProject"
    / "SpiderBotAIProject"
    / "assets"
    / "spider_robot"
    / "urdf"
    / "spider_robot"
)


def _remove_literal(text: str, needle: str) -> tuple[str, bool]:
    if needle not in text:
        return text, False
    return text.replace(needle, "", 1), True


def _remove_regex(text: str, pattern: str) -> tuple[str, bool]:
    updated, count = re.subn(pattern, "", text, count=1, flags=re.MULTILINE | re.DOTALL)
    return updated, count > 0


def _patch_root_joint_text(text: str) -> tuple[str, bool]:
    changed = False

    text, did_change = _remove_literal(text, '        </spider_robot/Physics/root_joint>,\n')
    changed |= did_change

    text, did_change = _remove_regex(
        text,
        r'\n\s*over "root_joint" \(\n'
        r'\s*prepend apiSchemas = \["IsaacJointAPI"\]\n'
        r'\s*\)\n'
        r'\s*\{\n'
        r'\s*\}\n',
    )
    changed |= did_change

    text, did_change = _remove_regex(
        text,
        r'\n\s*def PhysicsFixedJoint "root_joint"\n'
        r'\s*\{\n'
        r'\s*custom rel physics:body0\n'
        r'\s*prepend rel physics:body0 = </spider_robot>\n'
        r'\s*custom rel physics:body1\n'
        r'\s*prepend rel physics:body1 = </spider_robot/Geometry/base_link>\n'
        r'\s*point3f physics:localPos0 = \(0, 0, 0\)\n'
        r'\s*point3f physics:localPos1 = \(0, 0, 0\)\n'
        r'\s*quatf physics:localRot0 = \(1, 0, 0, 0\)\n'
        r'\s*quatf physics:localRot1 = \(1, 0, 0, 0\)\n'
        r'\s*\}\n',
    )
    changed |= did_change

    text, did_change = _remove_regex(
        text,
        r'\n\s*over "root_joint"\n'
        r'\s*\{\n'
        r'\s*\}\n',
    )
    changed |= did_change

    return text, changed


def _patch_usda(path: Path) -> bool:
    text = path.read_text()
    updated, changed = _patch_root_joint_text(text)
    if changed:
        path.write_text(updated)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch generated SpiderBot USD payloads.")
    parser.add_argument(
        "--usd-root",
        type=Path,
        default=DEFAULT_USD_ROOT,
        help="Path to the generated spider_robot USD directory.",
    )
    args = parser.parse_args()

    usd_root = args.usd_root.resolve()
    targets = sorted(usd_root.rglob("*.usda"))
    if not targets:
        raise SystemExit(f"No .usda files found under: {usd_root}")

    changed_paths: list[Path] = []
    unchanged_paths: list[Path] = []

    for path in targets:
        if _patch_usda(path):
            changed_paths.append(path)
        else:
            unchanged_paths.append(path)

    lingering = [path for path in targets if "root_joint" in path.read_text()]
    if lingering:
        lingering_text = "\n".join(f"  - {path}" for path in lingering)
        raise SystemExit(f"root_joint still present after patch:\n{lingering_text}")

    print(f"Patched SpiderBot USD root_joint issue under: {usd_root}")
    if changed_paths:
        print("Updated files:")
        for path in changed_paths:
            print(f"  - {path}")
    if unchanged_paths:
        print("Already clean:")
        for path in unchanged_paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
