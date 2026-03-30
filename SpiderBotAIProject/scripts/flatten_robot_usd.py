#!/usr/bin/env python3
"""Flatten nested Geometry hierarchy in URDF-converted USDA files.

IsaacSim 6.0's URDF converter produces USDA with a nested kinematic hierarchy
(Geometry/base_link/shoulder_link/inner_link/...).  IsaacLab's contact sensor
only searches one level deep, so all rigid-body links must be direct children
of a single scope.

This script flattens the Geometry hierarchy so every link prim becomes a direct
child of the Geometry scope, then updates all prim-path references (joint body
refs, robotLinks, etc.) in every USDA file under the given root.

Usage:
    python scripts/flatten_robot_usd.py <usda_root_dir>

Example:
    python scripts/flatten_robot_usd.py \\
        source/SpiderBotAIProject/SpiderBotAIProject/assets/spider_robot/urdf/spider_robot
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# USDA block parser
# ---------------------------------------------------------------------------

@dataclass
class PrimBlock:
    """A single prim block in a USDA file."""

    header: str  # full header including metadata, e.g. 'over "name" (\n  ...\n)'
    name: str
    prop_lines: list[str] = field(default_factory=list)  # non-block content lines
    children: list[PrimBlock] = field(default_factory=list)

    def is_link(self) -> bool:
        return self.name.endswith("_link")

    def serialize(self, indent: str) -> str:
        """Serialize this block back to USDA text at the given indent level."""
        lines: list[str] = []
        # Re-indent the header
        header = _reindent_header(self.header, indent)
        lines.append(header)
        lines.append(f"{indent}{{")
        child_indent = indent + "    "
        for pl in self.prop_lines:
            lines.append(f"{child_indent}{pl.strip()}" if pl.strip() else "")
        for child in self.children:
            lines.append(child.serialize(child_indent))
        lines.append(f"{indent}}}")
        return "\n".join(lines)


def _reindent_header(header: str, indent: str) -> str:
    """Re-indent a (possibly multi-line) prim header."""
    raw_lines = header.split("\n")
    out: list[str] = []
    for i, line in enumerate(raw_lines):
        stripped = line.strip()
        if i == 0:
            out.append(f"{indent}{stripped}")
        else:
            out.append(f"{indent}    {stripped}")
    return "\n".join(out)


def _find_matching_brace(text: str, start: int) -> int:
    """Return index of the '}' that closes the '{' at *start*."""
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        # skip string literals
        elif ch == '"':
            i += 1
            while i < len(text) and text[i] != '"':
                if text[i] == "\\":
                    i += 1
                i += 1
        i += 1
    raise ValueError(f"Unmatched '{{' at position {start}")


# Regex that matches the start of a prim declaration line.
_PRIM_RE = re.compile(
    r'^[ \t]*((?:def\s+\w+|over)\s+"([^"]+)")',
    re.MULTILINE,
)


def parse_block_content(text: str) -> tuple[list[str], list[PrimBlock]]:
    """Parse the text *between* a pair of braces into property lines + child blocks."""
    prop_lines: list[str] = []
    children: list[PrimBlock] = []
    pos = 0
    while pos < len(text):
        m = _PRIM_RE.search(text, pos)
        if m is None:
            # rest is all property lines
            prop_lines.extend(text[pos:].split("\n"))
            break

        # everything before this match → property lines
        before = text[pos : m.start()]
        prop_lines.extend(before.split("\n"))

        # collect the full header (may span multiple lines when metadata exists)
        header_start = m.start()
        header_end = m.end()
        # check for parenthesised metadata block
        rest = text[header_end:].lstrip(" \t")
        if rest.startswith("("):
            paren_depth = 0
            j = text.index("(", header_end)
            while j < len(text):
                if text[j] == "(":
                    paren_depth += 1
                elif text[j] == ")":
                    paren_depth -= 1
                    if paren_depth == 0:
                        header_end = j + 1
                        break
                j += 1

        header_text = text[header_start:header_end].strip()

        # find the opening brace
        brace_open = text.index("{", header_end)
        brace_close = _find_matching_brace(text, brace_open)

        inner = text[brace_open + 1 : brace_close]
        child_props, grandchildren = parse_block_content(inner)

        block = PrimBlock(
            header=header_text,
            name=m.group(2),
            prop_lines=child_props,
            children=grandchildren,
        )
        children.append(block)
        pos = brace_close + 1

    # strip empty leading/trailing prop lines
    while prop_lines and not prop_lines[0].strip():
        prop_lines.pop(0)
    while prop_lines and not prop_lines[-1].strip():
        prop_lines.pop()
    return prop_lines, children


# ---------------------------------------------------------------------------
# Geometry flattener
# ---------------------------------------------------------------------------

def collect_links_flat(block: PrimBlock) -> list[PrimBlock]:
    """Recursively collect all link children, removing them from their parents.

    Non-link children (visuals, collisions) stay attached to their parent link.
    Returns the collected link blocks in traversal order.
    """
    collected: list[PrimBlock] = []
    remaining_children: list[PrimBlock] = []
    for child in block.children:
        if child.is_link():
            # recursively collect from this child first
            collected.extend(collect_links_flat(child))
            # then add the child itself (now stripped of its own link children)
            collected.append(child)
        else:
            remaining_children.append(child)
    block.children = remaining_children
    return collected


def flatten_geometry_section(text: str) -> str:
    """Find the Geometry scope inside 'spider_robot' and flatten its link hierarchy."""
    # Locate 'over "Geometry"' or 'def Scope "Geometry"'
    geo_re = re.compile(r'^([ \t]*)((?:def\s+\w+|over)\s+"Geometry")', re.MULTILINE)
    geo_match = geo_re.search(text)
    if geo_match is None:
        return text  # no Geometry scope in this file

    # find the opening brace
    brace_open = text.index("{", geo_match.end())
    brace_close = _find_matching_brace(text, brace_open)
    geo_inner = text[brace_open + 1 : brace_close]

    # parse children of Geometry
    _props, geo_children = parse_block_content(geo_inner)

    # find base_link among the children
    base_link = None
    for child in geo_children:
        if child.name == "base_link":
            base_link = child
            break

    if base_link is None:
        return text  # no base_link found

    # flatten: extract all nested link prims from base_link
    extracted = collect_links_flat(base_link)

    # rebuild the Geometry block content
    geo_indent = geo_match.group(1)
    child_indent = geo_indent + "    "

    new_geo_parts: list[str] = []
    # base_link first (now only contains non-link children)
    new_geo_parts.append(base_link.serialize(child_indent))
    # then all extracted links
    for link_block in extracted:
        new_geo_parts.append(link_block.serialize(child_indent))
    # other non-base_link top-level children (unlikely but keep them)
    for child in geo_children:
        if child.name != "base_link":
            new_geo_parts.append(child.serialize(child_indent))

    new_geo_inner = "\n".join(new_geo_parts)

    # rebuild: header + { + inner + }
    geo_header = text[geo_match.start() : brace_open].rstrip()
    replacement = f"{geo_header}\n{geo_indent}{{\n{new_geo_inner}\n{geo_indent}}}"
    return text[: geo_match.start()] + replacement + text[brace_close + 1 :]


# ---------------------------------------------------------------------------
# Path reference updater
# ---------------------------------------------------------------------------

def build_path_map(base_usda_path: Path) -> dict[str, str]:
    """Parse base.usda to discover the nested link hierarchy and build old→new path map."""
    text = base_usda_path.read_text()

    # find Geometry block
    geo_re = re.compile(r'(def\s+\w+|over)\s+"Geometry"', re.MULTILINE)
    geo_match = geo_re.search(text)
    if not geo_match:
        print("ERROR: Could not find Geometry scope in base.usda", file=sys.stderr)
        sys.exit(1)

    brace_open = text.index("{", geo_match.end())
    brace_close = _find_matching_brace(text, brace_open)
    geo_inner = text[brace_open + 1 : brace_close]
    _, geo_children = parse_block_content(geo_inner)

    # find base_link
    base_link = None
    for child in geo_children:
        if child.name == "base_link":
            base_link = child
            break
    if not base_link:
        print("ERROR: Could not find base_link in Geometry", file=sys.stderr)
        sys.exit(1)

    # BFS to collect all nested link paths
    path_map: dict[str, str] = {}  # old_suffix → new_suffix

    def walk(block: PrimBlock, current_path: str):
        for child in block.children:
            if child.is_link():
                old_path = f"{current_path}/{child.name}"
                new_path = f"Geometry/{child.name}"
                path_map[old_path] = new_path
                walk(child, old_path)

    walk(base_link, "Geometry/base_link")
    return path_map


def replace_paths(text: str, path_map: dict[str, str], root_prim: str) -> str:
    """Replace all prim path references in text using the path map.

    Replaces longest paths first to avoid partial matches.
    """
    # Sort by length descending so longer (deeper) paths get replaced first
    sorted_entries = sorted(path_map.items(), key=lambda kv: len(kv[0]), reverse=True)

    for old_suffix, new_suffix in sorted_entries:
        old_ref = f"{root_prim}/{old_suffix}"
        new_ref = f"{root_prim}/{new_suffix}"
        text = text.replace(old_ref, new_ref)

    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_usda_files(root: Path) -> list[Path]:
    """Find all .usda files under root that need processing."""
    targets: list[Path] = []
    for p in sorted(root.rglob("*.usda")):
        # skip instances/materials that don't contain Geometry
        if "instances" in p.name or "materials" in p.name or "geometries" in p.name:
            continue
        targets.append(p)
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("usda_root", type=Path, help="Root directory of the USDA package (contains spider_robot.usda)")
    args = parser.parse_args()

    root: Path = args.usda_root
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    # find the base.usda to build path map
    base_usda = root / "payloads" / "base.usda"
    if not base_usda.exists():
        print(f"ERROR: {base_usda} not found", file=sys.stderr)
        sys.exit(1)

    # detect the root prim name from the top-level USDA
    top_usda = next(root.glob("*.usda"), None)
    if top_usda is None:
        print(f"ERROR: No .usda file found in {root}", file=sys.stderr)
        sys.exit(1)
    root_prim_match = re.search(r'defaultPrim\s*=\s*"(\w+)"', top_usda.read_text())
    root_prim = root_prim_match.group(1) if root_prim_match else "spider_robot"

    print(f"Root prim: {root_prim}")

    # Step 1: Build path map from base.usda
    path_map = build_path_map(base_usda)
    print(f"Found {len(path_map)} nested link paths to flatten:")
    for old, new in sorted(path_map.items()):
        print(f"  {old} -> {new}")

    # Step 2: Process each USDA file
    usda_files = find_usda_files(root)
    print(f"\nProcessing {len(usda_files)} USDA files:")

    for usda_path in usda_files:
        rel = usda_path.relative_to(root)
        text = usda_path.read_text()
        original = text

        # Replace path references (in <> angle brackets and bare paths)
        text = replace_paths(text, path_map, f"</{root_prim}")
        text = replace_paths(text, path_map, f"<{root_prim}")
        text = replace_paths(text, path_map, f"/{root_prim}")

        # Flatten the Geometry hierarchy
        text = flatten_geometry_section(text)

        if text != original:
            usda_path.write_text(text)
            print(f"  [MODIFIED] {rel}")
        else:
            print(f"  [unchanged] {rel}")

    print("\nDone. Remember to also update env_cfg.py prim_paths:")
    print('  contact_sensor:  "{ENV_REGEX_NS}/Robot/.*" -> "{ENV_REGEX_NS}/Robot/Geometry/.*"')
    print('  other sensors:   "{ENV_REGEX_NS}/Robot/base_link" -> "{ENV_REGEX_NS}/Robot/Geometry/base_link"')


if __name__ == "__main__":
    main()
