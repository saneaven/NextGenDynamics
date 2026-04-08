# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import safe_set_attribute_on_usd_prim
from isaaclab.sensors.contact_sensor.base_contact_sensor import BaseContactSensor
from isaaclab_physx.physics import PhysxManager as SimulationManager
from isaaclab_physx.sensors.contact_sensor.contact_sensor import ContactSensor as PhysXContactSensor
from pxr import Usd, UsdPhysics

if TYPE_CHECKING:
    from .nested_contact_sensor_cfg import SpiderBotNestedContactSensorCfg


class SpiderBotNestedContactSensor(PhysXContactSensor):
    """PhysX contact sensor that supports nested rigid bodies under a robot root prim."""

    cfg: "SpiderBotNestedContactSensorCfg"

    def __init__(self, cfg: "SpiderBotNestedContactSensorCfg"):
        super().__init__(cfg)
        self._ordered_body_names: list[str] = []
        self._ordered_body_path_exprs: list[str] = []
        # Prepare the template prototype before cloning so replicated environments inherit the schema.
        self._ensure_contact_reporters_on_stage(self._find_template_robot_roots())

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies with contact sensors attached."""
        return list(self._ordered_body_names)

    def _initialize_impl(self):
        # Only initialize the generic sensor bookkeeping from the base classes.
        BaseContactSensor._initialize_impl(self)
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        robot_roots = self._find_runtime_robot_roots()
        if not robot_roots:
            raise RuntimeError(f"Sensor at path '{self.cfg.prim_path}' could not find any robot roots.")

        template_robot_root = robot_roots[0]
        body_prims = self._resolve_ordered_body_prims(template_robot_root)
        if not body_prims:
            raise RuntimeError(
                f"Sensor at path '{self.cfg.prim_path}' could not find any rigid bodies under "
                f"'{template_robot_root.GetPath().pathString}'."
            )

        template_robot_path = template_robot_root.GetPath().pathString
        self._ordered_body_names = [prim.GetName() for prim in body_prims]
        self._ordered_body_path_exprs = [
            self.cfg.prim_path.replace(".*", "*") + prim.GetPath().pathString[len(template_robot_path) :]
            for prim in body_prims
        ]

        filter_prim_paths_glob = [expr.replace(".*", "*") for expr in self.cfg.filter_prim_paths_expr]
        self._body_physx_view = self._create_rigid_body_view(self._ordered_body_path_exprs)
        self._contact_view = self._create_rigid_contact_view(self._ordered_body_path_exprs, filter_prim_paths_glob)

        self._num_sensors = self.body_physx_view.count // self._num_envs
        if self._num_sensors != len(self._ordered_body_names):
            raise RuntimeError(
                "Failed to initialize nested contact reporter for all robot bodies."
                f"\n\tInput prim path    : {self.cfg.prim_path}"
                f"\n\tResolved body names: {self._ordered_body_names}"
                f"\n\tResolved body exprs: {self._ordered_body_path_exprs}"
            )

        if self.cfg.track_contact_points or self.cfg.track_friction_forces:
            if not self.cfg.filter_prim_paths_expr:
                raise ValueError(
                    "The 'filter_prim_paths_expr' is empty. Please specify a valid filter pattern to track"
                    f" {'contact points' if self.cfg.track_contact_points else 'friction forces'}."
                )
            if self.cfg.max_contact_data_count_per_prim < 1:
                raise ValueError(
                    f"The 'max_contact_data_count_per_prim' is {self.cfg.max_contact_data_count_per_prim}. "
                    "Please set it to a value greater than 0 to track"
                    f" {'contact points' if self.cfg.track_contact_points else 'friction forces'}."
                )

        self._create_buffers()

    def _find_runtime_robot_roots(self) -> list[Usd.Prim]:
        return sim_utils.find_matching_prims(self.cfg.prim_path, stage=self.stage)

    def _find_template_robot_roots(self) -> list[Usd.Prim]:
        # InteractiveScene spawns cloned assets under /World/template/<Asset>/proto_asset_* before replication.
        template_root_path = re.sub(r"/envs/env_\.\*", "/template", self.cfg.prim_path)
        template_root = self.stage.GetPrimAtPath(template_root_path)
        if not template_root.IsValid():
            return []

        prototype_roots: list[Usd.Prim] = []
        frontier = [template_root]
        while frontier:
            prim = frontier.pop(0)
            if prim.GetName().startswith("proto_asset_"):
                prototype_roots.append(prim)
                continue
            frontier.extend(prim.GetChildren())
        return prototype_roots

    def _ensure_contact_reporters_on_stage(self, robot_roots: list[Usd.Prim]) -> None:
        for robot_root in robot_roots:
            self.stage.Load(robot_root.GetPath())
            body_prims = self._resolve_ordered_body_prims(robot_root)
            with Usd.EditContext(self.stage, self.stage.GetRootLayer()):
                for rigid_body_prim in body_prims:
                    self._apply_contact_reporter_schema(rigid_body_prim)

    def _resolve_ordered_body_prims(self, robot_root: Usd.Prim) -> list[Usd.Prim]:
        relation = robot_root.GetRelationship("isaac:physics:robotLinks")
        if relation and relation.IsValid():
            ordered_prims: list[Usd.Prim] = []
            for target in relation.GetTargets():
                prim = self.stage.GetPrimAtPath(target)
                if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    ordered_prims.append(prim)
            if ordered_prims:
                return ordered_prims
        return self._collect_rigid_body_prims(robot_root)

    def _collect_rigid_body_prims(self, root_prim: Usd.Prim) -> list[Usd.Prim]:
        rigid_body_prims: list[Usd.Prim] = []

        def visit(prim: Usd.Prim) -> None:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_prims.append(prim)
            for child in prim.GetChildren():
                visit(child)

        visit(root_prim)
        return rigid_body_prims

    def _apply_contact_reporter_schema(self, prim: Usd.Prim) -> None:
        if prim.IsInstanceable():
            prim.SetInstanceable(False)

        applied_schemas = prim.GetAppliedSchemas()
        if "PhysxRigidBodyAPI" not in applied_schemas:
            prim.AddAppliedSchema("PhysxRigidBodyAPI")
        safe_set_attribute_on_usd_prim(prim, "physxRigidBody:sleepThreshold", 0.0, camel_case=False)

        if "PhysxContactReportAPI" not in applied_schemas:
            prim.AddAppliedSchema("PhysxContactReportAPI")
        safe_set_attribute_on_usd_prim(prim, "physxContactReport:threshold", self.cfg.force_threshold, camel_case=False)

    def _create_rigid_body_view(self, body_path_exprs: list[str]):
        try:
            return self._physics_sim_view.create_rigid_body_view(body_path_exprs)
        except Exception as exc:
            raise RuntimeError(
                "Failed to create rigid body view for nested SpiderBot contact sensor."
                f"\n\tBody expressions: {body_path_exprs}"
            ) from exc

    def _create_rigid_contact_view(self, body_path_exprs: list[str], filter_prim_paths_glob: list[str]):
        try:
            return self._physics_sim_view.create_rigid_contact_view(
                body_path_exprs,
                filter_patterns=filter_prim_paths_glob,
                max_contact_data_count=self.cfg.max_contact_data_count_per_prim * len(body_path_exprs) * self._num_envs,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to create rigid contact view for nested SpiderBot contact sensor."
                f"\n\tBody expressions: {body_path_exprs}"
                f"\n\tFilter patterns : {filter_prim_paths_glob}"
            ) from exc
