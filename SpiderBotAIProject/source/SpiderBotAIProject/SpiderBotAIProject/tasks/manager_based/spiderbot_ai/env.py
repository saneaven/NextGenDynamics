# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs.common import VecEnvStepReturn

from .environment.debug_plot import DebugPlotRegistry
from .environment.map_manager import MapManager
from .environment.robot_indices import RobotIndices
from .environment.runtime_state import SpiderBotStateCache
from .environment.terrain_data import TerrainData


class SpiderBotAIEnv(ManagerBasedRLEnv):
    """SpiderBotAIProject environment (manager-based).

    Design:
        Core data providers live as env attributes. CommandTerms are only used
        for legitimate commands (waypoint, mode). Sensor-derived per-step data
        is computed in ``_compute_step_data()`` which is inserted into the
        ``step()`` loop between physics and MDP computation.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # These must exist BEFORE super().__init__() which calls load_managers().
        # load_managers() creates CommandTerms that may reference these during init.
        self.terrain_data = TerrainData(cfg, device=cfg.sim.device)
        self.spawn_pos_w = torch.zeros(cfg.scene.num_envs, 3, device=cfg.sim.device)

        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Render with proper RTX pipeline pumping for headless video recording.

        In IsaacLab 3.0, ``sim.render()`` no longer calls ``app.update()`` — that
        is now handled by visualizers (e.g. KitVisualizer).  With ``--viz none``
        there are no visualizers, so the Replicator render product never produces
        frames.  This override calls ``ensure_isaac_rtx_render_update()`` (the
        same function Camera sensors use) to pump the RTX renderer before reading
        the annotator data.
        """
        if self.render_mode == "rgb_array":
            from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

            ensure_isaac_rtx_render_update()

            # Check rendering is possible
            has_visualizers = bool(self.sim.get_setting("/isaaclab/visualizer"))
            if not (self.sim.has_gui or self.sim.has_offscreen_render or has_visualizers):
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' - no GUI and offscreen rendering not enabled."
                    " If running headless, make sure --enable_cameras is set."
                )

            # Lazily create render product + annotator
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # Position the camera via USD API (set_camera_view relies on an active
                # viewport which does not exist with --viz none).
                self._position_video_camera()

                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])

                # Warm-up: pump a few frames so the renderer initialises the render product
                import omni.kit.app

                for _ in range(3):
                    omni.kit.app.get_app().update()

            rgb_data = self._rgb_annotator.get_data()
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)

            if rgb_data.size == 0:
                return np.zeros(
                    (self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8
                )
            return rgb_data[:, :, :3]

        # For non-rgb_array modes, delegate to parent
        if not self.has_rtx_sensors and not recompute:
            self.sim.render()
        return None

    def _position_video_camera(self) -> None:
        """Set the video-recording camera transform via USD API.

        ``set_camera_view()`` from ``isaacsim.core.utils.viewports`` requires an
        active Kit viewport which does not exist with ``--viz none``.  Instead we
        write directly to the existing xformOp attributes on the camera prim.
        """
        from pxr import Gf, Usd, UsdGeom

        viewer = self.cfg.viewer
        env_index = viewer.env_index if viewer.origin_type == "env" else 0
        env_origin = self.scene.env_origins[env_index].cpu().numpy()
        eye = np.array(viewer.eye, dtype=np.float64) + env_origin
        target = np.array(viewer.lookat, dtype=np.float64) + env_origin

        stage = self.sim.stage
        prim = stage.GetPrimAtPath(viewer.cam_prim_path)
        if not prim.IsValid():
            return

        # Build camera-to-world 4x4 matrix from look-at
        eye_gf = Gf.Vec3d(*eye.tolist())
        target_gf = Gf.Vec3d(*target.tolist())
        view_matrix = Gf.Matrix4d()
        view_matrix.SetLookAt(eye_gf, target_gf, Gf.Vec3d(0, 0, 1))
        cam_to_world = view_matrix.GetInverse()

        # Replace all existing xformOps with a single transform matrix
        # Must edit on the session layer where /OmniverseKit_Persp is authored
        session_layer = stage.GetSessionLayer()
        with Usd.EditContext(stage, session_layer):
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            xformable.AddTransformOp().Set(cam_to_world)


    def load_managers(self):
        # Robot body/joint/contact indices (resolved after scene is created)
        self.robot_idx = RobotIndices.from_scene(self.scene, self.cfg)

        # Map manager (owns staleness maps, computes BEV/height/nav)
        self._map_manager = MapManager(config=self.cfg, num_envs=self.num_envs, device=self.device)

        self.state_cache = SpiderBotStateCache(self, self._map_manager)
        self.state = self.state_cache.state
        self.state_cache.refresh()

        # Debug plot registry (used by --debug_plot in play.py)
        self.debug_plot = DebugPlotRegistry()

        # Create ObservationManager, TerminationManager, RewardManager etc.
        super().load_managers()

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        is_rendering = self.sim.is_rendering

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # post-step: update env counters
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._compute_step_data()

        # check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # reset envs that terminated/timed-out
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)
        state_was_reset = len(reset_env_ids) > 0

        # Force-respawn envs where robot is trapped (unreachable targets after 10 retries)
        waypoint = self.command_manager.get_term("waypoint")
        if waypoint._force_respawn.any():
            trapped_ids = waypoint._force_respawn.nonzero(as_tuple=False).squeeze(-1)
            waypoint._force_respawn[trapped_ids] = False
            self._reset_idx(trapped_ids)
            state_was_reset = True

        if state_was_reset:
            self.state_cache.refresh()

        # update commands
        self.command_manager.compute(dt=self.step_dt)
        # step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # Per-step reward logging (mean across all envs)
        log = self.extras.get("log")
        if not isinstance(log, dict):
            log = {}
            self.extras["log"] = log
        rm = self.reward_manager
        for idx, name in enumerate(rm._term_names):
            log[f"Step_Reward/{name}"] = rm._step_reward[:, idx].mean()

        tr = waypoint.targets_reached
        log["Waypoint/targets_reached_max"] = tr.max()
        for k in range(1, 10):
            log[f"Waypoint/ratio_reached_{k}"] = (tr == k).float().mean()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    # ------------------------------------------------------------------
    # Per-step computation
    # ------------------------------------------------------------------

    def _compute_step_data(self):
        """Compute shared per-step values after physics, before MDP managers."""
        self.command_manager.get_term("mode").ensure_updated()
        self.state_cache.refresh()
        self.debug_plot.image("BEV max_height", self.state.map.bev_data[0, 0])
        self.debug_plot.image("BEV mean_height", self.state.map.bev_data[0, 1])
        self.debug_plot.image("BEV density", self.state.map.bev_data[0, 2])
        self.command_manager.get_term("waypoint").ensure_updated()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int]):
        self.state_cache.reset(env_ids)

        # Parent reset: scene.reset → spawn event (writes spawn_pos_w) → command_manager.reset
        super()._reset_idx(env_ids)
        self.state_cache.refresh()

        # Promote waypoint metrics to Episode_Info keys (without manager prefixes)
        log = self.extras.get("log")
        if not isinstance(log, dict):
            return

        waypoint_prefix = "Metrics/waypoint/"
        for key in ("Episode_Info/targets_reached_avg", "Episode_Info/targets_reached_max"):
            prefixed = waypoint_prefix + key
            if prefixed in log and key not in log:
                log[key] = log.pop(prefixed)
