# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

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
