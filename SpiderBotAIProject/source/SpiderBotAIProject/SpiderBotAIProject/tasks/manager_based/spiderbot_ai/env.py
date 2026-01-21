# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv


class SpiderBotAIEnv(ManagerBasedRLEnv):
    """SpiderBotAIProject environment (manager-based).

    Design:
        All stateful logic lives in CommandTerms. This class stays intentionally thin.
    """

    def _reset_idx(self, env_ids: Sequence[int]):
        # Run standard reset pipeline (managers write into self.extras["log"]).
        super()._reset_idx(env_ids)

        # Promote waypoint metrics to Episode_Info keys (without manager prefixes).
        log = self.extras.get("log")
        if not isinstance(log, dict):
            return

        waypoint_prefix = "Metrics/waypoint/"
        for key in ("Episode_Info/targets_reached_avg", "Episode_Info/targets_reached_max"):
            prefixed = waypoint_prefix + key
            if prefixed in log and key not in log:
                log[key] = log.pop(prefixed)
