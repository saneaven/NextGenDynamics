# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .nested_contact_sensor import SpiderBotNestedContactSensor


@configclass
class SpiderBotNestedContactSensorCfg(ContactSensorCfg):
    """Project-local PhysX contact sensor config for nested robot body hierarchies."""

    class_type: type[SpiderBotNestedContactSensor] = SpiderBotNestedContactSensor
