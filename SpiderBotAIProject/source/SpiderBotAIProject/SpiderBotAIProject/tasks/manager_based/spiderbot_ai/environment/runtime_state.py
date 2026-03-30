from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import warp as wp

from .map_manager import MapManager, MapManagerOutput


@dataclass
class SpiderBotRobotState:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_yaw_w: torch.Tensor
    root_yaw_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_lin_vel_b: torch.Tensor
    root_ang_vel_w: torch.Tensor
    root_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    default_joint_pos: torch.Tensor
    default_joint_vel: torch.Tensor
    body_link_lin_vel_w: torch.Tensor
    body_link_ang_vel_w: torch.Tensor
    body_com_lin_acc_w: torch.Tensor
    applied_torque: torch.Tensor
    joint_acc: torch.Tensor


@dataclass
class SpiderBotSensorState:
    lidar_hits_w: torch.Tensor
    lidar_pos_w: torch.Tensor
    lidar_yaw_quat_w: torch.Tensor
    height_hits_w: torch.Tensor
    height_scanner_pos_w: torch.Tensor


@dataclass
class SpiderBotContactState:
    net_forces_w: torch.Tensor
    net_forces_w_history: torch.Tensor
    current_contact_time: torch.Tensor
    current_air_time: torch.Tensor
    last_air_time: torch.Tensor
    first_contact: torch.Tensor
    is_contact: torch.Tensor
    base_contact_time: torch.Tensor


@dataclass
class SpiderBotRuntimeState:
    env_origins: torch.Tensor
    robot: SpiderBotRobotState
    sensors: SpiderBotSensorState
    contact: SpiderBotContactState
    map: MapManagerOutput
    smoothed_vel_xy: torch.Tensor


class SpiderBotStateCache:
    def __init__(self, env, map_manager: MapManager):
        self._env = env
        self._map_manager = map_manager
        self._robot = env.scene.articulations["robot"]
        self._contact_sensor = env.scene.sensors["contact_sensor"]
        self._lidar_sensor = env.scene.sensors["lidar_sensor"]
        self._height_scanner = env.scene.sensors["height_scanner"]
        self._frame_transformer = env.scene.sensors["sensor_frame_transformer"]
        self._device = env.device

        target_frame_names = list(self._frame_transformer.data.target_frame_names)
        try:
            self._lidar_frame_index = target_frame_names.index("lidar_frame")
            self._height_scanner_frame_index = target_frame_names.index("height_scanner_frame")
        except ValueError as exc:
            raise RuntimeError(
                "sensor_frame_transformer must expose target frames ['lidar_frame', 'height_scanner_frame']."
            ) from exc

        empty = torch.empty(0, device=self._device)
        self.state = SpiderBotRuntimeState(
            env_origins=self._env.scene.env_origins,
            robot=SpiderBotRobotState(
                root_pos_w=empty.clone(),
                root_quat_w=empty.clone(),
                root_yaw_w=empty.clone(),
                root_yaw_quat_w=empty.clone(),
                root_lin_vel_w=empty.clone(),
                root_lin_vel_b=empty.clone(),
                root_ang_vel_w=empty.clone(),
                root_ang_vel_b=empty.clone(),
                projected_gravity_b=empty.clone(),
                joint_pos=empty.clone(),
                joint_vel=empty.clone(),
                default_joint_pos=empty.clone(),
                default_joint_vel=empty.clone(),
                body_link_lin_vel_w=empty.clone(),
                body_link_ang_vel_w=empty.clone(),
                body_com_lin_acc_w=empty.clone(),
                applied_torque=empty.clone(),
                joint_acc=empty.clone(),
            ),
            sensors=SpiderBotSensorState(
                lidar_hits_w=empty.clone(),
                lidar_pos_w=empty.clone(),
                lidar_yaw_quat_w=empty.clone(),
                height_hits_w=empty.clone(),
                height_scanner_pos_w=empty.clone(),
            ),
            contact=SpiderBotContactState(
                net_forces_w=empty.clone(),
                net_forces_w_history=empty.clone(),
                current_contact_time=empty.clone(),
                current_air_time=empty.clone(),
                last_air_time=empty.clone(),
                first_contact=empty.clone(),
                is_contact=empty.clone(),
                base_contact_time=empty.clone(),
            ),
            map=self._map_manager.allocate_output(),
            smoothed_vel_xy=torch.zeros(self._env.num_envs, 2, device=self._device, dtype=torch.float32),
        )

    def refresh(self) -> None:
        robot = self.state.robot
        sensors = self.state.sensors
        contact = self.state.contact

        robot.root_pos_w = self._copy_wp_tensor(robot.root_pos_w, self._robot.data.root_pos_w)
        robot.root_quat_w = self._copy_wp_tensor(robot.root_quat_w, self._robot.data.root_quat_w)
        robot.root_lin_vel_w = self._copy_wp_tensor(robot.root_lin_vel_w, self._robot.data.root_lin_vel_w)
        robot.root_lin_vel_b = self._copy_wp_tensor(robot.root_lin_vel_b, self._robot.data.root_lin_vel_b)
        robot.root_ang_vel_w = self._copy_wp_tensor(robot.root_ang_vel_w, self._robot.data.root_ang_vel_w)
        robot.root_ang_vel_b = self._copy_wp_tensor(robot.root_ang_vel_b, self._robot.data.root_ang_vel_b)
        robot.projected_gravity_b = self._copy_wp_tensor(robot.projected_gravity_b, self._robot.data.projected_gravity_b)
        robot.joint_pos = self._copy_wp_tensor(robot.joint_pos, self._robot.data.joint_pos)
        robot.joint_vel = self._copy_wp_tensor(robot.joint_vel, self._robot.data.joint_vel)
        robot.default_joint_pos = self._copy_wp_tensor(robot.default_joint_pos, self._robot.data.default_joint_pos)
        robot.default_joint_vel = self._copy_wp_tensor(robot.default_joint_vel, self._robot.data.default_joint_vel)
        robot.body_link_lin_vel_w = self._copy_wp_tensor(robot.body_link_lin_vel_w, self._robot.data.body_link_lin_vel_w)
        robot.body_link_ang_vel_w = self._copy_wp_tensor(robot.body_link_ang_vel_w, self._robot.data.body_link_ang_vel_w)
        robot.body_com_lin_acc_w = self._copy_wp_tensor(robot.body_com_lin_acc_w, self._robot.data.body_com_lin_acc_w)
        robot.applied_torque = self._copy_wp_tensor(robot.applied_torque, self._robot.data.applied_torque)
        robot.joint_acc = self._copy_wp_tensor(robot.joint_acc, self._robot.data.joint_acc)
        robot.root_yaw_w = self._yaw_from_quat_xyzw(robot.root_quat_w, robot.root_yaw_w)
        robot.root_yaw_quat_w = self._yaw_quat_xyzw(robot.root_yaw_w, robot.root_yaw_quat_w)

        frame_target_pos_w = wp.to_torch(self._frame_transformer.data.target_pos_w)
        sensors.lidar_hits_w = self._copy_wp_tensor(sensors.lidar_hits_w, self._lidar_sensor.data.ray_hits_w)
        sensors.lidar_pos_w = self._copy_wp_tensor(
            sensors.lidar_pos_w, frame_target_pos_w[:, self._lidar_frame_index, :]
        )
        sensors.lidar_yaw_quat_w = self._copy_wp_tensor(sensors.lidar_yaw_quat_w, robot.root_yaw_quat_w)
        sensors.height_hits_w = self._copy_wp_tensor(sensors.height_hits_w, self._height_scanner.data.ray_hits_w)
        sensors.height_scanner_pos_w = self._copy_wp_tensor(
            sensors.height_scanner_pos_w, frame_target_pos_w[:, self._height_scanner_frame_index, :]
        )

        contact.net_forces_w = self._copy_wp_tensor(contact.net_forces_w, self._contact_sensor.data.net_forces_w)
        contact.net_forces_w_history = self._copy_wp_tensor(
            contact.net_forces_w_history, self._contact_sensor.data.net_forces_w_history
        )
        contact.current_contact_time = self._copy_wp_tensor(
            contact.current_contact_time, self._contact_sensor.data.current_contact_time
        )
        contact.current_air_time = self._copy_wp_tensor(contact.current_air_time, self._contact_sensor.data.current_air_time)
        contact.last_air_time = self._copy_wp_tensor(contact.last_air_time, self._contact_sensor.data.last_air_time)
        contact.first_contact = self._copy_wp_tensor(
            contact.first_contact, self._contact_sensor.compute_first_contact(self._env.step_dt)
        )

        is_contact = torch.max(torch.linalg.norm(contact.net_forces_w_history, dim=-1), dim=1).values
        is_contact = is_contact > float(self._env.cfg.contact_threshold)
        contact.is_contact = self._copy_wp_tensor(contact.is_contact, is_contact.to(dtype=torch.float32))
        contact.base_contact_time = self._copy_wp_tensor(
            contact.base_contact_time,
            contact.current_contact_time[:, self._env.robot_idx.contact_sensor_base_ids].squeeze(-1),
        )

        alpha = 0.1
        vel_xy = robot.root_lin_vel_w[:, :2]
        self.state.smoothed_vel_xy.mul_(1.0 - alpha)
        self.state.smoothed_vel_xy.add_(alpha * vel_xy)

        self._map_manager.update_into(
            output=self.state.map,
            env_origins=self.state.env_origins,
            robot_pos_w=robot.root_pos_w,
            robot_yaw_w=robot.root_yaw_w,
            lidar_hits_w=sensors.lidar_hits_w,
            lidar_pos_w=sensors.lidar_pos_w,
            lidar_yaw_quat_w=sensors.lidar_yaw_quat_w,
            height_hits_w=sensors.height_hits_w,
            height_scanner_pos_w=sensors.height_scanner_pos_w,
            dt=self._env.step_dt,
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        env_ids_t = self._resolve_env_ids(env_ids)
        if env_ids_t.numel() == 0:
            return

        self._map_manager.reset(env_ids_t)
        self.state.map.nav_data[env_ids_t] = 0.0
        self.state.map.bev_data[env_ids_t] = 0.0
        self.state.map.height_data[env_ids_t] = 0.0
        self.state.map.far_staleness[env_ids_t] = 0.0
        self.state.map.exploration_bonus[env_ids_t] = 0.0
        self.state.map.staleness_peak_world[env_ids_t] = 0.0
        self.state.map.staleness_peak_value[env_ids_t] = 0.0
        self.state.contact.first_contact[env_ids_t] = 0.0
        self.state.contact.is_contact[env_ids_t] = 0.0
        self.state.contact.base_contact_time[env_ids_t] = 0.0
        self.state.smoothed_vel_xy[env_ids_t] = 0.0

    def _copy_wp_tensor(self, current: torch.Tensor, source: torch.Tensor | wp.array) -> torch.Tensor:
        source_t = source if isinstance(source, torch.Tensor) else wp.to_torch(source)
        if (
            current.numel() == 0
            or current.shape != source_t.shape
            or current.dtype != source_t.dtype
            or current.device != source_t.device
        ):
            return source_t.clone()
        current.copy_(source_t)
        return current

    def _yaw_from_quat_xyzw(self, quat_xyzw: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        x, y, z, w = quat_xyzw.unbind(-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).unsqueeze(-1)
        if out.numel() == 0 or out.shape != yaw.shape or out.dtype != yaw.dtype or out.device != yaw.device:
            return yaw
        out.copy_(yaw)
        return out

    def _yaw_quat_xyzw(self, yaw: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        half_yaw = 0.5 * yaw.squeeze(-1)
        yaw_quat = torch.zeros(yaw.shape[0], 4, device=yaw.device, dtype=yaw.dtype)
        yaw_quat[:, 2] = torch.sin(half_yaw)
        yaw_quat[:, 3] = torch.cos(half_yaw)
        if out.numel() == 0 or out.shape != yaw_quat.shape or out.dtype != yaw_quat.dtype or out.device != yaw_quat.device:
            return yaw_quat
        out.copy_(yaw_quat)
        return out

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self._env.num_envs, device=self._device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self._device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
