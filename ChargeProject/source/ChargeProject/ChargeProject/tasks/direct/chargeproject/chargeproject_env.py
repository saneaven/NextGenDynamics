from __future__ import annotations
import math
import colorsys
from time import time

import gymnasium as gym
import torch
import torch.compiler
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .spider_robot import SPIDER_ACTUATOR_CFG

from .chargeproject_env_cfg import ChargeprojectEnvCfg

from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

import inspect
import os


class ChargeprojectEnv(DirectRLEnv):
    cfg: ChargeprojectEnvCfg

    def __init__(
        self, cfg: ChargeprojectEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self._robot.find_joints(SPIDER_ACTUATOR_CFG.joint_names_expr)
        
        # Target positions
        self._desired_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._next_desired_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._time_since_target = torch.zeros(self.num_envs, device=self.device)
        self._targets_reached = torch.zeros(self.num_envs, device=self.device)
        self._time_outs = torch.zeros(self.num_envs, device=self.device)


        self._actions = torch.zeros(
            (self.num_envs, gym.spaces.flatdim(self.single_action_space)),
            device=self.device,
        )
        self._previous_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )

        self._last_targets_reached = torch.zeros(self.num_envs, device=self.device)

        self.died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # X/Y linear velocity and yaw angular velocity commands
        # self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self.base_id, _ = self._contact_sensor.find_bodies(self.cfg.base_name)
        self.feet_ids, _ = self._contact_sensor.find_bodies(self.cfg.foot_names)
        self.undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(
            self.cfg.undesired_contact_body_names
        )
        self.body_id = self._robot.data.body_names.index(self.cfg.base_name)
        self.lower_leg_ids, _ = self._robot.find_bodies(self.cfg.lower_leg_names)

        # Change to initialize with nulls
        self._distance_buffer = torch.zeros(self.num_envs, self.cfg.distance_lookback, device=self.device)

        log_dir = self.cfg.log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.extras["log"] = dict()

        # Save the variables in ChargeprojectEnvCfg to a text file for reference
        with open(os.path.join(log_dir, "env_config.txt"), "w") as f:
            for attr, value in vars(self.cfg).items():
                f.write(f"{attr}: {value}\n")

        current_file = inspect.getfile(inspect.currentframe())
        with open(os.path.join(log_dir, "env_code.py.txt"), "w") as f:
            with open(current_file, "r") as current_f:
                f.write(current_f.read())
        

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # we add a height scanner for perceptive locomotion
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # check if cameras exist
        if not hasattr(self.cfg, "cameras"):
            self.cfg.cameras = True
            print("No camera setting found in cfg, defaulting to cameras=True")

        if self.cfg.cameras:
            self.goal_pos_visualizer = self._create_sphere_markers(
                self.cfg.marker_colors, 0.25, "/Visuals/Command/goal_position"
            )
            self.identifier_visualizer = self._create_arrow_markers(
                self.cfg.marker_colors, "/Visuals/Command/identifier_arrow"
            )

        # add ground plane
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articulation to scene
        self.scene.articulations["robot"] = self._robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        target_distance = torch.linalg.norm(self._desired_pos - self._robot.data.root_pos_w[:, :2], dim=1)

        if self.cfg.cameras:
            self._visualize_markers(target_distance)

        index = self._sim_step_counter % self.cfg.distance_lookback
        
        self._distance_buffer[:, index] = target_distance.squeeze(-1)

    def _apply_action(self) -> None:
        self.processed_actions = (
            self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos[:, self.dof_idx]
        )
        self._robot.set_joint_position_target(self.processed_actions, joint_ids=self.dof_idx)
        # self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)
    
    def _get_relative_target_info(
        self, target_pos_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relative_target_pos_w = target_pos_w - self._robot.data.root_pos_w[:, :2]
        distance = torch.linalg.norm(relative_target_pos_w, dim=1, keepdim=True)

        # Pad the 2D world vector to 3D for rotation (z=0)
        relative_target_pos_w_3d = torch.cat(
            (relative_target_pos_w, torch.zeros_like(distance)), dim=1
        )

        # Rotate the world vector into the robot's local body frame
        relative_target_pos_b_3d = math_utils.quat_apply_inverse(
            self._robot.data.root_quat_w, relative_target_pos_w_3d
        )

        # Normalize the resulting 2D body-frame vector to get the unit vector
        unit_vector_b = relative_target_pos_b_3d[:, :2] / (distance + 1e-6)

        return unit_vector_b, distance
    
    #@torch.compile(mode="reduce-overhead")
    def _get_observations_impl(self, height_scanner_data) -> dict:
        height_data = (
            height_scanner_data.pos_w[:, 2].unsqueeze(1) - height_scanner_data.ray_hits_w[..., 2] - 0.5
        ).clip(-1.0, 1.0)

        target_unit_vector, target_distance = self._get_relative_target_info(
            self._desired_pos
        )

        next_target_unit_vector, next_target_distance = self._get_relative_target_info(
            self._next_desired_pos
        )
        
        # Concatenate the selected observations into a single tensor.
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                target_unit_vector,
                target_distance,
                next_target_unit_vector,
                next_target_distance,
                self._robot.data.joint_pos[:, self.dof_idx] - self._robot.data.default_joint_pos[:, self.dof_idx],
                self._robot.data.joint_vel[:, self.dof_idx],
                height_data,
                self._actions,
            ],
            dim=-1,
        )
        return obs
    
    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        height_scanner_data = self._height_scanner.data

        observations = self._get_observations_impl(height_scanner_data)
        # Need to clone because of torch.compile
        observations = {"policy": observations.clone()}# for rl_games, "critic": observations.clone()}
        return observations

   #@torch.compile(mode="reduce-overhead")    
    def _get_rewards_impl(self, reached_target_ids) -> torch.Tensor:
        target_unit_vector, target_distance = self._get_relative_target_info(self._desired_pos)

        # - Rewards -
        # Reward for progress towards the target
        sum_valid = torch.nansum(self._distance_buffer, dim=1)
        count_valid = torch.clamp(torch.sum(~torch.isnan(self._distance_buffer), dim=1), min=1.0)
        previous_buffered_distance = sum_valid / count_valid
        difference = (previous_buffered_distance - target_distance.squeeze(-1)) * (1 + 0.5 * (count_valid-1))
        progress_reward = torch.sign(difference) * torch.pow(torch.abs(difference), self.cfg.progress_pow)
        #progress_reward = difference
        progress_reward *= (self._targets_reached * 0.5) + 1

        # torch.exp(self._targets_reached / self.cfg.progress_target_divisor)
        # Multipliy progress reward by distance to target (closer = more reward)
        # progress_reward *= (5 / torch.maximum(target_distance, torch.tensor(1, device=self.device)))
        
        # Velocity alignment the target
        velocity_alignment_reward = torch.nn.functional.cosine_similarity(
           self._robot.data.root_lin_vel_w[:, :2], target_unit_vector, dim=1
        )
        # VAR * 2 if reached target
        # *2 for positive,*1/2 for negative
        # hit_target = self._targets_reached > 0
        # multiplier = torch.minimum(1 + self._targets_reached, torch.tensor(3.5, device=self.device))
        # hit_and_positive = hit_target & (velocity_alignment_reward > 0)
        # velocity_alignment_reward[hit_and_positive] *= multiplier[hit_and_positive]
        # hit_and_negative = hit_target & (velocity_alignment_reward < 0)
        # velocity_alignment_reward[hit_and_negative] /= multiplier[hit_and_negative] * 2.0
        # print("--- vel:", self._robot.data.root_lin_vel_w[:, :2], "\nunit_vector_to_target:", unit_vector_to_target,
        #    "\nvelocity_alignment_reward", velocity_alignment_reward)

        # Bonus for get_ting to target (1 reward for first target, 1.5 for second, 2 for third, etc...)
        target_reward = torch.zeros(self.num_envs, device=self.device)
        target_reward[reached_target_ids] = (
            0.5 + self._targets_reached[reached_target_ids] * 0.5
        )

        time_penalty = torch.ones(self.num_envs, device=self.device)

        net_contact_forces = self._contact_sensor.data.net_forces_w_history

        # died if gravity is near positive (flipped over)
        self.died = self._robot.data.projected_gravity_b[:, 2] > 0.0
        death_penalty = self.died.float()

        # Reward for just moving forward in the robot's frame (sqrt)
        # forward_vel_reward = torch.sign(self._robot.data.root_lin_vel_b[:, 0]) \
        #                   * torch.pow(torch.abs(self._robot.data.root_lin_vel_b[:, 0]), 1.0/2.0)
        # forward_vel_reward = torch.sign(self._robot.data.root_lin_vel_b[:, 0]) \
        #                   * torch.square(torch.abs(self._robot.data.root_lin_vel_b[:, 0]))
        forward_vel_reward = self._robot.data.root_lin_vel_b[:, 0]
        
        # Penalty for staying still (low horizontal velocity/rotating in the robot's frame)
        horizontal_speed = torch.linalg.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
        yaw_speed = torch.abs(self._robot.data.root_ang_vel_b[:, 2])
        # Combine linear and angular speed. The 0.25 factor balances their contributions.
        motion_metric = horizontal_speed + 0.25 * yaw_speed
        still_penalty = torch.exp(-2.0 * motion_metric)


        # linear velocity tracking
        # lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        # lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        # yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        # yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(
            torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1
        )
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )
        # joint velocity
        # dof_vel = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[
            :, self.feet_ids
        ]
        last_air_time = self._contact_sensor.data.last_air_time[:, self.feet_ids]
        air_time = torch.sum((last_air_time - self.cfg.feet_air_time_target) * first_contact, dim=1)
        air_time = torch.clamp(air_time, max=self.cfg.feet_air_time_max)
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.undesired_contact_body_ids], dim=-1
                ),
                dim=1,
            )[0]
            > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1
        )

        # Get rotation matrices of lower leg links (in world frame)
        #lower_leg_rot_mats = self._robot.data.body_link_quat_w[:, self.lower_leg_ids]  # shape [envs, legs, 4] (quaternions)

        #w, x, y, z = lower_leg_rot_mats.unbind(dim=-1)

        # z-axis of lower leg in world frame
        #z_world_z = 1 - 2 * (x**2 + y**2)

        # Cosine between leg z-axis and world up ([0,0,1])
        #cos_angle = z_world_z  # since world_up = [0, 0, 1]

        # Penalize deviation from upright
        #lower_leg_penalty = torch.mean(torch.square(cos_angle), dim=1)


        return {
            # "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            # "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "progress_reward": progress_reward * self.cfg.progress_reward_scale * self.step_dt,
            "time_penalty": time_penalty * self.cfg.time_penalty_scale * self.step_dt,
            "velocity_alignment_reward": velocity_alignment_reward * self.cfg.velocity_alignment_reward_scale * self.step_dt,
            "reach_target_reward": target_reward * self.cfg.reach_target_reward_scale * self.step_dt,
            "death_penalty": death_penalty * self.cfg.death_penalty_scale * self.step_dt,
            "forward_vel": forward_vel_reward * self.cfg.forward_vel_reward_scale * self.step_dt,
            "still_penalty": still_penalty * self.cfg.still_penalty_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            # "dof_vel_l2": dof_vel * self.cfg.dof_vel_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            #"lower_leg_penalty": lower_leg_penalty * self.cfg.lower_leg_penalty_scale * self.step_dt,
        }


    def _get_rewards(self) -> torch.Tensor:

        # Check if distance is within tolerance
        target_distance = torch.linalg.norm(self._desired_pos - self._robot.data.root_pos_w[:, :2], dim=1)
        reached_target = target_distance.squeeze(-1) < self.cfg.success_tolerance
        reached_target_ids = reached_target.nonzero(as_tuple=False).squeeze(-1)

        # Generate a new target immediately for the environments that reached theirs
        if len(reached_target_ids) > 0:
            self._move_next_targets(reached_target_ids)

        rewards = self._get_rewards_impl(reached_target_ids)

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        if self.cfg.log:
            self._log_data("Episode_Reward/total_reward", torch.mean(reward))
            for key, value in rewards.items():
                episodic_sum_avg = torch.mean(value)
                self._log_data(f"Episode_Reward/{key}", episodic_sum_avg)
        
            # Add the average and max amount of targets reached to log
            self._log_data(
                "Episode_Info/targets_reached_avg",
                self._last_targets_reached.float().mean(),
            )
            self._log_data(
                "Episode_Info/targets_reached_max", self._last_targets_reached.max()
            )
            # Counts for thresholds 1 through 8 in one go
            thresholds = torch.arange(1, self.cfg.log_targets_reached_max, self.cfg.log_targets_reached_step, device=self.device)
            counts = (self._last_targets_reached.unsqueeze(-1) >= thresholds).sum(dim=0) / self.num_envs

            for _, (t, c) in enumerate(zip(thresholds, counts), start=1):
                self._log_data(f"Episode_Info/targets_reached_{t}", c)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        full_time_out = self.episode_length_buf >= self.max_episode_length - 1

        self._time_since_target += self.step_dt
        timed_out = self._time_since_target > self._time_outs
        # change it so seperate gtime_outs
        #terminate = died | timed_out

        # also died if velocity is too high
        too_fast = torch.linalg.norm(self._robot.data.root_lin_vel_w, dim=1) > self.cfg.death_velocity_threshold
        
        # Logging deaths/time outs per second
        if self.cfg.log:
            self._log_data("Episode_Termination/full_time_out", torch.count_nonzero(full_time_out) / self.num_envs)
            self._log_data("Episode_Termination/time_out", torch.count_nonzero(timed_out) / self.num_envs)
            self._log_data("Episode_Termination/died", torch.count_nonzero(self.died) / self.num_envs)
            self._log_data("Episode_Termination/too_fast", torch.count_nonzero(too_fast) / self.num_envs)

        self.terminated = self.died | too_fast
        self.truncated = timed_out | full_time_out
        return self.terminated, self.truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset actions
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Sample new commands
        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)

        # Reset
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        #default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._last_targets_reached[env_ids] = self._targets_reached[env_ids].clone()

        # Set Next target positions to be reset position
        self._next_desired_pos[env_ids] = self._robot.data.root_pos_w[env_ids, :2].clone()
        self._move_next_targets(env_ids)
        self._move_next_targets(env_ids)  # call twice to initialize both current and next target positions
        if len(env_ids) == self.num_envs:
            # For initial randomize the initial timeout
            self._time_since_target[:] = (-self.cfg.time_out_per_target + 
                torch.rand(self.num_envs, device=self.device) * self.cfg.time_out_per_target)

        self._time_outs[env_ids] = self.cfg.time_out_per_target
        self._targets_reached[env_ids] = 0

    
    def _log_data(self, key, data) -> None:
        self.extras["log"][key] = data

    def _reached_target(self, env_ids):
        # Get robot and target positions (only x, y)
        robot_pos = self._robot.data.root_pos_w[env_ids, :2]
        target_pos = self._desired_pos[env_ids]
        # Calculate distance
        dist = torch.linalg.norm(robot_pos - target_pos, dim=1)

        # Check if distance is within tolerance
        return dist < self.cfg.success_tolerance

    def _move_next_targets(self, env_ids: Sequence[int]):
        # Update current position
        self._desired_pos[env_ids] = self._next_desired_pos[env_ids].clone()

        num_resets = len(env_ids)

        radius = self.cfg.point_max_distance
        radius += (
            self.cfg.point_min_distance - self.cfg.point_max_distance
        ) * torch.rand(num_resets, device=self.device)
        angle = 2 * math.pi * torch.rand(num_resets, device=self.device)
        new_target_pos_xy = self._desired_pos[env_ids] + torch.stack(
            [radius * torch.cos(angle), radius * torch.sin(angle)], dim=1
        )

        # Update current and next desired positions
        self._next_desired_pos[env_ids] = new_target_pos_xy

        # set _distance_buffer to nan and add current distance as element
        self._distance_buffer[env_ids] = torch.nan
        index = self._sim_step_counter % self.cfg.distance_lookback
        distance = torch.linalg.norm(
            self._robot.data.root_pos_w[env_ids, :2] - self._desired_pos[env_ids], dim=1
        )
        self._distance_buffer[env_ids, index] = distance

        # Reset timer
        self._time_since_target[env_ids] = 0.0
        self._targets_reached[env_ids] += 1
        self._time_outs[env_ids] = (
            self.cfg.time_out_per_target
            - self._targets_reached[env_ids] * self.cfg.time_out_decrease_per_target
        )

    def _get_random_colors(self, num_colors: int) -> list[tuple[float, float, float]]:
        colors = []
        for i in range(num_colors):
            # Evenly space hues for maximum color distinction
            hue = (i / min(19.0, num_colors)) % 1.0
            saturation = 0.9
            value = (i % 3) / 3.0 * 0.5 + 0.5  # Vary brightness
            rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb_color)
        return colors

    def _create_sphere_markers(
        self, num_markers: int, radius: float, prim_path: str, opacity: float = 1
    ) -> VisualizationMarkers:
        colors = self._get_random_colors(num_markers)
        markers = {}
        for i, color in enumerate(colors):
            marker_key = f"sphere{i}"
            markers[marker_key] = sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color, opacity=opacity
                ),
            )
        marker_cfg = VisualizationMarkersCfg(prim_path=prim_path, markers=markers)
        return VisualizationMarkers(marker_cfg)

    def _create_arrow_markers(
        self, num_markers: int, prim_path: str
    ) -> VisualizationMarkers:
        colors = self._get_random_colors(num_markers)
        markers = {}
        for i, color in enumerate(colors):
            marker_key = f"arrow{i}"
            # Load the arrow mesh provided by Isaac Lab
            markers[marker_key] = sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        marker_cfg = VisualizationMarkersCfg(prim_path=prim_path, markers=markers)
        return VisualizationMarkers(marker_cfg)

    def _visualize_markers_impl(self, target_distance):
        desired_pos_3d = torch.cat(
            [
                self._desired_pos,
                0.2 * torch.ones((self.num_envs, 1), device=self.device),
            ],
            dim=-1,
        )
        marker_indices = (
            torch.arange(self.num_envs, device=self.device) % self.cfg.marker_colors
        )

        # --- Identifier Arrow Marker ---
        robot_pos_2d = self._robot.data.root_pos_w[:, :2]
        relative_target_pos = self._desired_pos - robot_pos_2d
        yaw = torch.atan2(relative_target_pos[:, 1], relative_target_pos[:, 0])
        orientations = math_utils.quat_from_angle_axis(yaw, self._up_dir)

        # Linearly interpolate scale from 0.15 down to 0 as the robot gets closer
        # Clamp between 0.0 and 1.0 to handle cases where the robot is farther than max_dist
        # The maximum distance should be `self.cfg.point_max_distance`
        scale_factor = torch.clamp(
            target_distance / self.cfg.point_max_distance, 0.0, 1.0
        )

        # Base scale for the arrow's width and height
        arrow_thickness_scale = 0.15 * scale_factor.unsqueeze(1) + 0.05
        # Make the arrow's length (x-axis) a bit longer for better visibility, relative to thickness
        scales = torch.cat(
            [2 * arrow_thickness_scale, arrow_thickness_scale, arrow_thickness_scale],
            dim=1,
        ).squeeze(-1)

        arrow_positions = self._robot.data.root_pos_w + torch.tensor(
            [0.0, 0.0, 0.5], device=self.device
        )

        return desired_pos_3d, marker_indices, arrow_positions, orientations, scales

    def _visualize_markers(self, target_distance):
        (
            desired_pos_3d,
            marker_indices,
            arrow_positions,
            orientations,
            scales,
        ) = self._visualize_markers_impl(target_distance)
        self.goal_pos_visualizer.visualize(
            desired_pos_3d, marker_indices=marker_indices
        )
        self.identifier_visualizer.visualize(
            translations=arrow_positions,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
        )
