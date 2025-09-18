# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import math
import colorsys

import gymnasium as gym
import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .chargeproject_env_cfg import ChargeprojectEnvCfg

from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

import csv
import os

#import multiprocessing as mp
from multiprocessing.dummy import Process
import threading
import time
import matplotlib.pyplot as plt
import pandas as pd



def live_plotter(csv_filepath: str, stop_event: threading.Event):
    try:
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 8))

        # --- State variables ---
        lines = {}
        last_data_length = 0
        MA_WINDOW = 20  # Set the moving average window

        # Get a color map to automatically assign different colors to each line
        colormap = plt.get_cmap('tab20')
        colors = [colormap(i) for i in np.linspace(0, 1, 20)]

        # --- Plot setup ---
        ax.set_title("Live Reward Components (20-iteration Moving Average)")
        ax.set_xlabel("Episode Batch")
        ax.set_ylabel("Smoothed Reward Value")
        ax.grid(True)

        # Loop until the main thread signals to stop
        while not stop_event.is_set():
            try:
                # Read the latest data from the CSV
                df = pd.read_csv(csv_filepath)
                if df.empty or len(df) <= last_data_length:
                    # No new data, wait and continue
                    time.sleep(2)
                    continue

                last_data_length = len(df)

                # --- First-time setup for lines and legend ---
                if not lines:
                    color_index = 0
                    # Use the CSV columns to create plot lines
                    for col_name in df.columns:
                        if "Episode_Reward" in col_name:
                            # Clean up the name for the legend
                            label_name = col_name.replace("Episode_Reward/", "")
                            lines[col_name] = ax.plot([], [], label=label_name, color=colors[color_index % len(colors)])[0]
                            color_index += 1
                    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
                    fig.tight_layout(rect=[0, 0, 0.85, 1])

                # --- Update plot with all current data ---
                for col_name, line in lines.items():
                    # Calculate moving average for the column
                    moving_avg = df[col_name].rolling(window=MA_WINDOW, min_periods=1).mean()
                    line.set_data(df.index, moving_avg)

                ax.relim()
                ax.autoscale_view(True, True, True)
                fig.canvas.draw()
                fig.canvas.flush_events()

            except (FileNotFoundError, pd.errors.EmptyDataError):
                # Handle cases where the file doesn't exist yet or is empty
                print("Waiting for log file to be created...")
            except Exception as e:
                print(f"An error occurred in the plotter thread: {e}")

            # Wait before polling the file again
            time.sleep(2)

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        plt.ioff()
        plt.close(fig)
        print("Plotter thread has shut down.")





class ChargeprojectEnv(DirectRLEnv):
    cfg: ChargeprojectEnvCfg

    def __init__(self, cfg: ChargeprojectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Target positions
        self._desired_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._next_desired_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._time_since_target = torch.zeros(self.num_envs, device=self.device)
        self._targets_reached = torch.zeros(self.num_envs, device=self.device)

        self.actions = torch.zeros((self.num_envs, gym.spaces.flatdim(self.single_action_space)), device=self.device)
        self.previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        
        
        self.base_id, _ = self._contact_sensor.find_bodies(self.cfg.base_name)
        self.feet_ids, _ = self._contact_sensor.find_bodies(self.cfg.foot_names)
        self.undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(self.cfg.undesired_contact_body_names)
        self.body_id = self._robot.data.body_names.index(self.cfg.base_name)

        self._previous_dist_to_target = torch.zeros(self.num_envs, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "total_reward",
                "progress_reward",
                "velocity_alignment_reward",
                "reach_target_reward",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "dof_vel_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }
        
        log_dir = self.cfg.log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._log_file_path = os.path.join(log_dir, "training_log.csv")
        
        # Define the headers for your CSV file based on the keys in `extras["log"]`
        self._log_fieldnames = [
            "Episode_Reward/total_reward",
            "Episode_Reward/progress_reward",
            "Episode_Reward/velocity_alignment_reward",
            "Episode_Reward/reach_target_reward",
            "Episode_Reward/lin_vel_z_l2",
            "Episode_Reward/ang_vel_xy_l2",
            "Episode_Reward/dof_torques_l2",
            "Episode_Reward/dof_acc_l2",
            "Episode_Reward/action_rate_l2",
            "Episode_Reward/dof_vel_l2",
            "Episode_Reward/feet_air_time",
            "Episode_Reward/undesired_contacts",
            "Episode_Reward/flat_orientation_l2",
            "Episode_Termination/died",
            "Episode_Termination/time_out"
        ]

        # Open the file and create a CSV writer
        # Using 'w' mode will overwrite the file at the start of each new training run
        self._log_file = open(self._log_file_path, 'w', newline='')
        self._csv_writer = csv.DictWriter(self._log_file, fieldnames=self._log_fieldnames)
        self._csv_writer.writeheader()

        self._stop_plotter_event = threading.Event()
        self._plot_process = Process(target=live_plotter, args=(self._log_file_path, self._stop_plotter_event))
        self._plot_process.start()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
        self.goal_pos_visualizer = self._create_random_color_markers(self.cfg.marker_colors, 0.25, "/Visuals/Command/goal_position")
        self.identifier_visualizer = self._create_random_color_markers(self.cfg.marker_colors, 0.075, "/Visuals/Command/identifier")

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self._robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.processed_actions = self.cfg.action_scale * self.actions + self._robot.data.default_joint_pos
        self._visualize_markers()

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.processed_actions)#, joint_ids=self.dof_idx)
        #self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        self.previous_actions = self.actions.clone()
        
        # Get unit vector and distance to target
        relative_target_pos = self._desired_pos - self._robot.data.root_pos_w[:, :2]
        distance_to_target = torch.linalg.norm(relative_target_pos, dim=1, keepdim=True)
        unit_vector_to_target = relative_target_pos / (distance_to_target + 1e-6)

        # Get unit vector and distance to next target
        relative_next_target_pos = self._next_desired_pos - self._robot.data.root_pos_w[:, :2]
        distance_to_next_target = torch.linalg.norm(relative_next_target_pos, dim=1, keepdim=True)
        unit_vector_to_next_target = relative_next_target_pos / (distance_to_next_target + 1e-6)


        # Concatenate the selected observations into a single tensor.
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    unit_vector_to_target,
                    distance_to_target,
                    unit_vector_to_next_target,
                    distance_to_next_target,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    #height_data,
                    self.actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Robots were killing themselfs and hoping to spawn closer + not rewarded enough for getting closer
        # it also could be launching itself as fast as it can forward dying in the process

        # - Changes target if rached -
        # Get robot and target positions (only x, y)
        robot_pos = self._robot.data.root_pos_w[:, :2]
        # Calculate distance
        relative_pos = self._desired_pos - robot_pos
        dist_to_target = torch.linalg.norm(relative_pos, dim=1)

        # Check if distance is within tolerance
        reached_target = dist_to_target < self.cfg.success_tolerance
        reached_target_ids = reached_target.nonzero(as_tuple=False).squeeze(-1)

        if len(reached_target_ids) > 0:
            # Generate a new target immediately for the environments that reached theirs
            self._move_next_targets(reached_target_ids)


        # - Rewards -
        # Reward for progress towards the target
        #progress_reward = (self._previous_dist_to_target - dist_to_target) * 30 * self._targets_reached
        # Multipliy progress reward by distance to target (closer = more reward)
        #progress_reward *= (5 / torch.maximum(dist_to_target, torch.tensor(1, device=self.device)))
        progress_reward = torch.exp(-dist_to_target / 2.0) * torch.exp(self._targets_reached / self.cfg.progress_target_divisor)
        # Velocity alignment the target
        unit_vector_to_target = relative_pos / (dist_to_target.unsqueeze(1) + 1e-6)
        velocity_alignment_reward = torch.nn.functional.cosine_similarity(
            self._robot.data.root_lin_vel_w[:, :2], unit_vector_to_target, dim=1
        )
        #print("--- vel:", self._robot.data.root_lin_vel_w[:, :2], "\nunit_vector_to_target:", unit_vector_to_target,
        #    "\nvelocity_alignment_reward", velocity_alignment_reward)

        # Bonus for getting to target (1 reward for first target, 1.5 for second, 2 for third, etc...)
        target_reward = torch.zeros(self.num_envs, device=self.device)
        target_reward[reached_target_ids] = 0.5 + self._targets_reached[reached_target_ids] * 0.5

        self._previous_dist_to_target[:] = dist_to_target.clone()
        # linear velocity tracking
        #lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        #lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        #yaw_rate_error = torch.square(self.commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        #yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self.actions - self.previous_actions), dim=1)
        # joint velocity
        dof_vel = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self.feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self.feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1)
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        rewards = {
            #"track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            #"track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "progress_reward": progress_reward * self.cfg.progress_reward_scale * self.step_dt,
            "velocity_alignment_reward": velocity_alignment_reward * self.cfg.velocity_alignment_reward_scale * self.step_dt,
            "reach_target_reward": target_reward * self.cfg.reach_target_reward * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "dof_vel_l2": dof_vel * self.cfg.dof_vel_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_sums["total_reward"] += reward
        return reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        full_time_out = self.episode_length_buf >= self.max_episode_length - 1

        #net_contact_forces = self._contact_sensor.data.net_forces_w_history
        #died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self.base_id], dim=-1), dim=1)[0] > 10.0, dim=1)

        # Terminate if body is below certain height
        #body_height = self._robot.data.body_pos_w[:, self.body_id, 2]
        #died = body_height < 0.2

        # died if gravity is positive (flipped over)
        died = self._robot.data.projected_gravity_b[:, 2] > 0.0
        self._time_since_target += self.step_dt
        timed_out = self._time_since_target > self.cfg.time_out_per_target

        terminate = died | timed_out

        return terminate, full_time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
            
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
            self._time_since_target[:] = torch.rand(self.num_envs, device=self.device) * self.cfg.time_out_per_target

        # Reset actions
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        
        # Reset
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


        # Set Next target positions to be reset position
        self._next_desired_pos[env_ids] = self._robot.data.root_pos_w[env_ids, :2].clone()
        self._move_next_targets(env_ids)
        self._move_next_targets(env_ids)  # call twice to initialize both current and next target positions
        
        
        # Logging
        extras = dict()

        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

        if self._csv_writer is not None:
            # We need to detach tensors and move them to CPU before logging
            log_data_cpu = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in self.extras["log"].items()}
            self._csv_writer.writerow(log_data_cpu)
            # Flush the file buffer to ensure data is written to disk immediately
            self._log_file.flush()


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
        
        radius = (self.cfg.point_min_distance - self.cfg.point_max_distance) * torch.rand(num_resets, device=self.device) + self.cfg.point_max_distance
        angle = 2 * math.pi * torch.rand(num_resets, device=self.device)
        new_target_pos_xy = self._desired_pos[env_ids] + torch.stack([
            radius * torch.cos(angle),
            radius * torch.sin(angle)
        ], dim=1)

        # for debugging just go straight forward (2, 0)
        #new_target_pos_xy = self._desired_pos[env_ids] + torch.tensor([1.0, 0.0], device=self.device).unsqueeze(0).repeat(num_resets, 1)

        # Update current and next desired positions
        self._desired_pos[env_ids] = self._next_desired_pos[env_ids].clone()
        self._next_desired_pos[env_ids] = new_target_pos_xy
        
        # set _previous_dist_to_target to initial distance to target
        relative_pos = self._desired_pos[env_ids] - self._robot.data.root_pos_w[env_ids, :2]
        self._previous_dist_to_target[env_ids] = torch.linalg.norm(relative_pos, dim=1)

        self._time_since_target[env_ids] = 0.0

    def _create_random_color_markers(self, num_markers: int, radius: float, prim_path: str) -> VisualizationMarkers:
        # Seed the random number generator for reproducibility
        markers = {}
        for i in range(num_markers):
            hue = (i / min(23.0, num_markers)) % 1.0  # Evenly space hues for maximum distinction
            saturation = 0.9
            value = (i % 3) / 3.0 * 0.5 + 0.5  # Vary value to add brightness differences
            rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)

            # Create the unique key for the marker
            marker_key = f"sphere{i}"

            # Create the sphere configuration with the random color
            markers[marker_key] = sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb_color),
            )

        # Create the top-level configuration for the visualizer
        marker_cfg = VisualizationMarkersCfg(prim_path, markers)
        return VisualizationMarkers(marker_cfg)

    def _visualize_markers(self):
        # get marker locations and orientations
        desired_pos_3d = torch.cat(
            [self._desired_pos, 0.2 * torch.ones((self.num_envs, 1), device=self.device)],
            dim=-1
        )

        # marker index list (repeating 0-num_markers)
        marker_indices = torch.arange(self.num_envs, device=self.device) % self.cfg.marker_colors

        self.goal_pos_visualizer.visualize(desired_pos_3d, marker_indices=marker_indices)
        self.identifier_visualizer.visualize(
            self._robot.data.root_pos_w + torch.tensor([0.0, 0.0, 0.3], device=self.device).unsqueeze(0), 
            marker_indices=marker_indices)
        
    def close(self):
        """Cleans up resources used by the environment, like the plotting thread and log file."""
        print("Closing environment and shutting down plotter...")
        # Signal the plotter thread to terminate
        if hasattr(self, "_stop_plotter_event"):
            self._stop_plotter_event.set()

        # Wait for the plotter thread to finish
        if hasattr(self, "_plot_process") and self._plot_process.is_alive():
            self._plot_process.join(timeout=3)
            if self._plot_process.is_alive():
                print("Plotter thread did not shut down gracefully.")

        # Close the log file
        if hasattr(self, "_log_file") and not self._log_file.closed:
            self._log_file.close()

        super().close()

