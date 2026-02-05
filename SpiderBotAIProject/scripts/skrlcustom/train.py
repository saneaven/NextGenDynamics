# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl (custom PPO_RNN + SharedRecurrentModel).

This is a project-local version of the ChargeProject custom PPO workflow, adapted for SpiderBotAIProject.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl (custom PPO_RNN).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--debug_vis",
    action="store_true",
    default=False,
    help="Enable command debug visualization markers (spawn points + waypoints).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# torchrun passes `--local-rank` to each worker process. Consume it so Hydra doesn't see it.
if "--local_rank" not in parser._option_string_actions and "--local-rank" not in parser._option_string_actions:
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=0, help=argparse.SUPPRESS)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# In distributed runs, force AppLauncher to pick the correct GPU per-rank before Kit starts.
if args_cli.distributed:
    local_rank = int(os.environ.get("LOCAL_RANK", args_cli.local_rank))
    args_cli.device = f"cuda:{local_rank}"

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


#### SKRL TRAINING SCRIPT BELOW ####

import random
from datetime import datetime

import gymnasium as gym
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. " f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    raise SystemExit(1)

from skrl.agents.torch.ppo import PPO_RNN
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
import pickle

from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import SpiderBotAIProject.tasks  # noqa: F401
from SpiderBotAIProject.tasks.manager_based.spiderbot_ai.agents.skrl_custom_ppo_model import SharedRecurrentModel

# config shortcuts
algorithm = args_cli.algorithm.lower()
if args_cli.agent is None:
    agent_cfg_entry_point = (
        "skrl_custom_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_custom_{algorithm}_cfg_entry_point"
    )
else:
    agent_cfg_entry_point = args_cli.agent


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with custom skrl agent (PPO_RNN)."""
    dist = None
    created_process_group = False
    if args_cli.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", args_cli.local_rank))
        torch.cuda.set_device(local_rank)
        import torch.distributed as dist  # noqa: PLC0415

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
            created_process_group = True
        rank = dist.get_rank()
    else:
        local_rank = 0
        rank = 0
    is_main_process = rank == 0

    # override configurations with non-hydra CLI arguments
    num_envs = args_cli.num_envs
    if num_envs is None:
        env_num_envs = os.environ.get("NUM_ENVS", "").strip()
        if env_num_envs:
            try:
                num_envs = int(env_num_envs)
            except ValueError as exc:
                raise RuntimeError(f"Invalid NUM_ENVS={env_num_envs!r}; expected integer") from exc
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.debug_vis:
        env_cfg.commands.spawn.debug_vis = True
        env_cfg.commands.waypoint.debug_vis = True

    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # Resolve seed (support `--seed -1` by sampling on rank0 and broadcasting).
    seed = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    if seed == -1:
        seed = random.randint(0, 10000) if is_main_process else None
        if dist is not None:
            seed_list = [seed]
            dist.broadcast_object_list(seed_list, src=0)
            seed = seed_list[0]
    seed = int(seed)
    agent_cfg["seed"] = seed
    env_cfg.seed = seed + rank if args_cli.distributed else seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    if is_main_process:
        print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs: {time-stamp}_{run_name}
    if args_cli.distributed:
        log_dir = (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
            if is_main_process
            else None
        )
        if is_main_process and agent_cfg["agent"]["experiment"]["experiment_name"]:
            log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
        log_dir_list = [log_dir]
        dist.broadcast_object_list(log_dir_list, src=0)
        log_dir = log_dir_list[0]
    else:
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
        if agent_cfg["agent"]["experiment"]["experiment_name"]:
            log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'

    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    os.makedirs(log_dir, exist_ok=True)
    if is_main_process:
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        with open(os.path.join(log_dir, "params", "env.pkl"), "wb") as f:
            pickle.dump(env_cfg, f)
        with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as f:
            pickle.dump(agent_cfg, f)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir

    # create isaac environment
    env_cfg.log_dir = log_dir
    env_cfg.cameras = args_cli.enable_cameras
    from SpiderBotAIProject.tasks.manager_based.spiderbot_ai.terrain_gen_usd import ensure_custom_terrain_usd

    # Ensure the terrain USD exists before env creation.
    if args_cli.distributed:
        if is_main_process:
            ensure_custom_terrain_usd(
                size_x=float(env_cfg.height_map_size_x),
                size_y=float(env_cfg.height_map_size_y),
                meter_per_grid=float(env_cfg.height_map_meter_per_grid),
                seed=int(agent_cfg["seed"] or 42),
            )
        dist.barrier()
    else:
        ensure_custom_terrain_usd(
            size_x=float(env_cfg.height_map_size_x),
            size_y=float(env_cfg.height_map_size_y),
            meter_per_grid=float(env_cfg.height_map_meter_per_grid),
            seed=int(agent_cfg["seed"] or 42),
        )

    record_video = bool(args_cli.video and is_main_process)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if record_video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if record_video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    set_seed(env_cfg.seed)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    device = env.device

    memory = RandomMemory(memory_size=agent_cfg["agent"]["rollouts"], num_envs=env.num_envs, device=device)

    models = {}
    model = SharedRecurrentModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        num_envs=env.num_envs,
        init_log_std=agent_cfg["model"]["log_std_init"],
    )
    model = model.to(device)
    models["policy"] = model
    models["value"] = model

    cfg = agent_cfg["agent"].copy()
    if args_cli.distributed and not is_main_process:
        cfg["experiment"] = cfg["experiment"].copy()
        cfg["experiment"]["tensorboard"] = False
        disabled_interval = int(agent_cfg["trainer"]["timesteps"]) + 1
        cfg["experiment"]["write_interval"] = disabled_interval
        cfg["experiment"]["checkpoint_interval"] = disabled_interval

    scheduler_cfg = cfg.get("learning_rate_scheduler")
    if isinstance(scheduler_cfg, str):
        scheduler_map = {"KLAdaptiveLR": KLAdaptiveLR}
        try:
            cfg["learning_rate_scheduler"] = scheduler_map[scheduler_cfg]
        except KeyError as exc:
            raise ValueError(f"Unknown learning_rate_scheduler '{scheduler_cfg}'.") from exc

    shaper_scale = cfg.get("rewards_shaper_scale", 1.0)
    cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * shaper_scale

    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    agent = PPO_RNN(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # load checkpoint (if specified)
    if resume_path:
        if is_main_process:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)

    trainer = SequentialTrainer(cfg=agent_cfg["trainer"].copy(), env=env, agents=agent)
    try:
        trainer.train()
    finally:
        env.close()
        if args_cli.distributed and dist is not None:
            if created_process_group:
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
    simulation_app.close()
