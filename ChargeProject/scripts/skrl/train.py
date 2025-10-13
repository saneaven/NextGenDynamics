# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import torch.nn as nn

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
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
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

import omni
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import ChargeProject.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent

import torch  # Make sure torch is imported

# ADD THIS HELPER FUNCTION HERE
def flatten_rnn_parameters(model: torch.nn.Module):
    """
    Recursively finds all RNN modules in a model and calls flatten_parameters()
    to improve performance.
    """
    for module in model.modules():
        if isinstance(module, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
            print("Flattening RNN parameters to improve performance.")
            module.flatten_parameters()

class LSTMWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device,
                 rollout_length=24, minibatch_size=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.device = device
        self.hidden = None  # to store (h, c)
        self.rollout_length = rollout_length
        self.minibatch_size = minibatch_size

    def reset_rollout(self, batch_size, device=None):
        """Initialize hidden states for rollout (called once on init)"""
        if device is None:
            device = next(self.parameters()).device

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        self.hidden_states = [(h0.clone(), c0.clone()) for _ in range(self.rollout_length)]
        self.current_step = 0
        self.rollout_batch_size = batch_size

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)  # add seq dimension

        batch_size = x.shape[0]

        # Determine mode automatically
        rollout_mode = (self.rollout_batch_size is not None and batch_size == self.rollout_batch_size)

        if rollout_mode:
            # Rollout mode: stepwise hidden states
            if self.hidden_states is None:
                self.reset_rollout(batch_size, device=x.device)

            h, c = self.hidden_states[self.current_step]
            out, (h_new, c_new) = self.lstm(x, (h, c))
            self.hidden_states[self.current_step] = (h_new.detach(), c_new.detach())

            # Advance rollout step
            self.current_step = (self.current_step + 1) % self.rollout_length
            return out[:, -1, :]
        else:
            # Training/minibatch forward: ignore stored hidden states
            out, _ = self.lstm(x)
            return out[:, -1, :]

def insert_lstm(model_name, runner, agent_cfg, env_cfg):
    model = runner.agent.models[model_name]
    network = agent_cfg["models"][model_name]["network"]
    found = []
    for layer in network:
        if layer.get("type", "").lower() == "lstm":
            # change model.{layer["name"]}_container
            print(f"Inserting LSTM layer '{layer['name']}' into model '{model_name}'")
            setattr(model, f'{layer["name"]}_container', 
                LSTMWrapper(
                    input_size=env_cfg.observation_space,
                    hidden_size=layer["hidden_size"],
                    num_layers=layer["num_layers"],
                    device=env_cfg.sim.device
                ).to(env_cfg.sim.device)
            )
            found += [layer["name"]]
            
    #for layer in network:
        # if input is in found
        #if layer.get("input", "") in found:
            #lstm_out, (h, c) = self.lstm_container(states)
            #rnn_out = self.rnn_container(lstm_out)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    print("---------------")
    print(agent_cfg_entry_point, agent_cfg)
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = os.path.join(log_root_path, log_dir)
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # create isaac environment
    env_cfg.log_dir = log_dir
    env_cfg.cameras = args_cli.enable_cameras
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    insert_lstm("policy", runner, agent_cfg, env_cfg)
    insert_lstm("value", runner, agent_cfg, env_cfg)
    from types import MethodType

    def patched_pre_interaction(self, timestep: int, timesteps: int):
        done = (env.terminated | env.truncated).view(-1)
        if done.any():
            for name in ["policy", "value"]:
                lstm = getattr(self.models[name], "rnn_container", None)
                if lstm and lstm.hidden is not None:
                    h, c = lstm.hidden
                    h[:, done, :] = 0
                    c[:, done, :] = 0
                    lstm.hidden = (h, c)

    #runner.agent.pre_interaction = MethodType(patched_pre_interaction, runner.agent)

    # --------------------------------------------------------------------
    # recursively find and flatten RNN parameters to avoid warnings and improve performance
    if hasattr(runner.agent, 'model'):
        print("Searching for RNN modules to flatten...")
        flatten_rnn_parameters(runner.agent.model)
    # --------------------------------------------------------------------

    env_cfg._agent = runner.agent  # to access the agent from within the env for logging
    print(dir(runner))
    runner.agent.env = runner._env # to access terminations/truncations from within the agent for LSTM state reset

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
    torch.cuda.profiler.start()
    # run training
    runner.run()

    # close the simulator
    env.close()

    torch.cuda.profiler.stop()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
