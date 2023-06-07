from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry
from legged_gym.utils.rl import StepEstimatorOptimizer

from time import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg.runner.resume = True
    train_cfg.algorithm.train_step_estimator = True
    env_cfg.normalization.gait_profile = True
    env_cfg.steps_forecast.method = "network"

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg, cfg_ppo=train_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy()
    
    env_cfg.export(os.path.join(ppo_runner.log_dir, "config.yaml"))
    writer = SummaryWriter(log_dir=ppo_runner.log_dir, flush_secs=10)
    optim: StepEstimatorOptimizer = ppo_runner.alg.est_optim

    actions = torch.zeros(env.num_envs, env.num_actions + 2 * len(env.feet_indices), device=args.rl_device)
    obs = env.get_observations()
    num_steps_per_env = train_cfg.runner.num_steps_per_env
    
    loss = torch.zeros(num_steps_per_env, device=args.rl_device)
    t1 = time()
    for i in range(train_cfg.runner.max_iterations * num_steps_per_env):
        actions[:] = policy(obs.to(args.rl_device)).detach()
        obs, _, _, dones, _ = env.step(actions)
        loss[i % num_steps_per_env] = optim.process_env_step(obs, dones, env.root_states[:, 0:7], env.feet_new_step, env.feet_origin[..., 1:4])

        if i > 0 and (i+1) % num_steps_per_env == 0:
            it = i // num_steps_per_env
            lr = optim.complete_iteration()
            it_loss = loss.mean()
            writer.add_scalar("Step estimator/quadratic error", it_loss, it)
            writer.add_scalar("Step estimator/learning rate", lr, it)
            t3 = time()
            print(f"Iteration {it}/{train_cfg.runner.max_iterations}:")
            print(f"---- average loss:      {it_loss:.2e}")
            print(f"---- learning rate:     {lr:.2e}")
            print(f"---- iteration took     {t3 - t1:.3f} seconds in total")

            if it % ppo_runner.save_interval == 0:
                ppo_runner.save(os.path.join(ppo_runner.log_dir, 'model_{}.pt'.format(it)))

            t1 = time()

if __name__ == "__main__":
    args = get_args()
    train(args)
