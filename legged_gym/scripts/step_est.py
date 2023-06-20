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
    env_cfg.env.episode_length_s = 300
    env_cfg.env._enforce_episode_length_s = True
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s
    env_cfg.commands._enforce_resampling_time = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand._enforce_push_robots = True
    
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg, cfg_ppo=train_cfg)
    env.enable_viewer_sync = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy()
    
    env_cfg.export(os.path.join(ppo_runner.log_dir, "config.yaml"))
    writer = SummaryWriter(log_dir=ppo_runner.log_dir, flush_secs=10)
    optim: StepEstimatorOptimizer = ppo_runner.alg.est_optim

    actions = torch.zeros(env.num_envs, env.num_actions + 2 * len(env.feet_indices), device=args.rl_device)
    obs = env.get_observations()
    num_steps_per_env = train_cfg.runner.num_steps_per_env
    
    loss = torch.zeros(num_steps_per_env, device=args.rl_device)

    for it in range(train_cfg.runner.max_iterations):

        t1 = time()
        loss_min, loss_max = 9999, 0

        for i in range(num_steps_per_env):  
            while env.pause:
                env.render()        
            actions[:] = policy(obs.to(args.rl_device)).detach()
            obs, _, _, dones, _ = env.step(actions)
            loss[i], lmax, lmin = optim.process_env_step(obs, dones, env)
            loss_min = min(loss_min, lmin)
            loss_max = max(loss_max, lmax)

        lr = optim.complete_iteration()
        it_loss = loss.mean()
        
        t2 = time()
    
        writer.add_scalar("Step estimator/quadratic error", it_loss, it)
        writer.add_scalar("Step estimator/learning rate", lr, it)
   
        print(f"Iteration {it}/{train_cfg.runner.max_iterations}:")
        print(f"---- average loss:      {it_loss:.2e}")
        print(f"---- min loss:          {loss_min:.2e}")
        print(f"---- max loss:          {loss_max:.2e}")
        print(f"---- learning rate:     {lr:.2e}")
        print(f"---- iteration took     {t2 - t1:.3f} seconds in total")

        if it % ppo_runner.save_interval == 0:
            ppo_runner.save(os.path.join(ppo_runner.log_dir, 'model_{}.pt'.format(it)))

    ppo_runner.save(os.path.join(ppo_runner.log_dir, 'model_{}.pt'.format(train_cfg.runner.max_iterations)))

if __name__ == "__main__":
    args = get_args()
    train(args)
