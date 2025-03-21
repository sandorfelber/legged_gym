from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules.actor_critic import ActorCritic, create_network, get_activation
from rsl_rl.algorithms import PPO
import torch
import numpy as np

from math import prod
from legged_gym.utils.math import quat_apply_yaw_inverse, ravel_indices, unravel_indices

class OnLeggedPolicyRunner(OnPolicyRunner):

    env: "LeggedRobot"

    def __init__(self,
                 env: "LeggedRobot",
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        train_cfg["policy"]["num_feet"] = len(env.feet_indices)
        
        train_cfg["algorithm"]["num_obs"] = env.num_obs
        train_cfg["algorithm"]["num_envs"] = env.num_envs
        train_cfg["algorithm"]["num_feet"] = len(env.feet_indices)

        super().__init__(env, train_cfg, log_dir, device)

    def export(self, infos=None):
        d = super().export(infos)
        return d
    
    def load_dict(self, loaded_dict, load_optimizer=True):
        super().load_dict(loaded_dict, load_optimizer)

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)

class StepEstimatorPPO(PPO):
    def __init__(self, *args, train_step_estimator=False, num_obs=None, num_envs=None, num_feet=None, **kwargs):
        super().__init__(*args, **kwargs)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        super().init_storage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape)

    def process_env_step(self, rewards, dones, infos):
        super().process_env_step(rewards, dones, infos)
    
    def update(self):
        return super().update()

class Wrapper:
    def __init__(self, item: torch.nn.Module):
        self.item = item

class StepEstimatorActorCritic(ActorCritic):
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        estimator_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        estimate_steps = False,
                        num_feet=None,
                        **kwargs):
        
        self.num_actions = num_actions
        self.num_feet = num_feet
        self.num_actor_obs = num_actor_obs

        super().__init__(num_actor_obs, num_critic_obs, num_actions,
                         actor_hidden_dims, critic_hidden_dims, activation,
                         init_noise_std, **kwargs)
        
    def update_distribution(self, observations):
        super().update_distribution(observations[:, :self.num_actor_obs])

    def act(self, observations, **kwargs):
        actions = super().act(observations[:, :self.num_actor_obs], **kwargs)
        return self._cat_est(observations, actions)
        
    def get_actions_log_prob(self, actions):
        return super().get_actions_log_prob(actions[..., :self.num_actions])

    def act_inference(self, observations):
        actions = super().act_inference(observations[:, :self.num_actor_obs])
        return self._cat_est(observations, actions)
   
    def _cat_est(self, observations, actions):
        return actions
    
    def train(self, mode=True):
        return super().train(mode)

    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)

class TensorBatchList:
    def __init__(self, batch_shape, data_size, capacity=32, dtype=torch.float, device="cuda:0"):
        """ - batch_size (tuple)
            - data_size (int): num of observations per object
            - capacity (int): max num of data per object. Once the capacity is exceeded, new data will override
            existing observations."""
        self.device = device
        self.capacity = capacity
        self.data = torch.zeros(*batch_shape, capacity, data_size, device=self.device, dtype=dtype)
        self.len = torch.zeros(batch_shape, device=self.device, dtype=torch.long)

        self._sample_cache = torch.zeros(*batch_shape, data_size, device=self.device, dtype=dtype)
        self._len_index = torch.zeros(*batch_shape, 1, data_size, device=self.device, dtype=torch.long)
        
        self._range = torch.arange(capacity, device=self.device)
        self._mask = torch.zeros(*batch_shape, capacity, device=self.device, dtype=torch.bool)
        self._nonzero = torch.zeros(prod(batch_shape)*capacity, device=self.device, dtype=torch.long)

    def append(self, data):
        """Add observations for each object.
        - data: Tensor of shape [*batch_size, data_size] containing the data for each object"""
        self._len_index[:] = self.len[..., None, None] % self.capacity
        self._sample_cache.copy_(data)
        self.data.scatter_(self.len.dim(), self._len_index, self._sample_cache[..., None, :])
        self.len += 1

    def generate_batch(self, where, target, max_size, data_out=None, target_out=None, indices_out=None):
        """Generates batches from the observations
        - where (bool tensor of shape batch_size)
        - target (tensor of shape [*batch_size, K] where K is any int)

        Buffers:
        - data_out (optional tensor of shape [max_size, data_size])
        - target_out (optional tensor of shape [max_size, K])
        - indices_out (optional tensor of shape [max_size])
        
        Yields:
        (data, target, indices) tensors of shape [N, data_size], [N, K], [N, 3] where N <= max_size
        indices contains the position of the data in self.data."""
        if data_out is None:
            data_out = torch.zeros(max_size, self.data.shape[-1], device=self.device, dtype=self.data.dtype)
        if target_out is None:
            target_out = torch.zeros(max_size, target.shape[-1], device=self.device, dtype=target.dtype)
        if indices_out is None:
            indices_out = torch.zeros(max_size, 3, device=self.device, dtype=torch.long)

        mask = self.remove(where).view(-1)
        l = mask.sum().item()      
        self._nonzero[:l] = mask.nonzero().squeeze(1)

        target = target.view(-1, target.shape[-1])
        data = self.data.view(-1, self.data.shape[-1])

        for i in range(0, l, max_size):
            j = min(i+max_size, l)

            data_out[:j-i] = data[self._nonzero[i:j]]
            unravel_indices(self._nonzero[i:j], (*self.len.shape, self.capacity), out=indices_out[:j-i])
            ravel_indices(indices_out[:j-i, :-1], self.len.shape, in_place=False, out=self._nonzero[i:j])
            target_out[:j-i] = target[self._nonzero[i:j]]
            
            yield data_out[:j-i], target_out[:j-i], indices_out[:j-i]

    def remove(self, where):
        """Removes the observations for some objects.
        - where (bool tensor of shape batch_size)
        
        Returns: a mask of shape [*batch_size, capacity] that can be used to retrieve the data from 'self.data'.
        The mask is valid until the next call to 'append' or 'remove'."""
        self._mask.copy_(where[..., None])
        range_mask = self._range.expand(self._mask.shape) < self.len[..., None]
        self.len[self._mask[..., 0]] = 0
        self._mask.logical_and_(range_mask)
        return self._mask
    
    def complete_iteration(self):
        lr = self.scheduler.get_last_lr()[0]
        self.scheduler.step()
        return lr
    
    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()}
    
    def load_state_dict(self, dict):
        self.optimizer.load_state_dict(dict["optimizer"])
        self.scheduler.load_state_dict(dict["scheduler"])