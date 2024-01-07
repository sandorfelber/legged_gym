
from time import time, sleep
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.utils.math import wrap_to_pi
from .solo12_config import Solo12Cfg, Solo12CfgPPO

class Solo12(LeggedRobot):

    cfg: Solo12Cfg
    cfg_ppo = Solo12CfgPPO()

    def _init_buffers(self):
        super()._init_buffers()
        # super()._process_dof_props(,0)
        # self.torque_limits= torch.tensor([1.0, 2.0, 3.0, 4, 5])
        #self.torque_weights = torch.tensor([1.0, 1.0, 5.0,
                                            #1.0, 1.0, 5.0,
                                            #1.0, 1.0, 5.0,
                                            #1.0, 1.0, 5.0])
        #self.torque_weights = self.torque_weights.to("cuda:0")
        #self.torque_limits = torch.tensor([1.9, 1.9, 1.9, 1.9,
                                             #1.9, 1.9, 1.9, 1.9,
                                             #1.9, 1.9, 1.9, 1.9])
        #self.torque_limits = self.torque_limits.to("cuda:0")

        # q_target(t-2)
        self.last_last_q_target = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        # q_target(t-1)
        self.last_q_target = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        # q_target(t)
        self.q_target = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        
        self.last_last_q_target[:] = self.default_dof_pos
        self.last_q_target[:] = self.default_dof_pos
        self.q_target[:] = self.default_dof_pos

        ################################
        self.torque_limits = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        self.torque_limits[:] = torch.tensor([1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9])

        self.torque_weights = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        self.torque_weights[:] = torch.tensor([1., 1., 1.25, 1., 1., 1.25, 1., 1., 1.25, 1., 1., 1.25])
        ################################

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.last_last_q_target[env_ids] = self.default_dof_pos
        self.last_q_target[env_ids] = self.default_dof_pos
        self.q_target[env_ids] = self.default_dof_pos

    def check_termination(self):
        super().check_termination()
        self.reset_buf |= torch.abs(self.roll[:]) > 2
        # HACK: this partially fixes contact on base/truck not being detected (why?)        

    def _post_physics_step_callback(self):
        self.last_last_q_target[:] = self.last_q_target
        self.last_q_target[:] = self.q_target
        self.q_target[:] = self._get_q_target(self.actions)

        self.roll, self.pitch = self._get_roll_pitch()

        super()._post_physics_step_callback()
       
    def _get_q_target(self, actions):
        return self.cfg.control.action_scale * actions + self.default_dof_pos
     
    def _get_roll_pitch(self):
        roll, pitch, _ = get_euler_xyz(self.root_states[:, 3:7])
        roll, pitch = wrap_to_pi(roll), wrap_to_pi(pitch)
        return roll, pitch

    # --- rewards (see paper) ---

    def _reward_velocity(self):
        v_speed = torch.hstack((self.base_lin_vel[:, :2], self.base_ang_vel[:, 2:3]))
        vel_error = torch.sum(torch.square(self.commands[:, :3] - v_speed), dim=1)
        #print("VEL : ", torch.exp(-vel_error).size())
        return torch.exp(-vel_error)
    
    def _reward_foot_clearance(self):
        feet_z = self.get_feet_height()
        height_err = torch.square(feet_z - self.cfg.control.feet_height_target)
        feet_speed = torch.sum(torch.square(self.body_state[:, self.feet_indices, 7:9]), dim=2)
        #print("footclearance : ", torch.sum(height_err * torch.sqrt(feet_speed), dim=1).size())
        return torch.sum(height_err * torch.sqrt(feet_speed), dim=1)

    def _reward_foot_slip(self):
        # inspired from LeggedRobot::_reward_feet_air_time
        speed = torch.sum(torch.square(self.body_state[:, self.feet_indices, 7:9]), dim=2)

        return torch.sum(self.filtered_feet_contacts * speed, dim=1)
    
    def _reward_vel_z(self):
        r = torch.square(self.base_lin_vel[:, 2])
        return r
   
    def _reward_roll_pitch(self):
        return torch.sum(torch.square(torch.stack((self.roll, self.pitch), dim=1)), dim=1)
    
    def _reward_joint_pose(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _reward_power_loss(self):
        TAU_U = 0.0477 # [Nm]
        B = 0.000135 # [Nm.s]
        K = 4.81 # [Nms]

        q_dot = self.dof_vel
        tau_f = TAU_U * torch.sign(q_dot) + B * q_dot
        P_f = tau_f * q_dot
        P_j = torch.square(self.torques + tau_f) / K

        return torch.sum(P_f + P_j, dim=1)
    
    def _reward_smoothness_1(self):
        return torch.sum(torch.square(self.q_target - self.last_q_target), dim=1)
    
    def _reward_smoothness_2(self):
        #print("SMOOTHNESS REWARD: ", torch.sum(torch.square(self.q_target - 2 * self.last_q_target + self.last_last_q_target), dim=1).size())
        return torch.sum(torch.square(self.q_target - 2 * self.last_q_target + self.last_last_q_target), dim=1)
    
    def _reward_torques(self):
         # Penalize torques
         #print("TORQUE WEIGHTED : ", torch.sum(torch.square(self.torques * self.torque_weights), dim=1))
         return torch.sum(torch.square(self.torques * self.torque_weights), dim=1)
    
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        # self.torque_limits =  [2, 2, 2,
        #                        2, 2, 2,
        #                        2, 2, 2,
        #                        2, 2, 2]
        #self.torque_limits = 1.9
        #print(self.torque_limits)
        #return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
        #print("SHAPE:", self.torques.shape)
        #print("LIMITSSHAPE:", self.torque_limits.shape)
        # print("VIEW TORQUES", self._compute_torques(self.actions).view(self.torques.shape))
        #print("TORQUES : ", torch.abs(self.torques).size())
        #print("TORQUE LIMITS: ", self.torque_limits.size())
        #print("TORQUE REWARDS NO CLIP:" , torch.sum((((torch.abs(self.torques)) - self.torque_limits)), dim=1).size())
        # print("TORQUE REWARDS CLIP INSIDE:" , torch.sum((((torch.abs(self.torques)) - self.torque_limits).clip(min=0.)), dim=1))
        # print("TORQUE REWARDS:" , torch.sum((torch.abs(self.torques) - self.torque_limits).clip(min=0.), dim=1))
        #print("TORQUE LIMIT TENSOR:", torch.sum((torch.abs(self.torques) - self.torque_limits* self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1))
        #print("EXPONENTIAL outside : ", torch.exp(torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1))-1)
        #print("EXPONENTIAL inside : ", torch.sum(torch.exp(torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1))
        return torch.exp(torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1))-1
    
    #def _reward_exp_torque_limits(self):
        # penalize torques too close to the limit
        # self.torque_limits =  [2, 2, 2,
        #                        2, 2, 2,
        #                        2, 2, 2,
        #                        2, 2, 2]
        #self.torque_limits = 1.9
        #print(self.torque_limits)
        #return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
        #print("SHAPE:", self.torques.shape)
        #print("LIMITSSHAPE:", self.torque_limits.shape)
        # print("VIEW TORQUES", self._compute_torques(self.actions).view(self.torques.shape))
        #print("TORQUES : ", torch.abs(self.torques).size())
        #print("TORQUE LIMITS: ", self.torque_limits.size())
        #print("TORQUE REWARDS NO CLIP:" , torch.sum((((torch.abs(self.torques)) - self.torque_limits)), dim=1).size())
        # print("TORQUE REWARDS CLIP INSIDE:" , torch.sum((((torch.abs(self.torques)) - self.torque_limits).clip(min=0.)), dim=1))
        # print("TORQUE REWARDS:" , torch.sum((torch.abs(self.torques) - self.torque_limits).clip(min=0.), dim=1))
        #print("TORQUE LIMIT TENSOR:", torch.sum((torch.abs(self.torques) - self.torque_limits).clip(min=0.), dim=1).size())
        #return torch.sum(torch.exp(torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.exp_soft_torque_limit).clip(min=0.), dim=1)
    