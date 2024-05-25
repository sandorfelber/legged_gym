
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
        self.tunnels_on = True # REDUNDANT!!! (ALREADY DEFINED IN legged_robot.PY)
        self.bridges_on = True # REDUNDANT!!! (ALREADY DEFINED IN legged_robot.PY)
        ################################
        self.torque_limits = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        self.torque_limits[:] = torch.tensor([1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9])

        self.torque_weights = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        self.torque_weights[:] = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
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
     
    def _get_roll_pitch(self): # returns roll and pitch angles (not angular velocity)
        roll, pitch, _ = get_euler_xyz(self.root_states[:, 3:7])
        roll, pitch = wrap_to_pi(roll), wrap_to_pi(pitch)
        return roll, pitch

    # --- rewards (see paper) ---

    # def _reward_velocity(self):
    #     v_speed = torch.hstack((self.base_lin_vel[:, :2], self.base_ang_vel[:, 2:3]))
    #     vel_error = torch.sum(torch.square(self.commands[:, :3] - v_speed), dim=1)
    #     #print("VEL : ", torch.exp(-vel_error).size())
    #     return torch.exp(-vel_error)
    
    def _reward_lin_vel_x_in_tunnel_bridge(self):
        lin_vel_x = self.base_lin_vel[:, 0]
        condition = (self.tunnels_on & self.tunnel_condition) | (self.bridges_on & self.bridge_condition)
        return torch.where(condition, torch.square(lin_vel_x), torch.zeros_like(lin_vel_x))

        
    def _reward_lin_vel_y_in_tunnel_bridge(self):
        lin_vel_y = torch.square(self.commands[:, 1])
        condition = (self.tunnels_on & self.tunnel_condition) | (self.bridges_on & self.bridge_condition)
        return torch.where(condition, torch.exp(-lin_vel_y), torch.zeros_like(lin_vel_y))

        
    def _reward_tunnel_entrance(self):
        condition = self.tunnels_on & self.tunnel_condition
        reward = torch.sum(torch.square(self.side_heights - (2 * self.middle_heights)), dim=1)
        return torch.where(condition, reward, torch.zeros_like(reward))

        
    def _reward_bridge_entrance(self):
        condition = self.bridges_on & self.bridge_condition
        reward = torch.sum(torch.square((2 * self.middle_heights) - self.side_heights), dim=1)
        return torch.where(condition, reward, torch.zeros_like(reward))

    
    def _reward_foot_clearance(self):
        feet_z = self.get_feet_height()
        height_err = torch.square(feet_z - self.cfg.control.feet_height_target)
        feet_speed = torch.sum(torch.square(self.body_state[:, self.feet_indices, 7:9]), dim=2)
        condition = self.tunnels_on & self.tunnel_condition
        return torch.where(~condition, torch.sum(height_err * torch.sqrt(feet_speed), dim=1), torch.zeros_like(torch.sum(height_err * torch.sqrt(feet_speed), dim=1)))

        
    def _reward_foot_clearance_tunnel_bridge(self):
        feet_z = self.get_feet_height()
        height_err = torch.square(feet_z - self.cfg.control.feet_height_target)
        feet_speed = torch.sum(torch.square(self.body_state[:, self.feet_indices, 7:9]), dim=2)
        condition = (self.tunnels_on & self.tunnel_condition) | (self.bridges_on & self.bridge_condition)
        return torch.where(condition, torch.sum(height_err * torch.sqrt(feet_speed), dim=1), torch.zeros_like(torch.sum(height_err * torch.sqrt(feet_speed), dim=1)))


    def _reward_foot_slip(self):
        # inspired from LeggedRobot::_reward_feet_air_time
        speed = torch.sum(torch.square(self.body_state[:, self.feet_indices, 7:9]), dim=2)

        return torch.sum(self.filtered_feet_contacts * speed, dim=1)
    
    def _reward_vel_z(self):
        r = torch.square(self.base_lin_vel[:, 2])
        return r
   
    def _reward_pitch(self):
        #print(torch.sum(torch.stack([torch.square(self.pitch), torch.zeros_like(torch.square(self.pitch))], dim=1), dim=1))
        # This (pitch reward = pitch^2 + 0) is done to ensure that the reward is always positive and the same size as the batch size N as the first dimension and the same size as the output of the sum operation
        return torch.sum(torch.stack([torch.square(self.pitch), torch.zeros_like(self.pitch)], dim=1), dim=1)
        #return torch.sum(torch.square(self.pitch), dim=1)
    
    # def _reward_roll(self):
    # # ensure stability by default, when tunnels are on then allow for roll
    #     if self.tunnel_on:
    #         # if in tunnel mode, then allow for roll
    #         if self.tunnel_condition[self.ref_env] == True:
    #             print("TUNNEL CONDITION TRUE")
    #             return torch.sum(torch.square(self.roll), dim=1)
    #     else:
    #         return torch.sum(torch.square(self.roll), dim=1)
        
    def _reward_roll(self):
        # Ensure stability by default; when tunnels are on, then allow for roll.
        # Now checking the tunnel_condition for the specific ref_env.
        #print("TUNNEL CONDITION: ", self.tunnel_condition[self.ref_env])
        if self.tunnels_on and self.tunnel_condition[self.ref_env]:
            #print("TUNNEL CONDITION TRUE")
            #print(torch.sum(torch.stack([torch.zeros_like(self.roll), torch.zeros_like(self.roll)], dim=1), dim=1)) 
            #import sys
            #sys.exit()
            # Apply no penalty when in tunnel condition
            #print("ROLL_0")
            return torch.sum(torch.stack([torch.zeros_like(self.roll), torch.zeros_like(self.roll)], dim=1), dim=1)
        else:
            #print("TUNNEL CONDITION FALSE")
            #print(torch.sum(torch.stack([torch.square(self.roll), torch.square(self.roll)], dim=1), dim=1))
            # If not in a tunnel, or if the tunnel feature is turned off, apply the regular penalty for roll.
            #print("ROLL_1")
            return torch.sum(torch.stack([torch.square(self.roll), torch.zeros_like(self.roll)], dim=1), dim=1)
        
    # def _reward_roll_in_tunnel(self):
    #     #print(self.tunnel_on, self.tunnel_condition[self.ref_env])
    #     #print("TUNNEL CONDITION: ", self.tunnel_condition[self.ref_env])
    #     if self.tunnels_on and self.tunnel_condition[self.ref_env]:
    #         # print("_reward_roll_in_tunnel")
    #         #print("ROLL_TUNNEL_1")
    #         #print(torch.sum(torch.stack([torch.square(self.roll), torch.zeros_like(self.roll)], dim=1), dim=1))
    #         #print("HERE")
    #         # import sys
    #         # sys.exit()
    #         return torch.sum(torch.stack([torch.square(self.roll), torch.zeros_like(self.roll)], dim=1), dim=1)
    #     else:
    #         #print("ROLL_TUNNEL_0")
    #         return torch.sum(torch.stack([torch.zeros_like(self.roll), torch.zeros_like(self.roll)], dim=1), dim=1)
        
    def _reward_roll_in_tunnel(self):
        # Check if tunnels are enabled and if the robot is currently inside a tunnel.
        condition = self.tunnels_on & self.tunnel_condition[self.ref_env]

        # Define the desired roll level for tunnels; adjust this based on your specific needs.
        # Assuming a desired roll of around ±0.3 radians might help navigate the tunnel's curvature or terrain.
        desired_roll = 1.56
        roll_deviation = torch.abs(self.roll - desired_roll)

        # Reward achieving near the desired roll. Here, smaller deviations are better.
        roll_reward = torch.exp(-roll_deviation)

        # Apply this reward only when the condition (inside a tunnel) is met.
        # When not in a tunnel, use a general penalty for roll to maintain usual stability.
        reward = torch.where(condition, roll_reward, torch.exp(-torch.square(self.roll)))

        return reward

        
    # def _reward_roll_on_bridge(self):
    #     #print(self.tunnel_on, self.tunnel_condition[self.ref_env])
    #     #print("TUNNEL CONDITION: ", self.tunnel_condition[self.ref_env])
    #     if self.bridges_on and self.bridge_condition[self.ref_env]:
    #         # print("_reward_roll_in_tunnel")
    #         #print("ROLL_TUNNEL_1")
    #         #print(torch.sum(torch.stack([torch.square(self.roll), torch.zeros_like(self.roll)], dim=1), dim=1))
    #         #print("HERE")
    #         # import sys
    #         # sys.exit()
    #         return torch.sum(torch.stack([torch.square(self.roll), torch.zeros_like(self.roll)], dim=1), dim=1)
    #     else:
    #         #print("ROLL_TUNNEL_0")
    #         return torch.sum(torch.stack([torch.zeros_like(self.roll), torch.zeros_like(self.roll)], dim=1), dim=1)

    def _reward_roll_on_bridge(self):
        # Check if bridges are enabled and if the robot is currently on a bridge.
        condition = self.bridges_on & self.bridge_condition[self.ref_env]

        # Calculate the desired roll level; this might be a specific angle or range you target.
        # For example, let's assume you want to encourage a roll of around ±0.5 radians.
        desired_roll = 1.56
        roll_deviation = torch.abs(self.roll - desired_roll)

        # Penalize deviation from the desired roll level.
        # This means the closer the roll to 0.5 radians, the less the penalty (more the reward).
        roll_reward = torch.exp(-roll_deviation)

        # Apply this reward only when the condition (on a bridge) is met.
        # Outside of bridges, you might still want to penalize excessive roll.
        reward = torch.where(condition, roll_reward, torch.exp(-torch.square(self.roll)))

        return reward




    # def _reward_roll(self):
    #     return torch.sum(torch.stack([torch.square(self.pitch), torch.square(self.roll)], dim=1), dim=1)
    #     # ensure stability by default, when tunnels are on then check if in tunnel and then allow for roll
    #     if self.tunnel_on:
    #         # if in tunnel mode and the condition is True, apply no penalty (reward of 0s)
    #         if self.tunnel_condition:
    #             # Assuming self.roll has the same batch size N as the first dimension
    #             # Return a tensor of zeros with the same shape as the output of the sum operation
    #             return torch.zeros_like(torch.sum(torch.square(self.roll), dim=1))
    #         else:
    #             # If not in tunnel condition, calculate penalty for roll
    #             return torch.sum(torch.square(self.roll), dim=1)
    #     else:
    #         # If tunnel mode is not on, calculate penalty for roll
    #         return torch.sum(torch.square(self.roll), dim=1)

    #     return torch.sum(torch.stack([torch.square(self.pitch), torch.square(self.roll)], dim=1), dim=1)
    #     # ensure stability by default, when tunnels are on then check if in tunnel and reward roll linearly
    #     if self.tunnel_on and self.tunnel_condition:
    #         return torch.sum(torch.abs(self.roll), dim=1)
    #     else:
    #         # If tunnel mode is not on, calculate penalty for roll
    #         return torch.zeros_like(torch.sum(torch.square(self.roll), dim=1))
    
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
        #print("SMOOTHNESS REWARD: ", torch.sum(torch.square(self.q_target - self.last_q_target), dim=1))
        return torch.sum(torch.square(self.q_target - self.last_q_target), dim=1)
    
    def _reward_smoothness_2(self):
        #print("SMOOTHNESS REWARD: ", torch.sum(torch.square(self.q_target - 2 * self.last_q_target + self.last_last_q_target), dim=1).size())
        return torch.sum(torch.square(self.q_target - 2 * self.last_q_target + self.last_last_q_target), dim=1)
    
    def _reward_torques(self):
         # Penalize torques
         return torch.sum(torch.square(self.torques * self.torque_weights), dim=1)
    
    def _reward_collision(self):
        condition = self.tunnels_on & self.tunnel_condition
        return torch.where(~condition, torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1), torch.zeros_like(torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)))

        
    def _reward_collision_tunnel(self):
        condition = self.tunnels_on & self.tunnel_condition
        return torch.where(condition, torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1), torch.zeros_like(torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)))

    
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
    