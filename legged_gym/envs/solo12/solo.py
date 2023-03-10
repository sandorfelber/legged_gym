
from time import time, sleep
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

class Solo12(LeggedRobot):

    def _init_buffers(self):
        super()._init_buffers()
        self.last_last_q_target = self.default_dof_pos
        self.last_q_target = self.default_dof_pos
        self.q_target = self.default_dof_pos

    def _post_physics_step_callback(self):

        self.last_last_q_target = self.last_q_target
        self.last_q_target = self.q_target
        self.q_target = self._get_q_target(self.actions)

        self.feet_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.feet_state = gymtorch.wrap_tensor(self.feet_state).view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices]

        super()._post_physics_step_callback()

    def _get_q_target(self, actions):
        return self.default_dof_pos + self.cfg.control.action_scale * actions

    def _reward_velocity(self):
        v_speed = torch.hstack((self.base_lin_vel[:, :2], self.base_ang_vel[:, 2:3]))
        vel_error = torch.sum(torch.square(self.commands[:, :3] - v_speed), dim=1)
        return torch.exp(-vel_error)

    def _reward_foot_clearance(self):
        # state: x, y, z, [0:3]
        #       q0, q1, q2, q3, [3:7]
        #       v_x, v_y, v_z, [7:10]
        #       v_w_x, v_w_y, v_w_z [10:13]
        # (q0, q1, q2, q3) is the quaternion representing the orientation of the body
        
        feet_z = self.feet_state[..., 2]
        height_err = torch.square(feet_z - self.cfg.control.feet_height_target)
        feet_speed = torch.sum(torch.square(self.feet_state[..., 7:9]), dim=2)
        return torch.sum(height_err * torch.sqrt(feet_speed), dim=1)

    def _reward_foot_slip(self):
        # inspired from LeggedRobot::_reward_feet_air_time
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 

        speed = torch.sum(torch.square(self.feet_state[..., 7:9]), dim=2)

        return torch.sum(contact_filt * speed, dim=1)
    
    def _reward_vel_z(self):
        r = torch.square(self.base_lin_vel[:, 2])
        return r
    
    @staticmethod
    def _abs_angle(angle):
        return torch.where(angle > torch.pi, 2 * torch.pi - angle, angle)
    
    def _reward_roll_pitch(self):
        roll, pitch, _ = get_euler_xyz(self.root_states[:, 3:7])
        roll, pitch = Solo12._abs_angle(roll), Solo12._abs_angle(pitch)
        self.reset_buf |= roll[:] > 3
        # HACK: this partially fixes contact on base/truck not being detected (why?)
        # this works because "check_termination" (which resets "self.reset_buf") is called before the rewards
        
        return torch.sum(torch.square(torch.stack((roll, pitch), dim=1)), dim=1)
    
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
        return torch.sum(torch.square(self.q_target - 2 * self.last_q_target + self.last_last_q_target), dim=1)