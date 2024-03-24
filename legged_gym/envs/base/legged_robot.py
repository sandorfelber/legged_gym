# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, quat_apply_yaw_inverse, wrap_to_pi
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, CurriculumConfig, AVERAGE_MEASUREMENT, FEET_ORIGIN

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, cfg_ppo: LeggedRobotCfgPPO, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.cfg_ppo = cfg_ppo
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.debug_only_one = False
        self.debug_height_map = False
        self.disable_heights = False # False default - if True then robot goes vrooom vroom, massive speed boost but also blind
        self.tunnels_on = True
        self.init_done = False

        self._parse_cfg()
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        #self.tunnel_condition = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Initialize tunnel_condition flags as False
        self.tunnel_condition = [False] * (1 if self.debug_only_one else self.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()

        self._prepare_reward_function()

        if self.debug_viz:
            import matplotlib
            self.cmap = matplotlib.cm.get_cmap("RdYlGn")
        
        if self.cfg.commands.joystick:
            from legged_gym.utils import gamepad_client
            self.gamepad_client = gamepad_client.GamepadClient()

        self.init_done = True

    def render(self):
        super().render()
        
        if self.viewer and not self.headless and self.enable_viewer_sync and self.follow_env:
            target = self.root_states[self.ref_env,:3].cpu()
            quat = self.root_states[self.ref_env,3:7].cpu()
            
            SIDE = False
            if SIDE:
                lookat = quat_apply_yaw(quat, torch.tensor([0.,0.,-0.5]))
                offset = quat_apply_yaw(quat, torch.tensor([0.,-1,0.5]))    
            else:
                lookat = quat_apply_yaw(quat, torch.tensor([5.,0,0.]))
                offset = quat_apply_yaw(quat, torch.tensor([-2.,0,0.5]))            
                
            self.set_camera(target+offset, target+lookat)


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions[:] = torch.clip(actions[..., :self.num_actions], -clip_actions, clip_actions)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.get_privileged_observations(), self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)        
        self.gym.refresh_rigid_body_state_tensor(self.sim)
      
        self.episode_length_buf += 1
        self.common_step_counter += 1

        contacts = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.filtered_feet_contacts[:] = torch.logical_or(self.last_feet_contacts, contacts)
        self.last_feet_contacts[:] = contacts

        if self.require_feet_origin:
            self._update_feet_origin()

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]


        if self.tunnels_on and not self.disable_heights:
            if self.viewer and self.enable_viewer_sync and self.debug_viz:
                pass
            self._draw_debug_vis()

        # if self.viewer and self.enable_viewer_sync and self.debug_viz:
        #     self._draw_debug_vis()

    def _update_feet_origin(self):

        force_init = self.reset_buf[:, None].bool().expand(*self.feet_origin.shape[:-1])
        self._set_feet_origin(force_init, False)
        self.last_feet_origin[force_init] = self.feet_origin[force_init]

        height = self.get_feet_height(force_terrain_origin=True)
        self.feet_on_ground[:] = ((height < 1e-3) | self.filtered_feet_contacts)
        self._set_feet_origin(self.feet_on_ground & ~force_init)

        if self.trace_gait_phases:
            self.feet_new_step[:] = self.feet_on_ground & ((self.feet_origin[..., 0] - self.last_feet_origin[..., 0]) > 1)
            self.stance_start_time[self.feet_new_step] = self.common_step_counter
            self.feet_leaving_ground[:] = ~self.feet_on_ground & (self.feet_origin[..., 0] == (self.common_step_counter - 1))
        
    def _set_feet_origin(self, which, assume_on_ground=True): 
        self.last_feet_origin[which] = self.feet_origin[which]
        # Note: the order of the indexes does matter because 'which' is a bool tensor
        self.feet_origin[..., 1:][which] = self.body_state[:, self.feet_indices, :3][which]
        self.feet_origin[..., 0][which] = self.common_step_counter
        if not assume_on_ground:
            self.feet_origin[..., 3][which] = self.get_terrain_height(self.body_state[:, self.feet_indices, :2])[which] + self.cfg.asset.feet_offset

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        if self.cfg.commands.joystick:
            self.time_out_buf[self.ref_env] = self.gamepad_client.startButton.value
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        if self.cfg.commands.curriculum and self.common_step_counter % self.cfg_ppo.runner.num_steps_per_env == 0:
            self.update_command_curriculum()
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
       
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            for command in self.commands_curriculum:
                self.extras["episode"]["min_" + command] = self.command_ranges[command][0]
                self.extras["episode"]["max_" + command] = self.command_ranges[command][1]
        if self.cfg.rewards.curriculum:            
            self.extras["episode"]["default_scale_neg_rewards"] = self.default_reward_curriculum.factor()
            for i, cur in enumerate(self.reward_curriculums):
                if cur is not None:
                    self.extras["episode"]["scale_reward_" + self.reward_names[i]] = cur.factor()
            if self.termination_reward_curriculum is not None:
                self.extras["episode"]["scale_reward_termination"] = self.termination_reward_curriculum.factor()

        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf


    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        if self.cfg.rewards.curriculum:
            default_cur_factor = self.default_reward_curriculum.factor()

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]

            if self.reward_curriculums[i] is not None:
                cur_factor = self.reward_curriculums[i].factor()
            elif self.cfg.rewards.curriculum and self.reward_scales[name] < 0:
                cur_factor = default_cur_factor
            else:
                cur_factor = 1.

            if cur_factor == 0:
                continue
            #print("\033[91mNAMES:", name)
            #print("\033[92mSCALES:", self.reward_scales[name])
            #print("\033[93mREWARDS:\033[0m", cur_factor * self.reward_functions[i]() * self.reward_scales[name])   
            # print("functions:", self.reward_functions[i]())
            # print("REWARDS:", cur_factor * self.reward_functions[i]() * self.reward_scales[name])
            rew = cur_factor * self.reward_functions[i]() * self.reward_scales[name]           
            
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

        # add termination reward after clipping
        if "termination" in self.reward_scales:
            if self.termination_reward_curriculum is not None:
                cur_factor = self.termination_reward_curriculum.factor()
            elif self.cfg.rewards.curriculum and self.reward_scales["termination"] < 0:
                cur_factor = default_cur_factor
            else:
                cur_factor = 1.
            
            rew = cur_factor * self._reward_termination() * self.reward_scales["termination"]
      
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def get_privileged_observations(self):
        return super().get_privileged_observations()

    def compute_observations(self):
        """ Computes observations
        """
        # not modifying the buffer in-place is intended
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind        
        if self.measure_heights:
            clip = self.cfg.normalization.clip_measurements
            heights = (self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_height_points) 
            heights = torch.clip(heights, -clip, clip)
            if self.cfg.terrain.measure_heights:
                self.obs_buf = torch.cat((self.obs_buf, heights * self.obs_scales.height_measurements), dim=-1)
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1])
        if self.cfg.commands.joystick:
            self.commands[self.ref_env, 0] = -self.gamepad_client.leftJoystickY.value * self.command_ranges["lin_vel_x"][1]
            self.commands[self.ref_env, 1] = -self.gamepad_client.leftJoystickX.value * self.command_ranges["lin_vel_y"][1]
            #self.commands[self.ref_env, 2] = -self.gamepad_client.rightJoystickX.value * self.command_ranges["ang_vel_yaw"][1]
            yaw_input = self.gamepad_client.rightJoystickX.value or self.gamepad_client.yawControl.value
            self.commands[self.ref_env, 2] = -yaw_input * self.command_ranges["ang_vel_yaw"][1]

        if self.measure_heights:
            self.measured_height_points[:] = self._get_heights()
            #print(self.measured_height_points)
            
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    # def _resample_commands(self, env_ids):
    #     """ Randommly select commands of some environments

    #     Args:
    #         env_ids (List[int]): Environments ids for which new commands are needed
    #     """
    #     self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     if self.cfg.commands.heading_command:
    #         self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     else:
    #         self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    #     # set small commands to zero
    #     self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


    def _resample_commands(self, env_ids):
        """ Randomly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        if self.cfg.commands.heading_command:
            # Generate random integers between 0 and 7 for each environment
            random_directions = torch.randint(0, 8, (len(env_ids), 1), device=self.device)
            # Convert these to degrees (0, 45, 90, ..., 315) and then to float
            quantized_headings = (random_directions * 45).float()
            # Assign these headings
            self.commands[env_ids, 3] = quantized_headings.squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)


        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            #self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            self.root_states[env_ids, :2] += torch_rand_float(-0.05, 0.05, (len(env_ids), 2), device=self.device) # xy position within 0.05m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self):
        """ Implements a curriculum of increasing commands
        """
        offset = self.cfg.commands.curriculum.offset
        for command in self.commands_curriculum:
            curriculum = self.commands_curriculum[command]
            length_mean = self.cmd_length_mean[command]
            progress = curriculum.factor()
            self.command_ranges[command] = [length_mean[1] - length_mean[0] * progress - offset,
                                            length_mean[1] + length_mean[0] * progress + offset]
          

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions

        i = 48
        if self.cfg.terrain.measure_heights:
            noise_vec[48:(48+self.num_height_points)] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
            i += self.num_height_points

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        # state: x, y, z, [0:3]
        #       q0, q1, q2, q3, [3:7]
        #       v_x, v_y, v_z, [7:10]
        #       v_w_x, v_w_y, v_w_z [10:13]
        # (q0, q1, q2, q3) is the quaternion representing the orientation of the body

        self.num_obs = 48 # will be increased below if measure_heights is True

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
      
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        
        self.body_state = gymtorch.wrap_tensor(body_state).view(self.num_envs, self.num_bodies, 13)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = { }
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.cmd_lin_vel, self.obs_scales.cmd_lin_vel, self.obs_scales.cmd_ang_vel], device=self.device, requires_grad=False,) # TODO change this
        
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_feet_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.filtered_feet_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
    
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
            
        self.trace_gait_phases = self.cfg.normalization.gait_profile
        self.require_feet_origin = self.cfg.rewards.height_estimation == FEET_ORIGIN or self.trace_gait_phases
        if self.require_feet_origin:
            # foot origin: (t, x, y, z) when/where the foot was on the ground for the last time
            self.feet_origin = torch.zeros(self.num_envs, len(self.feet_indices), 4, dtype=torch.float, device=self.device, requires_grad=False)
            self.last_feet_origin = torch.zeros_like(self.feet_origin)
            self.feet_on_ground = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
            
            if self.trace_gait_phases:
                self.feet_new_step = torch.zeros_like(self.feet_on_ground)
                self.feet_leaving_ground = torch.zeros_like(self.feet_on_ground)
                self.stance_start_time = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        
        self.measure_heights = self.cfg.terrain.measure_heights
        if self.measure_heights:
            self.height_points = self._init_height_points()
            self.measured_height_points = torch.zeros(self.num_envs, self.num_height_points, device=self.device)
        else:
            self.num_height_points = 0
       
        if self.cfg.terrain.measure_heights:
            self.num_obs += self.num_height_points

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.noise_scale_vec = self._get_noise_scale_vec()

        if self.cfg.commands.curriculum:            
            global_cur = Curriculum(self, self.cfg.commands.curriculum)
            self.cmd_length_mean = {}
            self.commands_curriculum = {}
            for command in ["lin_vel_x", "lin_vel_y", "ang_vel_yaw", "heading"]:
                limits = getattr(self.cfg.commands.ranges, command)
                if len(limits) <= 2:
                    self.commands_curriculum[command] = global_cur
                elif limits[2]:
                    self.commands_curriculum[command] = Curriculum(self, limits[2])
                else:
                    continue

                self.cmd_length_mean[command] = [(limits[1] - limits[0]) / 2 - self.cfg.commands.curriculum.offset, (limits[0] + limits[1]) / 2]
                if self.cmd_length_mean[command][0] < 0:
                    del self.commands_curriculum[command]

            self.update_command_curriculum()
           
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        self.reward_curriculums = []
        self.default_reward_curriculum = Curriculum(self, self.cfg.rewards.curriculum)
        self.termination_reward_curriculum = None
        if hasattr(self.cfg.rewards.scales, "termination") and self.cfg.rewards.scales.termination < 0. and \
              hasattr(self.cfg.rewards, "termination_curriculum"):
            self.termination_reward_curriculum = Curriculum(self, self.cfg.rewards.termination_curriculum)

        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue

            if hasattr(self.cfg.rewards, name + "_curriculum"):
                self.reward_curriculums.append(Curriculum(self, getattr(self.cfg.rewards, name + "_curriculum")))
            else:
                self.reward_curriculums.append(None)

            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))
            #print("REWARD NAMES : ", self.reward_names)
            #print("REWARD VALUES : ", self.reward_functions)

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names = body_names
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = self._get_body_indices([self.cfg.asset.foot_name])
        self.shoulder_indices = self._get_body_indices([self.cfg.asset.shoulder_name])
        self.penalised_contact_indices = self._get_body_indices(self.cfg.asset.penalize_contacts_on)
        self.termination_contact_indices = self._get_body_indices(self.cfg.asset.terminate_after_contacts_on)

    def _get_body_indices(self, names):
        full_names = []
        for name in names:
            full_names.extend([s for s in self.body_names if name in s])
    
        indices = torch.zeros(len(full_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(full_names)):
            indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], full_names[i])

        return indices

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.follow_env = self.cfg.viewer.follow_env
        self.ref_env = self.cfg.viewer.ref_env

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    # SELF HEIGHT POINTS VS MEASURED HEIGHT POINTS torch.Size([693, 3]) torch.Size([693])
    def _draw_debug_vis(self):
        """Draws visualizations for debugging and checks for tunnel conditions."""
        height_difference_threshold = torch.tensor(0.14, device=self.device, dtype=torch.float)  # Height difference to consider it a tunnel
        #vertical_height_scaling = torch.tensor(2.0, device=self.device, dtype=torch.float)  # Scaling factor for the side points

        # Number of points per "row" in your conceptual grid layout
        points_per_row = 21

        if not self.disable_heights:

            base_pos = self.root_states[:, :3]  # Assuming this works for all environments
            
            heights = self.measured_height_points  # Assuming this tensor is [num_envs, num_height_points]

            # Generate a tensor of indices for each height point, replicated across all environments
            indices = torch.arange(heights.shape[1], device=self.device).repeat(heights.shape[0], 1)

            # Side points condition: indices modulo points_per_row is less than 7 or greater than 13
            side_condition = (indices % points_per_row < 7) | (indices % points_per_row > 13)

            # Middle points condition: complement of side_condition
            middle_condition = ~side_condition

            # Use side_condition and middle_condition to filter heights directly
            # Note: torch.where can be used for more complex operations, but basic indexing suffices for filtering
            self.side_heights = torch.where(side_condition, heights, torch.tensor(0.0, device=self.device))
            self.middle_heights = torch.where(middle_condition, heights, torch.tensor(0.0, device=self.device))

            # # Scale side heights conditionally
            # scaled_side_heights = torch.where(side_heights < 0.25, side_heights, side_heights * vertical_height_scaling)

            # Calculate average heights
            # Use torch.sum and torch.count_nonzero to compute average only on non-zero entries
            avg_side_height = torch.mean(self.side_heights, dim=1)
            avg_middle_height = torch.mean(self.middle_heights, dim=1)

            # Determine tunnel condition for each environment
            #print("AVG_SIDE_HEIGHT", avg_side_height)
            #print("AVG_MIDDLE_HEIGHT", avg_middle_height)
            abs_diff = torch.abs(avg_side_height - avg_middle_height)
            #print("ABS_DIFF", abs_diff)
            #print("HEIGHT DIFFERENCE THRESHOLD", height_difference_threshold.unsqueeze(0))
            self.tunnel_condition = abs_diff > height_difference_threshold.unsqueeze(0)  # Ensure broadcasting works correctly
            #print(self.tunnel_condition.shape)
            #exit(0)

            self.debug_height_map = False
            if self.debug_height_map:
                self.gym.clear_lines(self.viewer)

                base_pos = self.root_states[self.ref_env, :3]
                #heights = self._get_heights(self.ref_env)
                #print(heights.shape)
                # Assume quat_apply_yaw returns transformed points in the shape [num_points, 3]
                #print("SELF HEIGHT POINTS VS MEASURED HEIGHT POINTS", self.height_points[self.ref_env].shape, self.measured_height_points[self.ref_env].shape)
                height_points = quat_apply_yaw(self.base_quat[self.ref_env].repeat(heights.shape[0], 1), self.height_points[self.ref_env])
                #print("REF ENV", self.ref_env)
                #print(height_points)
                # Calculate the visualization colors based on the conditions
                # Here, we assume colors will be a tensor with shape [num_points, 3] indicating RGB colors
                #colors = torch.zeros_like(height_points)
                #side_condition = (torch.arange(heights.shape[0], device=self.device) % 21 < 7) | (torch.arange(heights.shape[0], device=self.device) % 21 > 13)
                #colors[side_condition, :] = torch.tensor([1.0, 0.0, 0.0], device=self.device)  # Red for sides
                #colors[~side_condition, :] = torch.tensor([0.0, 0.0, 1.0], device=self.device)  # Blue for middle

                # Visualization loop
                for j in range(height_points.shape[0]):
                    base_pos = self.root_states[self.ref_env, :3]
                    x, y = height_points[j, 0] + base_pos[0], height_points[j, 1] + base_pos[1]
                    z = torch.maximum(self.measured_height_points[self.ref_env][j], base_pos[2] - self.cfg.rewards.base_height_target) + 0.05
                    #z = np.maximum(self.measured_height_points[self.ref_env].cpu().numpy()[j], base_pos[2].cpu().numpy() - self.cfg.rewards.base_height_target) + 0.05 # Adding a small offset for visualization
                    color = (1, 0, 0) if (j % 21) < 7 else (0, 1, 0) if (j % 21) > 13 else (1, 0.84, 0) if self.tunnel_condition[self.ref_env] else (0, 0, 1)
                    #color = tuple(colors[j].tolist())  # Convert the color tensor to a list for the API call
                    #color_tensor = torch.tensor(color, dtype=torch.float, device=self.device)
                    #sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=color_tensor)

                    sphere_pose = gymapi.Transform(gymapi.Vec3(x.item(), y.item(), z.item()), r=None)
                    sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=color)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.ref_env], sphere_pose)


                # self.gym.clear_lines(self.viewer)
                # quat_repeated = self.base_quat[self.ref_env].repeat(heights.shape[0], 1)
                # height_points = quat_apply_yaw(quat_repeated, self.height_points[self.ref_env])  # Assuming this returns GPU tensor

                # for j in range(height_points.shape[0]):
                #     x = height_points[j, 0] + base_pos[0]
                #     y = height_points[j, 1] + base_pos[1]
                #     z = torch.maximum(heights[j], base_pos[2] - self.cfg.rewards.base_height_target) + 0.05
                #     color = (1, 0, 0) if (j % 21) < 7 else (0, 1, 0) if (j % 21) > 13 else (0, 0, 1) 
                    
                #     # Assuming gymapi.Transform() can take a tensor directly, if not, convert to CPU numpy here
                #     sphere_pose = gymapi.Transform(gymapi.Vec3(x.item(), y.item(), z.item()), r=None)
                #     sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=color)
                #     gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.ref_env], sphere_pose)


    # def _draw_debug_vis(self):
    #     """Draws visualizations for debugging and checks for tunnel conditions."""
    #     height_difference_threshold = torch.tensor(0.04, device=self.device, dtype=torch.float)  # Height difference to consider it a tunnel
    #     vertical_height_scaling = height_difference_threshold = torch.tensor(10.0, device=self.device, dtype=torch.float)  # Scaling factor for the side points
    #     #side_heights = []  # For red and green points
    #     #middle_heights = []  # For blue points
    #     # Number of points per "row" in your conceptual grid layout
    #     points_per_row = 21
    #     if not self.disable_heights:
            
    #         for i in range(self.num_envs if not self.debug_only_one else 1):

    #             base_pos = self.root_states[self.ref_env, :3] 
                
    #             heights = self.measured_height_points[self.ref_env]

    #             # Generate a tensor of indices for the heights tensor, same shape as heights
    #             indices = torch.arange(heights.shape[0], device='cuda:0')

    #             # Side points condition: indices modulo points_per_row is less than 7 or greater than 13
    #             side_condition = (indices % points_per_row < 7) | (indices % points_per_row > 13)

    #             # Middle points condition: complement of side_condition
    #             middle_condition = ~side_condition

    #             # Apply conditions to heights tensor to get side and middle heights directly
    #             side_heights = heights[side_condition]
    #             middle_heights = heights[middle_condition]

    #             scaled_side_heights = torch.where(heights[side_condition] < 0.15,
    #                               heights[side_condition],
    #                               heights[side_condition] * vertical_height_scaling)

    #             # Assuming side_heights and middle_heights are tensors on the GPU from the previous step
    #             avg_side_height = torch.mean(scaled_side_heights)
    #             avg_middle_height = torch.mean(middle_heights)

    #             # Calculate the absolute difference between the averages
    #             abs_diff = torch.abs(avg_side_height - avg_middle_height)

    #             # Determine if the absolute difference exceeds the height difference threshold
    #             self.tunnel_condition[self.ref_env] = abs_diff > height_difference_threshold
                

                # # Iterate over each point to classify and accumulate heights
                # for j in range(height_points.shape[0]):
                #     if (j % 21) < 7 or (j % 21) > 13:  # Side points, assuming 0-5 and 16-20 are sides
                #         if heights[j] < 0.15:
                #             side_heights.append(heights[j].item())
                #         else:
                #             side_heights.append(heights[j].item() * vertical_height_scaling)
                #     else:  # Middle points, assuming 6-15 are middle
                #         middle_heights.append(heights[j].item())

                # # Convert lists to tensors for easier manipulation and ensure they're on the correct device
                # side_heights = torch.tensor(side_heights, device='cuda:0')
                # middle_heights = torch.tensor(middle_heights, device='cuda:0')

                # # Compute average heights for side and middle
                # avg_side_height = torch.mean(side_heights)
                # avg_middle_height = torch.mean(middle_heights)
                

    # # def _draw_debug_vis(self):
    #     """Draws visualizations for debugging and checks for tunnel conditions."""
    #     if not self.debug_height_map and not self.disable_heights:
    #         middle_stripe_margin = 0.19
    #         height_difference_threshold = 0.18  # Height difference to consider it a tunnel

    #         for i in range(1 if self.debug_only_one else self.num_envs):
    #             pos = self.ref_env if self.debug_only_one else i
    #             base_pos = self.root_states[pos, :3]
    #             height_points = self.measured_height_points[pos]
    #             #height_points = quat_apply_yaw(self.base_quat[pos].repeat(heights.shape[0]), self.height_points[pos]).cpu().numpy()
    #             #height_points = quat_apply_yaw(self.base_quat[pos].repeat(heights.shape[0], 1), self.height_points[pos])
        
    #             front_left_heights = []
    #             front_right_heights = []
    #             rear_right_heights = []
    #             rear_left_heights = []
    #             other_heights = []

    #             for j in range(height_points.shape[0]):
    #                 #x, y = height_points[j, 0] + base_pos[0], height_points[j, 1] + base_pos[1]
    #                 x_rot, y_rot, z = height_points[j]
    #                 # Adjust for visualization offset
    #                 z += 0.05
    #                 #For double rotation:
    #                 #local_point = torch.tensor([height_points[j, 0], height_points[j, 1], 0], dtype=torch.float32, device=self.device)  # Adding a z-component of 0
    #                 # Rotate the point by the robot's orientation quaternion
    #                 #rotated_point = quat_apply_yaw_inverse(self.base_quat[pos], local_point)  # Assuming quat_apply_yaw correctly applies the rotation
    #                 # Convert the rotated point back to numpy if necessary and add the base position for the final world coordinates
    #                 #x_rot, y_rot = rotated_point[:2] + base_pos[:2]
    #                 #z = heights[j] + 0.05  # Adding a small offset to the height for visualization
    #                 #z = np.maximum(heights[j], base_pos[2] - self.cfg.rewards.base_height_target) + 0.05
                    
    #                 is_front_left = (x_rot - base_pos[0] > 0) & (x_rot - base_pos[0] < 1.) & (y_rot - base_pos[1] > -1.) & (y_rot - base_pos[1] < -middle_stripe_margin)
    #                 is_front_right = (x_rot - base_pos[0] > 0) & (x_rot - base_pos[0] < 1.) & (y_rot - base_pos[1] > middle_stripe_margin) & (y_rot - base_pos[1] < 1.)
    #                 is_rear_left = (x_rot - base_pos[0] < 0) & (x_rot - base_pos[0] > -1.) & (y_rot - base_pos[1] > -1.) & (y_rot - base_pos[1] < -middle_stripe_margin)
    #                 is_rear_right = (x_rot - base_pos[0] < 0) & (x_rot - base_pos[0] > -1.) & (y_rot - base_pos[1] > middle_stripe_margin) & (y_rot - base_pos[1] < 1.)

    #                 if is_front_left or is_front_right:
    #                     front_left_heights.append(z) if is_front_left else front_right_heights.append(z)
    #                 elif is_rear_left or is_rear_right:
    #                     rear_left_heights.append(z) if is_rear_left else rear_right_heights.append(z)
    #                 else:
    #                     other_heights.append(z)
                
    #             # Combine all corner heights and compare to other heights
    #             corner_heights = front_left_heights + front_right_heights + rear_left_heights + rear_right_heights
    #             if corner_heights:  # Ensure there are corner heights to consider
    #                 avg_corner_height = sum(corner_heights) / len(corner_heights)
    #                 avg_other_height = sum(other_heights) / len(other_heights) if other_heights else 0

    #                 # Determine tunnel condition based on height difference
    #                 self.tunnel_condition[pos] = (avg_corner_height - avg_other_height) > height_difference_threshold
    #             else:
    #                 self.tunnel_condition[pos] = False
            
    #         return
    #     elif not self.disable_heights:
    #         #print("Drawing debug visualizations")
    #         self.gym.clear_lines(self.viewer)
    #         # Define the width of the middle stripe to leave unhighlighted
    #         middle_stripe_margin = 0.19
    #         height_difference_threshold = 0.18  # Height difference to consider it a tunnel

    #         for i in range(1 if self.debug_only_one else self.num_envs):
    #             pos = self.ref_env if self.debug_only_one else i
    #             #base_pos = self.root_states[pos, :3].cpu().numpy()
    #             #heights = self.measured_height_points[pos].cpu().numpy()
    #             base_pos = self.root_states[pos, :3]
    #             height_points = self.measured_height_points[pos]
    #             print("Height points: ", height_points)
    #             #height_points = quat_apply_yaw(self.base_quat[pos].repeat(heights.shape[0]), self.height_points[pos]).cpu().numpy()
    #             #height_points = quat_apply_yaw(self.base_quat[pos].repeat(heights.shape[0], 1), self.height_points[pos])
        
    #             front_left_heights = []
    #             front_right_heights = []
    #             rear_right_heights = []
    #             rear_left_heights = []
    #             other_heights = []

    #             for j in range(height_points.shape[0]):
    #                 #x, y = height_points[j, 0] + base_pos[0], height_points[j, 1] + base_pos[1]
    #                 x_rot = height_points[j, 0] + base_pos[0]
    #                 y_rot = height_points[j, 1] + base_pos[1]

    #                 # Adjust for visualization offset
    #                 z += 0.05
    #                 #For double rotation:
    #                 #local_point = torch.tensor([height_points[j, 0], height_points[j, 1], 0], dtype=torch.float32, device=self.device)  # Adding a z-component of 0
    #                 # Rotate the point by the robot's orientation quaternion
    #                 #rotated_point = quat_apply_yaw_inverse(self.base_quat[pos], local_point)  # Assuming quat_apply_yaw correctly applies the rotation
    #                 # Convert the rotated point back to numpy if necessary and add the base position for the final world coordinates
    #                 #x_rot, y_rot = rotated_point[:2] + base_pos[:2]
                    
    #                 #z = heights[j] + 0.05  # Adding a small offset to the height for visualization
    #                 #z = np.maximum(heights[j], base_pos[2] - self.cfg.rewards.base_height_target) + 0.05
                    
    #                 is_front_left = (x_rot - base_pos[0] > 0) & (x_rot - base_pos[0] < 1.) & (y_rot - base_pos[1] > -1.) & (y_rot - base_pos[1] < -middle_stripe_margin)
    #                 is_front_right = (x_rot - base_pos[0] > 0) & (x_rot - base_pos[0] < 1.) & (y_rot - base_pos[1] > middle_stripe_margin) & (y_rot - base_pos[1] < 1.)
    #                 is_rear_left = (x_rot - base_pos[0] < 0) & (x_rot - base_pos[0] > -1.) & (y_rot - base_pos[1] > -1.) & (y_rot - base_pos[1] < -middle_stripe_margin)
    #                 is_rear_right = (x_rot - base_pos[0] < 0) & (x_rot - base_pos[0] > -1.) & (y_rot - base_pos[1] > middle_stripe_margin) & (y_rot - base_pos[1] < 1.)

    #                 #Determine the color based on the condition
    #                 if self.tunnel_condition[pos]:
    #                     # If tunnel condition is true, non-area points change to amber/yellowish
    #                     color = (1, 0.84, 0)  # Amber/Yellowish
    #                 else:
    #                     color = (0, 0, 1)  # Default blue for points outside specified areas
                        
    #                 if is_front_left or is_rear_left:
    #                     color = (0, 1, 0)  # Green for both front and rear left areas
    #                 elif is_front_right or is_rear_right:
    #                     color = (1, 0, 0)  # Red for both front and rear right areas
    #                     #pass

    #                 if is_front_left or is_front_right:
    #                     front_left_heights.append(z) if is_front_left else front_right_heights.append(z)
    #                 elif is_rear_left or is_rear_right:
    #                     rear_left_heights.append(z) if is_rear_left else rear_right_heights.append(z)
    #                 else:
    #                     other_heights.append(z)

    #                 sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z))
    #                 #sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z))
    #                 sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=color)

    #                 gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[pos], sphere_pose)

    #             # Combine all corner heights and compare to other heights
    #             corner_heights = front_left_heights + front_right_heights + rear_left_heights + rear_right_heights
    #             if corner_heights:  # Ensure there are corner heights to consider
    #                 avg_corner_height = sum(corner_heights) / len(corner_heights)
    #                 avg_other_height = sum(other_heights) / len(other_heights) if other_heights else 0

    #                 # Determine tunnel condition based on height difference
    #                 self.tunnel_condition[pos] = (avg_corner_height - avg_other_height) > height_difference_threshold
    #             else:
    #                 self.tunnel_condition[pos] = False
    #     #print("Tunnel condition: ", self.tunnel_condition[self.ref_env])

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = self.to_world_coords(self.height_points, env_ids)

        return self.get_terrain_height(points, in_place=True)
    
    def to_world_coords(self, points, env_ids=None):
        """Returns, for each environment, the points exprimed in world's coordinates.
        
        Args:
            points (torch.Tensor of shape [E, N, 3]): The tensor containing the x,y coords
            of the points, relative to each robot's base.

            env_ids (List[int], optional) Subset of environments for which to convert the points.
            If env_ids is None, E must be equal to self.num_envs
            
        Returns:
            torch.tensor, shape [E, N, 3]"""
        
        if not env_ids:
            env_ids = slice(None)

        return quat_apply_yaw(self.base_quat[env_ids].repeat(1, points.shape[1]), points) + (self.root_states[env_ids, :3]).unsqueeze(1)

    def get_terrain_height(self, points, in_place=False):
        """Get (sampled) terrain heights at specified points.
        
        Args:
            points [torch.Tensor of shape [..., 2]]: Tensor containing the X and Y coords (in world frame) of
            the points
        
        Returns:
            torch.tensor, shape [...]"""
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(points.shape[:-1], device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")
        
        if not in_place: points = points.clone()

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[..., 0]
        py = points[..., 1]
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights * self.terrain.cfg.vertical_scale
    
    # def check_tunnel_condition(self):
    #     # Assuming self.height_points is already in world coordinates
    #     # and self.get_terrain_height(self.height_points) has been called
    #     # to populate a tensor with the heights at each point.
        
    #     heights = self._get_heights()  # This should return the heights at all self.height_points

    #     # Define the square areas in front left and front right relative to the robot
    #     # Assuming the robot's forward direction aligns with the positive x-axis
    #     front_left_area = (self.height_points[:, :, 0] > 0) & (self.height_points[:, :, 0] < 0.1) & \
    #                     (self.height_points[:, :, 1] > -0.15) & (self.height_points[:, :, 1] < 0)
    #     front_right_area = (self.height_points[:, :, 0] > 0) & (self.height_points[:, :, 0] < 0.1) & \
    #                     (self.height_points[:, :, 1] > 0) & (self.height_points[:, :, 1] < 0.15)
        
    #     # Initialise tunnel_condition as False for all environments
    #     self.tunnel_condition = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    #     # Check if the heights in these areas exceed the threshold (e.g., 0.5m)
    #     # relative to the floor. This example assumes you have a way to determine
    #     # the "floor" height or use 0 as a reference.
    #     floor_height = 0  # This could be dynamic based on your environment
    #     height_threshold = 0.5  # Heights must be greater than 0.5m to consider it a tunnel
        
    #     # For each environment, check if the condition is met in both areas
    #     for env_id in range(self.num_envs):
    #         fl_heights = heights[env_id, front_left_area[env_id]]
    #         fr_heights = heights[env_id, front_right_area[env_id]]
            
    #         # Check if any point in the specified areas exceeds the height threshold
    #         if torch.any(fl_heights - floor_height > height_threshold) and \
    #         torch.any(fr_heights - floor_height > height_threshold):
    #             self.tunnel_condition[env_id] = True

    
    def get_base_height(self):
        roots = self.root_states.unsqueeze(1)
        if len(self.shoulder_indices) > 0:
            roots = torch.cat((roots, self.body_state[:, self.shoulder_indices]), dim=1)

        if self.cfg.rewards.height_estimation == AVERAGE_MEASUREMENT:
            origin = self.get_terrain_height(roots[:, :, :2], in_place=True)
        elif self.cfg.rewards.height_estimation == FEET_ORIGIN:
            origin =  torch.mean(self.feet_origin[..., 3], dim=1).unsqueeze(1)
        else:
            raise ValueError("Unsupported estimation mode: " + self.cfg.rewards.height_estimation)

        return torch.mean(roots[:, :, 2] - origin, dim=1)
    
    def get_feet_height(self, force_terrain_origin=False):
        if force_terrain_origin or self.cfg.rewards.height_estimation == AVERAGE_MEASUREMENT:
            origin = self.get_terrain_height(self.body_state[:, self.feet_indices, :2]) + self.cfg.asset.feet_offset
        elif self.cfg.rewards.height_estimation == FEET_ORIGIN:
            origin = self.feet_origin[..., 3]
        else:
            raise ValueError("Unsupported estimation mode: " + self.cfg.rewards.height_estimation)
        return self.body_state[:, self.feet_indices, 2] - origin
    
    def quit(self):
        if self.cfg.commands.joystick:
            self.gamepad_client.stop()

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.get_base_height()
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        #print("Tunnel condition   : ", self.tunnel_condition[self.ref_env])
        if self.tunnel_condition[self.ref_env] == True:
            # Scale down the collision penalty if the tunnel condition is met
            scaling_factor = 0.02
            # print(torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1))
            # print(torch.sum(1.*(torch.norm(scaling_factor * self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1))
            # import sys
            # sys.exit()
            return torch.sum(1.*(torch.norm(scaling_factor * self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
        else:
            # Penalize collisions on selected bodies normally
            return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes      
        first_contact = (self.feet_air_time > 0.) * self.filtered_feet_contacts
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~self.filtered_feet_contacts
        return rew_airTime
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

class Curriculum:
    def __init__(self, robot: LeggedRobot, cfg: CurriculumConfig):
        self.robot = robot
        self.enabled = cfg.enabled
        self.duration = cfg.duration
        self.interpolation = cfg.interpolation
        self.delay = cfg.delay

    def factor(self) -> float:
        if not self.enabled: return 1.
        
        iteration = self.robot.common_step_counter // self.robot.cfg_ppo.runner.num_steps_per_env - self.delay
        if iteration < 0: return 0.

        return min(1., (iteration / self.duration) ** self.interpolation)
    
    # def _draw_debug_vis(self):
    #     """ Draws visualizations for debugging (slows down simulation a lot).
    #         Default behaviour: draws height measurement points
    #     """
    #     if not self.debug_height_map:
    #         return   

    #     # if self.cfg.contact_classification.enabled:               
    #     #     if self.cfg.contact_classification.normalize:
    #     #         colors = self.obs_quality_buf #already normalized
    #     #     else:
    #     #         colors = (self.obs_quality_buf - torch.min(self.contacts_quality[..., 0])) / \
    #     #                 (torch.max(self.contacts_quality[..., 0]) - torch.min(self.contacts_quality[..., 0]))
            
    #     self.gym.clear_lines(self.viewer)
        
    #     # Define your front left and front right areas relative to the base
    #     # Example definitions, adjust according to your robot's coordinate system
    #     front_left_def = lambda x, y: (x > 0) & (x < 0.1) & (y > -0.15) & (y < 0)
    #     front_right_def = lambda x, y: (x > 0) & (x < 0.1) & (y > 0) & (y < 0.15)
    
    #     for i in range(1 if self.debug_only_one else self.num_envs):
    #         pos = self.ref_env if self.debug_only_one else i
 
    #         if self.debug_height_map:
    #             base_pos = (self.root_states[pos, :3]).cpu().numpy()              
    #             heights = self.measured_height_points[pos].cpu().numpy()
    #             height_points = quat_apply_yaw(self.base_quat[pos].repeat(heights.shape[0]), self.height_points[pos]).cpu().numpy()
    #             for j in range(heights.shape[0]):
    #                 x = height_points[j, 0] + base_pos[0]
    #                 y = height_points[j, 1] + base_pos[1]
    #                 z = np.maximum(heights[j], base_pos[2] - self.cfg.rewards.base_height_target) + 0.05

    #                 # Determine if this point is in the front left or front right area
    #                 is_front_left = front_left_def(x - base_pos[0], y - base_pos[1])
    #                 is_front_right = front_right_def(x - base_pos[0], y - base_pos[1])

    #                 # Choose color based on the area
    #                 if is_front_left:
    #                     color = (0, 1, 0)  # Green for front left
    #                 elif is_front_right:
    #                     color = (1, 0, 0)  # Red for front right
    #                 else:
    #                     color = (0, 0, 1)  # Default to blue for other points

    #                 sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
    #                 sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=color[:3])

    #                 gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[pos], sphere_pose) 

    # def _draw_debug_vis(self):
    #     """Draws visualizations for debugging."""
    #     if not self.debug_height_map:
    #         return

    #     self.gym.clear_lines(self.viewer)

    #     # Define the width of the middle stripe to leave unhighlighted
    #     middle_stripe_margin = 0.2  # Adjust the width of the middle stripe as needed

    #     for i in range(1 if self.debug_only_one else self.num_envs):
    #         pos = self.ref_env if self.debug_only_one else i
    #         base_pos = self.root_states[pos, :3].cpu().numpy()
    #         heights = self.measured_height_points[pos].cpu().numpy()
    #         height_points = quat_apply_yaw(self.base_quat[pos].repeat(heights.shape[0]), self.height_points[pos]).cpu().numpy()

    #         for j in range(heights.shape[0]):
    #             x, y = height_points[j, 0] + base_pos[0], height_points[j, 1] + base_pos[1]
    #             z = np.maximum(heights[j], base_pos[2] - self.cfg.rewards.base_height_target) + 0.05

    #             # Check if point is within the desired front corner areas, excluding the middle stripe
    #             is_front_left = (x - base_pos[0] > 0) & (x - base_pos[0] < 1.) & \
    #                             (y - base_pos[1] > -1.) & (y - base_pos[1] < -middle_stripe_margin)
    #             is_front_right = (x - base_pos[0] > 0) & (x - base_pos[0] < 1.) & \
    #                             (y - base_pos[1] > middle_stripe_margin) & (y - base_pos[1] < 1.)

    #             # Choose color based on the area, excluding the middle stripe
    #             color = (0, 1, 0) if is_front_left else (1, 0, 0) if is_front_right else (0, 0, 1)

    #             sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z))
    #             sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=color)

    #             gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[pos], sphere_pose)

    #         # Additional visualization logic for gait phases, if applicable
    #         if self.trace_gait_phases:
    #             for j in range(len(self.feet_indices)):
    #                 if self.feet_new_step[pos, j]:
    #                     x = self.feet_origin[pos, j, 1]
    #                     y = self.feet_origin[pos, j, 2]
    #                     z = self.feet_origin[pos, j, 3]
    #                     cube_pos = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)

    #                     c = j / len(self.feet_indices)
    #                     cube_geom = gymutil.WireframeBoxGeometry(0.02, 0.02, 0.02, color=(0, 1, c))

    #                     gymutil.draw_lines(cube_geom, self.gym, self.viewer, self.envs[pos], cube_pos)
        