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

AVERAGE_MEASUREMENT = 0
FEET_ORIGIN = 1

import numpy as np
from .base_config import BaseConfig

class CurriculumConfig:
    """Configuration a of curriculum (except for terrain curriculum)
    How the curriculum is really used depends on the context, but it should provide a gradually varying
    quantity such that:

            - actual_value = target_value * factor
            - factor = clip(0, 1, ((current_iteration - delay) / duration) ** interpolation)"""
    
    enabled = False
    duration = 0
    delay = 0
    interpolation = 1.

    def __init__(self, **kwargs):
        for param in ["enabled", "duration", "delay", "interpolation"]:
            if param in kwargs:
                setattr(self, param, kwargs[param])

    def __bool__(self):
        return self.enabled

class LeggedRobotCfg(BaseConfig):

    def __init__(self):
        super().__init__()
        self.enforce()

    training = True
    class env:
        num_envs = 4096
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        steps_height_scale = 1.
        horizontal_difficulty_scale = 1. # may require decreasing horizontal_scale as well
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        #measured_points_x = np.arange(-1.6, 1.61, 0.2).tolist() # 1mx1.6m rectangle (without center line)
        #measured_points_y = np.arange(-1., 1.01, 0.2).tolist()
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.3, 0.2, 0.1, 0.1, 0.1] # overwritten in config file
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        class curriculum(CurriculumConfig):
            # curriculum for command ranges
            # NOTE: by default, this setting applies to all the commands
            # but it is possible to apply a specific curriculum to each command:
            # ranges.<commmand> = [min, max, False] to disable the curriculum
            # ranges.<commands> = [min, max, CurriculumConfig] to use a specific config
            offset = 0

        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error

        class ranges: 
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-0.75, 0.75]    # min max [rad/s]
            heading = [-3.14, 3.14]

        _enforce_joystick = True # do not load from yaml
        joystick = False # overrriden in play.py, should not be used in training mode

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        shoulder_name = "None" # name of the shoulder/hip bodies, used to estimate the base height
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01
        feet_offset = 0. # offset between the actual bottom of feet and the position of the foot bodies [m]

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.0001 #-0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            step_forecast = -0

        class curriculum(CurriculumConfig):
            # default curriculum for *negative* rewards
            pass

        # you can use specific curriculum (even for positive rewards) this way:
        #class <reward_name>_curriculum(CurriculumConfig):
        #   your specific config here...

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized
        height_estimation = AVERAGE_MEASUREMENT # should use FEET_ORIGIN if the terrain has gaps

    #deprecated
    class planning:
        pass

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            cmd_lin_vel = 2.0
            ang_vel = 0.25
            cmd_ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            contacts_quality = 1.
            feet_on_ground = 10
        clip_observations = 100.
        clip_actions = 100.
        clip_measurements = 1.
        
        _enforce_gait_profile = True # do not load from yaml
        gait_profile = None # set in play.py / do not override

    #deprecated (inefficient, aborted)
    class contact_classification:      
        enabled = False
        # NOTE: contacts qualities are saved in/loaded from checkpoints, along with the RL model
        frozen = False # do not learn qualities - should not be True at the beginning of a new training
        only_positive_qualities = True # clip negative qualities to 0
        normalize = True # force observations to be in [0,1]
        max_convergence_factor = None

        class learn_curriculum( CurriculumConfig ):
            pass
   
    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.4
            classification = 0.05

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]
        follow_env = False

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    def eval(self):
        self.training = False
        self.env.num_envs = min(self.env.num_envs, 50)
        self.commands.curriculum.enabled = False
        self.rewards.curriculum.enabled = False
        self.terrain.num_rows = 5
        self.terrain.num_cols = 5
        self.terrain.curriculum = False
        self.noise.add_noise = False
        self.domain_rand.randomize_friction = False
        self.domain_rand.push_robots = False
        self.contact_classification.frozen = True

    def update(self, config_str):
        training = self.training
        super().update(config_str)
        if not training:
            self.eval()
        self.enforce()

    def enforce(self):
        """Settings to apply after initialization, or to forcefully reapply after importing a YAML file"""
        pass

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnLeggedPolicyRunner'  # 'OnPolicyRunner' 
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        estimator_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0 
        use_clipped_value_loss = True 
        clip_param = 0.2 # PPO clip parameter
        entropy_coef = 0.01 
        num_learning_epochs = 5 # number of PPO epochs
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-4 #1.e-3 
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99 # discount factor 
        lam = 0.95 # GAE lambda
        desired_kl = 0.01 # desired kl divergence
        max_grad_norm = 1.

        train_step_estimator = False
        
    class runner:
        policy_class_name = '__import__("legged_gym").utils.rl.StepEstimatorActorCritic'# 'ActorCritic'
        algorithm_class_name = '__import__("legged_gym").utils.rl.StepEstimatorPPO'# 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 2500 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt