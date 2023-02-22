
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Solo12Cfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_actions = 12

    class init_state( LeggedRobotCfg.init_state ):
        default_joint_angles = { # = target angles [rad] when action = 0.0
            
            # front left leg
            'FL_HAA': 0., # hip AA (abduction/adduction)
            'FL_HFE': 0.9, # hip FE (flexion/extension)
            'FL_KFE': -1.64, # knee FE

            'FR_HAA': 0., # front right
            'FR_HFE': 0.9,
            'FR_KFE': -1.64,

            'HL_HAA': 0., # hind (back) left
            'HL_HFE': 0.9,
            'HL_KFE': -1.64,

            'HR_HAA': 0., # hind (back) right
            'HR_HFE': 0.9,
            'HR_KFE': -1.64

        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters: (paper page 5)
        stiffness = {'HAA': 3., 'HFE': 3., 'KFE': 3.}  # K_p [N*m/rad]
        damping = {'HAA': .2, 'HFE': .2, 'KFE': .2}     # K_d [N*m*s/rad]

        action_scale = 0.3 # paper (page 6)
        feet_height_target = 0.06 # p_z^max [m]

    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False
        class scales:
            velocity = 6. # c_vel
            foot_clearance = -20. # -c_clear
            foot_slip = -0.07 # -c_slip
            roll_pitch = -3. # -c_orn
            vel_z = -1.2 # -c_vz
            joint_pose = -0.5 # -c_q
            power_loss = -2.0 # -c_E
            smoothness_1 = -2.5 # -c_a1
            smoothness_2 = -1.5 # -c_a2
            
    class commands( LeggedRobotCfg.commands ):
        num_commands = 3 # lin_vel_x, lin_vel_y, ang_vel_yaw
        heading_command = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/solo12/urdf/solo.urdf'
        name = "solo"
        foot_name = 'FOOT'
        collapse_fixed_joints = False # otherwise feet are collapsed, and robot.feet_indices is not populated

class Solo12CfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'solo12'


  