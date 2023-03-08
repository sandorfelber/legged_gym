#from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
#
#class Solo12Cfg( LeggedRobotCfg ):
#        
#    class env( LeggedRobotCfg.env):
#        num_envs = 1
#
#    class init_state( LeggedRobotCfg.init_state ):
#
#    ### Initial state parameters ###
#        pos = [0.0, 0.0, 0.23]  # x,y,z [m]
#        default_joint_angles = {  # = target angles [rad] when action = 0.0
#            "FL_hip_joint": 0.01,
#            "RL_hip_joint": 0.01,
#            "FR_hip_joint": -0.01,
#            "RR_hip_joint": -0.01,
#            "FL_thigh_joint": 0.8,
#            "RL_thigh_joint": 0.8,
#            "FR_thigh_joint": 0.8,
#            "RR_thigh_joint": 0.8,
#            "FL_calf_joint": -1.6,
#            "RL_calf_joint": -1.6,
#            "FR_calf_joint": -1.6,
#            "RR_calf_joint": -1.6,
#        }
#
#    class control( LeggedRobotCfg.control ):
#        control_type = "P"
#        stiffness = {"joint": 3.0}  # [N*m/rad]
#        damping = {"joint": 0.3}  # [N*m*s/rad]
#        # action scale: target angle = actionScale * action + defaultAngle
#        action_scale = 0.25
#        hip_scale_reduction = 1.0
#        # decimation: Number of control action updates @ sim DT per policy DT
#        decimation = 4
#
#
#    class asset( LeggedRobotCfg.asset ):
#            file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/solo12/solo12_isaac.urdf'
#            foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
#            penalize_contacts_on = []
#            terminate_after_contacts_on = []
#            disable_gravity = False
#            # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
#            collapse_fixed_joints = True
#            fix_base_link = False  # fixe the base of the robot
#            default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
#            self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
#            # replace collision cylinders with capsules, leads to faster/more stable simulation
#            replace_cylinder_with_capsule = True
#            flip_visual_attachments = (
#                  False  # Some .obj meshes must be flipped from y-up to z-up
#            )
#
#            density = 0.001
#            angular_damping = 0.0
#            linear_damping = 0.0
#            max_angular_velocity = 1000.0
#            max_linear_velocity = 1000.0
#            armature = 0.0
#            thickness = 0.01
#
#
#    class sim( LeggedRobotCfg.sim ):
#            dt = 0.005
#            substeps = 1
#            gravity = [0.0, 0.0, -9.81]  # [m/s^2]
#            up_axis = 1  # 0 is y, 1 is z
#
#    #        use_gpu_pipeline = True
#
#            class physx( LeggedRobotCfg.sim.physx ):
#                num_threads = 10
#                solver_type = 1  # 0: pgs, 1: tgs
#                num_position_iterations = 4
#                num_velocity_iterations = 0
#                contact_offset = 0.01  # [m]
#                rest_offset = 0.0  # [m]
#                bounce_threshold_velocity = 0.5  # 0.5 [m/s]
#                max_depenetration_velocity = 1.0
#                max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
#                default_buffer_size_multiplier = 5
#                contact_collection = (
#                    2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
#                )
#
#class Solo12CfgPPO( LeggedRobotCfgPPO ):
#    
#    class runner( LeggedRobotCfgPPO.runner ):
#        run_name = ''
#        experiment_name = 'solo12'


from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# NAMES OF JOINTS IN URDF FILE
# front left leg
FL_HAA = 'FL_hip_joint' # hip AA (abduction/adduction)
FL_HFE = 'FL_thigh_joint' # hip FE (flexion/extension)
FL_KFE = 'FL_calf_joint' # knee FE

FR_HAA = 'FR_hip_joint' # front right
FR_HFE = 'FR_thigh_joint'
FR_KFE = 'FR_calf_joint'

HL_HAA = 'RL_hip_joint'  # hind (back) left
HL_HFE = 'RL_thigh_joint'
HL_KFE = 'RL_calf_joint'

HR_HAA = 'RR_hip_joint'  # hind (back) right
HR_HFE = 'RR_thigh_joint'
HR_KFE = 'RR_calf_joint'

class Solo12Cfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_actions = 12

    class init_state( LeggedRobotCfg.init_state ):
        default_joint_angles = { # = target angles [rad] when action = 0.0
            
            FL_HAA: 0.,
            FL_HFE: 0.9,
            FL_KFE: -1.64,

            FR_HAA: 0.,
            FR_HFE: 0.9,
            FR_KFE: -1.64,

            HL_HAA: 0.,
            HL_HFE: 0.9,
            HL_KFE: -1.64,

            HR_HAA: 0.,
            HR_HFE: 0.9,
            HR_KFE: -1.64

        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters: (paper page 5)
        # "joint" will apply to all DoF since all joint names contain "joint" 
        stiffness = { "joint": 3. } # {'HAA': 3., 'HFE': 3., 'KFE': 3.}  # K_p [N*m/rad]
        damping = { "joint": .2 }  # {'HAA': .2, 'HFE': .2, 'KFE': .2}     # K_d [N*m*s/rad]

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
        pass
    #    num_commands = 3 # lin_vel_x, lin_vel_y, ang_vel_yaw
    #    heading_command = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/solo12/solo12_isaac.urdf'
        name = "solo"
        foot_name = 'calf'
        collapse_fixed_joints = False # otherwise feet are collapsed, and robot.feet_indices is not populated
      #  default_dof_drive_mode = 1 # pos tgt
        flip_visual_attachments = False # fix visual problem with meshes
        terminate_after_contacts_on = ["base", "thigh"]
        self_collisions = 1 

class Solo12CfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'solo12'