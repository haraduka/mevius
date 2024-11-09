
CAN_ID = [
        10, 11, 12,
        7, 8, 9,
        4, 5, 6,
        1, 2, 3,
        ]

JOINT_NAME = [
        'BL_collar', 'BL_hip', 'BL_knee',
        'BR_collar', 'BR_hip', 'BR_knee',
        'FL_collar', 'FL_hip', 'FL_knee',
        'FR_collar', 'FR_hip', 'FR_knee',
        ]

MOTOR_DIR = [
         1, -1, -1,
         1,  1,  1,
        -1, -1, -1,
        -1,  1,  1,
        ]

STANDBY_ANGLE = [
         0.2222, 1.2710, -2.8754,
        -0.2222, 1.2710, -2.8754,
         0.2398, 1.3063, -2.8754,
        -0.2398, 1.3063, -2.8754,
        ]

DEFAULT_ANGLE = [
         0.1, 1.0, -1.4, # BL
        -0.1, 1.0, -1.4, # BR
         0.1, 0.8, -1.4, # FL
        -0.1, 0.8, -1.4, # FR
        ]

DEBUG_ANGLE = [
         0.4, 0.8, -1.4,
        -0.4, 0.8, -1.4,
         0.2, 0.8, -1.4,
        -0.2, 0.8, -1.4,
        ]

CONTROL_HZ = 50

CAN_HZ = 50

class commands:
    heading_command = False # if true: compute ang vel command from heading error
    class ranges:
        lin_vel_x = [-1.0, 1.0] # min max [m/s]
        lin_vel_y = [-1.0, 1.0] # min max [m/s]
        ang_vel_yaw = [-1.0, 1.0] # min max [rad/s]
        heading = [-3.14, 3.14]

class control:
    stiffness = {'collar': 50.0, 'hip': 50.0, 'knee': 30.}  # [N*m/rad]
    damping = {'collar': 2.0, 'hip': 2.0, 'knee': 0.2}  # [N*m*s/rad]

    # action scale: target angle = actionScale * action + defaultAngle
    action_scale = 0.5
    action_clipping = 20
    decimation = 4
    dt = 0.005

