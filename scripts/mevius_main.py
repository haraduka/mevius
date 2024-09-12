import argparse
import os
import sys
import select
import time
import threading
import numpy as np

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Imu, Joy
from nav_msgs.msg import Odometry
from tmotor_lib import CanMotorController
from mevius.msg import MeviusLog
import mevius_utils

# TODO add last time that the sensor is received and check if it is too old
# TODO add terminal display of thermometer, etc.

np.set_printoptions(precision=3)

################ constant ##################


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
        0.15, 1.325, -2.8731,
        -0.15, 1.325, -2.8731,
        0.15, 1.325, -2.8731,
        -0.15, 1.325, -2.8731,
        ]
STANDUP_ANGLE = mevius_utils.init_state.default_joint_angles[:]
DEBUG_ANGLE = [
        0.4, 0.8, -1.4,
        -0.4, 0.8, -1.4,
        0.2, 0.8, -1.4,
        -0.2, 0.8, -1.4,
        ]
CONTROL_HZ = 50
CAN_HZ = 200

################ class ##################

class RobotState:
    def __init__(self, n_motor=12):
        self.angle = [0.0] * n_motor
        self.velocity = [0.0] * n_motor
        self.current = [0.0] * n_motor
        self.temperature = [0.0] * n_motor
        self.lock = threading.Lock()

class PeripheralState:
    def __init__(self):
        self.body_vel = [0.0] * 3
        self.body_quat = [0.0] * 4
        self.body_gyro = [0.0] * 3
        self.body_acc = [0.0] * 3
        self.spacenav = [0.0] * 8
        self.lock = threading.Lock()

class RobotCommand:
    def __init__(self, n_motor=12):
        self.angle = [0.0] * n_motor
        self.velocity = [0.0] * n_motor
        # 1.5 can be calculated from the difference of the actual and simulated torque
        self.kp = []
        self.kd = []
        self.coef = 1.2
        for name in JOINT_NAME:
            for key in mevius_utils.control.stiffness.keys():
                if key in name:
                    self.kp.append(mevius_utils.control.stiffness[key]*self.coef)
                    self.kd.append(mevius_utils.control.damping[key]*self.coef)
        assert len(self.kp) == n_motor
        assert len(self.kd) == n_motor
        self.torque = [0.0] * n_motor

        self.command = "STANDBY"
        self.initial_angle = [0.0] * n_motor
        self.final_angle = [0.0] * n_motor
        self.interpolating_time = 0.0
        self.remaining_time = 0.0
        self.initialized = False

        self.lock = threading.Lock()

################ function ##################

def ros_command_callback(msg, params):
    robot_state, robot_command = params
    print("Received ROS Command: {}".format(msg.data))
    with robot_command.lock:
        prev_command = robot_command.command
        if not robot_command.initialized:
            pass
        elif msg.data == "STANDBY":
            robot_command.command = "STANDBY"
            with robot_state.lock:
                robot_command.initial_angle = robot_state.angle[:]
                robot_command.final_angle = STANDBY_ANGLE[:]
                robot_command.interpolating_time = 3.0
                robot_command.remaining_time = robot_command.interpolating_time
        elif msg.data == "STANDUP":
            robot_command.command = "STANDUP"
            with robot_state.lock:
                robot_command.initial_angle = robot_state.angle[:]
                robot_command.final_angle = STANDUP_ANGLE[:]
                robot_command.interpolating_time = 3.0
                robot_command.remaining_time = robot_command.interpolating_time
        elif msg.data == "DEBUG":
            robot_command.command = "DEBUG"
            with robot_state.lock:
                robot_command.initial_angle = robot_state.angle[:]
                robot_command.final_angle = DEBUG_ANGLE[:]
                robot_command.interpolating_time = 3.0
                robot_command.remaining_time = robot_command.interpolating_time
        elif robot_command.command == "STANDUP" and msg.data == "WALK":
            robot_command.command = "WALK"
        print("Command changed from {} to {}".format(prev_command, robot_command.command))

def realsense_vel_callback(msg, params):
    peripherals_state = params
    with peripherals_state.lock:
        peripherals_state.body_vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        # get odom quat
        peripherals_state.body_quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]

def realsense_gyro_callback(msg, params):
    peripherals_state = params
    with peripherals_state.lock:
        # peripherals_state.body_gyro = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        peripherals_state.body_gyro = [msg.angular_velocity.z, msg.angular_velocity.x, msg.angular_velocity.y]

def realsense_acc_callback(msg, params):
    peripherals_state = params
    with peripherals_state.lock:
        # peripherals_state.body_acc = [msg.linear_acceleration.x, msg.linear_acceleration.y, -msg.linear_acceleration.z]
        peripherals_state.body_acc = [msg.linear_acceleration.z, msg.linear_acceleration.x, msg.linear_acceleration.y]

def spacenav_joy_callback(msg, params):
    peripherals_state = params
    with peripherals_state.lock:
        peripherals_state.spacenav = [msg.axes[0], msg.axes[1], msg.axes[2], msg.axes[3], msg.axes[4], msg.axes[5], msg.buttons[0], msg.buttons[1]]

def main_controller(robot_state, robot_command, peripherals_state):
    policy_path = os.path.join(os.path.dirname(__file__), "../models/policy.pt")
    policy = mevius_utils.read_torch_policy(policy_path)
    last_actions = [0.0] * 12 # TODO initialize

    rate = rospy.Rate(CONTROL_HZ)
    while not rospy.is_shutdown():
        with robot_command.lock:
            if robot_command.command in ["STANDBY", "STANDUP", "DEBUG"]:
                robot_command.remaining_time -= 1.0/CONTROL_HZ
                robot_command.remaining_time = max(0, robot_command.remaining_time)
                if robot_command.remaining_time <= 0:
                    pass
                else:
                    ratio = 1 - robot_command.remaining_time / robot_command.interpolating_time
                    robot_command.angle = [a + (b-a)*ratio for a, b in zip(robot_command.initial_angle, robot_command.final_angle)]
            elif robot_command.command == "WALK":
                with peripherals_state.lock:
                    base_quat = peripherals_state.body_quat[:]
                    base_lin_vel = peripherals_state.body_vel[:]
                    base_ang_vel = peripherals_state.body_gyro[:]
                    nav = peripherals_state.spacenav[:]
                max_command = 0.6835
                ranges = mevius_utils.commands.ranges
                coefs = [ranges.lin_vel_x[1], ranges.lin_vel_y[1], ranges.ang_vel_yaw[1], ranges.heading[1]]
                commands_ = [nav[0], nav[1], nav[5], nav[5]]
                commands = [coef * command / max_command for coef, command in zip(coefs, commands_)]
                with robot_state.lock:
                    dof_pos = robot_state.angle[:]
                    dof_vel = robot_state.velocity[:]
                # print(base_quat, base_lin_vel, base_ang_vel, commands, dof_pos, dof_vel, last_actions)
                obs = mevius_utils.get_policy_observation(base_quat, base_lin_vel, base_ang_vel, commands, dof_pos, dof_vel, last_actions)
                actions = mevius_utils.get_policy_output(policy, obs)
                # print(actions)
                robot_command.angle = [mevius_utils.control.action_scale*a + b for a, b in zip(actions, STANDUP_ANGLE[:])]
                # print(robot_command.angle)
                last_actions = actions[:]
        # with peripherals_state.lock:
        #     print("Body Velocity: {}".format(peripherals_state.body_vel))
        #     print("Body Gyro: {}".format(peripherals_state.body_gyro))
        #     print("Body Acc: {}".format(peripherals_state.body_acc))

        rate.sleep()
        # time.sleep(1)

def can_communication(robot_state, robot_command, peripherals_state):
    device = "can0"
    motor_type = "AK70_10_V1p1"
    n_motor = 12
    motors = [None]*n_motor
    for i in range(n_motor):
        motors[i] = CanMotorController("can0", CAN_ID[i], motor_type=motor_type, motor_dir=MOTOR_DIR[i])

    print("Enabling Motors...")
    for i, motor in enumerate(motors):
        pos, vel, cur, tem = motor.enable_motor()
        print("Enabling Motor {} [Status] Pos: {:.3f}, Vel: {:.3f}, Cur: {:.3f}, Temp: {:.3f}".format(JOINT_NAME[i], pos, vel, cur, tem))
        with robot_state.lock:
            robot_state.angle[i] = pos
            robot_state.velocity[i] = vel
            robot_state.current[i] = cur
            robot_state.temperature[i] = tem

    urdf_fullpath = os.path.join(os.path.dirname(__file__), "../models/mevius.urdf")
    joint_params = mevius_utils.get_urdf_joint_params(urdf_fullpath, JOINT_NAME)

    state_pub = rospy.Publisher("mevius_log", MeviusLog, queue_size=2)

    print("Setting Initial Offset...")
    for i, motor in enumerate(motors):
        motor.set_angle_offset(STANDBY_ANGLE[i], deg=False)
        # motor.set_angle_range(joint_params[i][0], joint_params[i][1], deg=False)

    with robot_state.lock:
        robot_state.angle = STANDBY_ANGLE[:]

    with robot_command.lock:
        robot_command.command = "STANDBY"
        robot_command.angle = STANDBY_ANGLE[:]
        robot_command.initial_angle = STANDBY_ANGLE[:]
        robot_command.final_angle = STANDBY_ANGLE[:]
        robot_command.interpolating_time = 3.0
        robot_command.remaining_time = robot_command.interpolating_time
        robot_command.initialized = True

    rate = rospy.Rate(CAN_HZ)

    while not rospy.is_shutdown():
        msg = MeviusLog()
        msg.header.stamp = rospy.Time.now()

        with robot_command.lock:
            ref_angle = robot_command.angle[:]
            ref_velocity = robot_command.velocity[:]
            ref_kp = robot_command.kp[:]
            ref_kd = robot_command.kd[:]
            ref_torque = robot_command.torque[:]

        with robot_state.lock:
            # print(np.asarray(ref_angle))
            for i, motor in enumerate(motors):
                pos, vel, cur, tem = motor.send_rad_command(ref_angle[i], ref_velocity[i], ref_kp[i], ref_kd[i], ref_torque[i])
                robot_state.angle[i] = pos
                robot_state.velocity[i] = vel
                robot_state.current[i] = cur
                robot_state.temperature[i] = tem

            msg.angle = robot_state.angle[:]
            msg.velocity = robot_state.velocity[:]
            msg.current = robot_state.current[:]
            msg.temperature = robot_state.temperature[:]
            # print([b-a for a, b in zip(msg.angle, STANDUP_ANGLE[:])])

        with peripherals_state.lock:
            msg.body_vel = peripherals_state.body_vel[:]
            msg.body_quat = peripherals_state.body_quat[:]
            msg.body_gyro = peripherals_state.body_gyro[:]
            msg.body_acc = peripherals_state.body_acc[:]

        msg.ref_angle = ref_angle
        msg.ref_velocity = ref_velocity
        msg.ref_kp = ref_kp
        msg.ref_kd = ref_kd
        msg.ref_torque = ref_torque

        state_pub.publish(msg)

        rate.sleep()
        # time.sleep(1)

if __name__ == "__main__":
    rospy.init_node("mevius")

    robot_state = RobotState()
    peripheral_state = PeripheralState()
    robot_command = RobotCommand()

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_device", action="store_true", help="No device connection")
    args = parser.parse_args()

    rospy.Subscriber("/mevius_command", String, ros_command_callback, (robot_state, robot_command), queue_size=1)
    rospy.Subscriber("/camera/odom/sample", Odometry, realsense_vel_callback, peripheral_state, queue_size=1)
    rospy.Subscriber("/camera/gyro/sample", Imu, realsense_gyro_callback, peripheral_state, queue_size=1)
    rospy.Subscriber("/camera/accel/sample", Imu, realsense_acc_callback, peripheral_state, queue_size=1)
    rospy.Subscriber("/spacenav/joy", Joy, spacenav_joy_callback, peripheral_state, queue_size=1)
    main_controller_thread = threading.Thread(target=main_controller, args=(robot_state, robot_command, peripheral_state))
    if not args.no_device:
        can_communication_thread = threading.Thread(target=can_communication, args=(robot_state, robot_command, peripheral_state))

    main_controller_thread.start()
    if not args.no_device:
        can_communication_thread.start()

    main_controller_thread.join()
    if not args.no_device:
        can_communication_thread.join()

