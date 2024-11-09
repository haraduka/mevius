import argparse
import os
import sys
import select
import time
import threading
import numpy as np

import rospy
from scipy.spatial.transform import Rotation
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Imu, Joy, JointState
from nav_msgs.msg import Odometry
from tmotor_lib import CanMotorController
from mevius.msg import MeviusLog
import mevius_utils
import parameters as P

# TODO add terminal display of thermometer, etc.

np.set_printoptions(precision=3)

class RobotState:
    def __init__(self, n_motor=12):
        self.angle = [0.0] * n_motor
        self.velocity = [0.0] * n_motor
        self.current = [0.0] * n_motor
        self.temperature = [0.0] * n_motor
        self.lock = threading.Lock()

class PeripheralState:
    def __init__(self):
        self.realsense_last_time = None
        self.body_vel = [0.0] * 3
        self.body_quat = [0.0] * 4
        self.body_gyro = [0.0] * 3
        self.body_acc = [0.0] * 3
        self.spacenav_enable = False
        self.spacenav = [0.0] * 8
        self.virtual_enable = False
        self.virtual = [0.0] * 4
        self.lock = threading.Lock()

class RobotCommand:
    def __init__(self, n_motor=12):
        self.angle = [0.0] * n_motor
        self.velocity = [0.0] * n_motor
        self.kp = []
        self.kd = []
        self.coef = 1.0
        for name in P.JOINT_NAME:
            for key in P.control.stiffness.keys():
                if key in name:
                    self.kp.append(P.control.stiffness[key]*self.coef)
                    self.kd.append(P.control.damping[key]*self.coef)
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

def command_callback(command, robot_state, robot_command):
    with robot_command.lock:
        prev_command = robot_command.command
        if not robot_command.initialized:
            pass
    if command == "STANDBY-STANDUP":
        with robot_command.lock:
            if robot_command.remaining_time < 0.1:
                if prev_command == "STANDBY":
                    robot_command.command = "STANDUP"
                    with robot_state.lock:
                        robot_command.initial_angle = robot_state.angle[:]
                        robot_command.final_angle = P.DEFAULT_ANGLE[:]
                        robot_command.interpolating_time = 3.0
                        robot_command.remaining_time = robot_command.interpolating_time
                elif prev_command == "STANDUP":
                    robot_command.command = "STANDBY"
                    with robot_state.lock:
                        robot_command.initial_angle = robot_state.angle[:]
                        robot_command.final_angle = P.STANDBY_ANGLE[:]
                        robot_command.interpolating_time = 3.0
                        robot_command.remaining_time = robot_command.interpolating_time
    elif command == "STANDUP-WALK":
        with robot_command.lock:
            if robot_command.remaining_time < 0.1:
                if prev_command == "STANDUP":
                    robot_command.command = "WALK"
                    robot_command.interpolating_time = 3.0
                    robot_command.remaining_time = robot_command.interpolating_time
                elif prev_command == "WALK":
                    robot_command.command = "STANDUP"
                    with robot_state.lock:
                        robot_command.initial_angle = robot_state.angle[:]
                        robot_command.final_angle = P.DEFAULT_ANGLE[:]
                        robot_command.interpolating_time = 3.0
                        robot_command.remaining_time = robot_command.interpolating_time
    elif command == "STANDBY":
        with robot_command.lock:
            robot_command.command = "STANDBY"
            with robot_state.lock:
                robot_command.initial_angle = robot_state.angle[:]
                robot_command.final_angle = P.STANDBY_ANGLE[:]
                robot_command.interpolating_time = 3.0
                robot_command.remaining_time = robot_command.interpolating_time
    elif command == "STANDUP":
            robot_command.command = "STANDUP"
            with robot_state.lock:
                robot_command.initial_angle = robot_state.angle[:]
                robot_command.final_angle = P.DEFAULT_ANGLE[:]
                robot_command.interpolating_time = 3.0
                robot_command.remaining_time = robot_command.interpolating_time
    elif command == "DEBUG":
            robot_command.command = "DEBUG"
            with robot_state.lock:
                robot_command.initial_angle = robot_state.angle[:]
                robot_command.final_angle = P.DEBUG_ANGLE[:]
                robot_command.interpolating_time = 3.0
                robot_command.remaining_time = robot_command.interpolating_time
    elif prev_command == "STANDUP" and command == "WALK":
            robot_command.command = "WALK"

    with robot_command.lock:
        print("Command changed from {} to {}".format(prev_command, robot_command.command))

def realsense_vel_callback(msg, params):
    peripherals_state = params
    with peripherals_state.lock:
        peripherals_state.body_vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        # get odom quat
        peripherals_state.body_quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        peripherals_state.realsense_last_time = time.time()

def realsense_gyro_callback(msg, params):
    peripherals_state = params
    with peripherals_state.lock:
        # peripherals_state.body_gyro = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        # for realsenes arrangement
        peripherals_state.body_gyro = [msg.angular_velocity.z, msg.angular_velocity.x, msg.angular_velocity.y]

def realsense_acc_callback(msg, params):
    peripherals_state = params
    with peripherals_state.lock:
        # peripherals_state.body_acc = [msg.linear_acceleration.x, msg.linear_acceleration.y, -msg.linear_acceleration.z]
        # for realsenes arrangement
        peripherals_state.body_acc = [msg.linear_acceleration.z, msg.linear_acceleration.x, msg.linear_acceleration.y]

def virtual_joy_callback(msg, params):
    peripherals_state = params
    with peripherals_state.lock:
        peripherals_state.virtual_enable = True
        peripherals_state.virtual = [msg.axes[0], msg.axes[1], msg.buttons[0], msg.buttons[1]]
        one_pushed = peripherals_state.virtual[2]
        two_pushed = peripherals_state.virtual[3]

    if one_pushed == 1:
        command_callback("STANDBY-STANDUP", robot_state, robot_command)
    elif two_pushed == 1:
        command_callback("STANDUP-WALK", robot_state, robot_command)

def ros_command_callback(msg, params):
    robot_state, robot_command = params
    print("Received ROS Command: {}".format(msg.data))
    command_callback(msg.data, robot_state, robot_command)

def spacenav_joy_callback(msg, params):
    peripherals_state = params
    with peripherals_state.lock:
        peripherals_state.spacenav_enable = True
        peripherals_state.spacenav = [msg.axes[0], msg.axes[1], msg.axes[2], msg.axes[3], msg.axes[4], msg.axes[5], msg.buttons[0], msg.buttons[1]]
        left_pushed = peripherals_state.spacenav[6]
        right_pushed = peripherals_state.spacenav[7]

    if left_pushed == 1:
        command_callback("STANDBY-STANDUP", robot_state, robot_command)
    elif right_pushed == 1:
        command_callback("STANDUP-WALK", robot_state, robot_command)

def main_controller(robot_state, robot_command, peripherals_state):
    policy_path = os.path.join(os.path.dirname(__file__), "../models/policy.pt")
    policy = mevius_utils.read_torch_policy(policy_path).to("cpu")

    urdf_fullpath = os.path.join(os.path.dirname(__file__), "../models/mevius.urdf")
    joint_params = mevius_utils.get_urdf_joint_params(urdf_fullpath, P.JOINT_NAME)

    is_safe = True
    last_actions = [0.0] * 12 # TODO initialize

    rate = rospy.Rate(P.CONTROL_HZ)
    while not rospy.is_shutdown():
        with robot_command.lock:
            command = robot_command.command
        if command in ["STANDBY", "STANDUP", "DEBUG"]:
            with robot_command.lock:
                robot_command.remaining_time -= 1.0/P.CONTROL_HZ
                robot_command.remaining_time = max(0, robot_command.remaining_time)
                if robot_command.remaining_time <= 0:
                    pass
                else:
                    ratio = 1 - robot_command.remaining_time / robot_command.interpolating_time
                    robot_command.angle = [a + (b-a)*ratio for a, b in zip(robot_command.initial_angle, robot_command.final_angle)]
        elif command in ["WALK"]:
            with robot_command.lock:
                robot_command.remaining_time -= 1.0/P.CONTROL_HZ
                robot_command.remaining_time = max(0, robot_command.remaining_time)

            with peripherals_state.lock:
                base_quat = peripherals_state.body_quat[:]
                base_lin_vel = peripherals_state.body_vel[:]
                base_ang_vel = peripherals_state.body_gyro[:]

                ranges = P.commands.ranges
                coefs = [ranges.lin_vel_x[1], ranges.lin_vel_y[1], ranges.ang_vel_yaw[1], ranges.heading[1]]
                if peripherals_state.spacenav_enable:
                    nav = peripherals_state.spacenav[:]
                    max_command = 0.6835
                    commands_ = [nav[0], nav[1], nav[5], nav[5]]
                    commands = [[min(max(-coef, coef * command / max_command), coef) for coef, command in zip(coefs, commands_)]]
                elif peripherals_state.virtual_enable:
                    nav = peripherals_state.virtual[:]
                    max_command = 1.0
                    commands_ = [nav[1], nav[0], 0, 0]
                    commands = [[min(max(-coef, coef * command / max_command), coef) for coef, command in zip(coefs, commands_)]]
                else:
                    commands = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float, requires_grad=False)

        # for safety
        if command in ["WALK"]:
            # no realsense
            with peripherals_state.lock:
                if peripherals_state.realsense_last_time is None:
                    is_safe = False
                    print("No Connection to Realsense. PD gains become 0.")
                if (peripherals_state.realsense_last_time is not None) and (time.time() - peripherals_state.realsense_last_time > 0.1):
                    print("Realsense data is too old. PD gains become 0.")
                    is_safe = False
            # falling down
            if is_safe and (Rotation.from_quat(base_quat).as_matrix()[2, 2] < 0.6):
                is_safe = False
                print("Robot is almost fell down. PD gains become 0.")

            if not is_safe:
                print("Robot is not safe. Please reboot the robot.")
                with robot_command.lock:
                    robot_command.kp = [0.0] * 12
                    robot_command.kd = [0.0] * 12
                    with robot_state.lock:
                        robot_command.angle = robot_state.angle[:]
                rate.sleep()
                continue


        if command in ["WALK"]:
            with robot_state.lock:
                dof_pos = robot_state.angle[:]
                dof_vel = robot_state.velocity[:]
            # print(base_quat, base_lin_vel, base_ang_vel, commands, dof_pos, dof_vel, last_actions)
            obs = mevius_utils.get_policy_observation(base_quat, base_lin_vel, base_ang_vel, commands, dof_pos, dof_vel, last_actions)
            actions = mevius_utils.get_policy_output(policy, obs)
            scaled_actions = P.control.action_scale * actions

        if command in ["WALK"]:
            ref_angle = [a + b for a, b in zip(scaled_actions, P.DEFAULT_ANGLE[:])]
            with robot_state.lock:
                for i in range(len(ref_angle)):
                    if robot_state.angle[i]  < joint_params[i][0] or robot_state.angle[i] > joint_params[i][1]:
                        ref_angle[i] = max(joint_params[i][0]+0.1, min(ref_angle[i], joint_params[i][1]-0.1))
                        print("# Joint {} out of range: {:.3f}".format(P.JOINT_NAME[i], robot_state.angle[i]))
            with robot_command.lock:
                robot_command.angle = ref_angle

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
        motors[i] = CanMotorController("can0", P.CAN_ID[i], motor_type=motor_type, motor_dir=P.MOTOR_DIR[i])

    print("Enabling Motors...")
    for i, motor in enumerate(motors):
        pos, vel, cur, tem = motor.enable_motor()
        print("Enabling Motor {} [Status] Pos: {:.3f}, Vel: {:.3f}, Cur: {:.3f}, Temp: {:.3f}".format(P.JOINT_NAME[i], pos, vel, cur, tem))
        with robot_state.lock:
            robot_state.angle[i] = pos
            robot_state.velocity[i] = vel
            robot_state.current[i] = cur
            robot_state.temperature[i] = tem

    state_pub = rospy.Publisher("mevius_log", MeviusLog, queue_size=2)
    jointstate_pub = rospy.Publisher("joint_states", JointState, queue_size=2)

    print("Setting Initial Offset...")
    for i, motor in enumerate(motors):
        motor.set_angle_offset(P.STANDBY_ANGLE[i], deg=False)
        # motor.set_angle_range(joint_params[i][0], joint_params[i][1], deg=False)

    with robot_state.lock:
        robot_state.angle = P.STANDBY_ANGLE[:]

    with robot_command.lock:
        robot_command.command = "STANDBY"
        robot_command.angle = P.STANDBY_ANGLE[:]
        robot_command.initial_angle = P.STANDBY_ANGLE[:]
        robot_command.final_angle = P.STANDBY_ANGLE[:]
        robot_command.interpolating_time = 3.0
        robot_command.remaining_time = robot_command.interpolating_time
        robot_command.initialized = True

    rate = rospy.Rate(P.CAN_HZ)

    error_count = [0]*12
    while not rospy.is_shutdown():
        start_time = time.time()
        msg = MeviusLog()
        msg.header.stamp = rospy.Time.now()

        jointstate_msg = JointState()
        jointstate_msg.header.stamp = rospy.Time.now()

        with robot_command.lock:
            ref_angle = robot_command.angle[:]
            ref_velocity = robot_command.velocity[:]
            ref_kp = robot_command.kp[:]
            ref_kd = robot_command.kd[:]
            ref_torque = robot_command.torque[:]

        n_motor = 12
        pos_list = [0]*n_motor
        vel_list = [0]*n_motor
        cur_list = [0]*n_motor
        tem_list = [0]*n_motor
        for i, motor in enumerate(motors):
            try:
                pos, vel, cur, tem = motor.send_rad_command(ref_angle[i], ref_velocity[i], ref_kp[i], ref_kd[i], ref_torque[i])
            except:
                error_count[i] += 1
                print("# Can Reciver is Failed for {}, ({})".format(P.JOINT_NAME[i], error_count[i]))
                continue
            pos_list[i] = pos
            vel_list[i] = vel
            cur_list[i] = cur
            tem_list[i] = tem

        with robot_state.lock:
            robot_state.angle = pos_list
            robot_state.velocity = vel_list
            robot_state.current = cur_list
            robot_state.temperature = tem_list

        jointstate_msg.name = P.JOINT_NAME
        jointstate_msg.position = pos_list
        jointstate_msg.velocity = vel_list
        jointstate_msg.effort = cur_list
        jointstate_pub.publish(jointstate_msg)

        msg.angle = pos_list
        msg.velocity = vel_list
        msg.current = cur_list
        msg.temperature = tem_list

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

        # rate.sleep()
        end_time = time.time()
        if end_time - start_time < 1.0/P.CAN_HZ:
            time.sleep(1.0/P.CAN_HZ - (end_time - start_time))
            # end_time = time.time()
            # print(end_time-start_time)


def sim_communication(robot_state, robot_command, peripherals_state):
    import mujoco
    import mujoco_viewer
    import tf

    xml_path = os.path.abspath('models/scene.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_step(model, data)

    mujoco_joint_names = [model.joint(i).name for i in range(model.njnt)]
    with robot_state.lock:
        for i, name in enumerate(P.JOINT_NAME):
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            robot_state.angle[i] = data.qpos[7+idx]
            robot_state.velocity[i] = data.qvel[6+idx]
            robot_state.current[i] = 0.0
            robot_state.temperature[i] = 25.0

    mujoco_actuator_names = [model.actuator(i).name for i in range(model.nu)]
    for i, name in enumerate(P.JOINT_NAME):
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        data.ctrl[idx] = P.STANDBY_ANGLE[i]

    state_pub = rospy.Publisher("mevius_log", MeviusLog, queue_size=2)
    jointstate_pub = rospy.Publisher("joint_states", JointState, queue_size=2)

    with robot_state.lock:
        robot_state.angle = P.STANDBY_ANGLE[:]

    with robot_command.lock:
        robot_command.command = "STANDBY"
        robot_command.angle = P.STANDBY_ANGLE[:]
        robot_command.initial_angle = P.STANDBY_ANGLE[:]
        robot_command.final_angle = P.STANDBY_ANGLE[:]
        robot_command.interpolating_time = 3.0
        robot_command.remaining_time = robot_command.interpolating_time
        robot_command.initialized = True

    rate = rospy.Rate(200) # mujoco hz

    while not rospy.is_shutdown() and viewer.is_alive:

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        msg = MeviusLog()
        msg.header.stamp = rospy.Time.now()

        jointstate_msg = JointState()
        jointstate_msg.header.stamp = rospy.Time.now()

        with robot_command.lock:
            ref_angle = robot_command.angle[:]
            ref_velocity = robot_command.velocity[:]
            ref_kp = robot_command.kp[:]
            ref_kd = robot_command.kd[:]
            ref_torque = robot_command.torque[:]

        mujoco_actuator_names = [model.actuator(i).name for i in range(model.nu)]
        for i, name in enumerate(P.JOINT_NAME): # mevius
            if name in mujoco_actuator_names: # mujoco
                idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) # mujoco
                data.ctrl[idx] = ref_angle[i]

        mujoco.mj_step(model, data)

        with robot_state.lock:
            for i, name in enumerate(P.JOINT_NAME):
                if name in mujoco_joint_names:
                    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                    robot_state.angle[i] = data.qpos[7+idx]
                    robot_state.velocity[i] = data.qvel[6+idx]
                    robot_state.current[i] = 0.0
                    robot_state.temperature[i] = 25.0

        with robot_state.lock:
            msg.angle = robot_state.angle[:]
            msg.velocity = robot_state.velocity[:]
            msg.current = robot_state.current[:]
            msg.temperature = robot_state.temperature[:]

        jointstate_msg.name = P.JOINT_NAME
        jointstate_msg.position = msg.angle
        jointstate_msg.velocity = msg.velocity
        jointstate_msg.effort = msg.current
        jointstate_pub.publish(jointstate_msg)

        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.twist.twist.linear.x = data.qvel[0]
        odom_msg.twist.twist.linear.y = data.qvel[1]
        odom_msg.twist.twist.linear.z = data.qvel[2]
        # CAUTION! mujoco and isaacgym's quat ordre is different
        odom_msg.pose.pose.orientation.w = data.qpos[3]
        odom_msg.pose.pose.orientation.x = data.qpos[4]
        odom_msg.pose.pose.orientation.y = data.qpos[5]
        odom_msg.pose.pose.orientation.z = data.qpos[6]
        realsense_vel_callback(odom_msg, peripherals_state)

        gyro_msg = Imu()
        gyro_msg.header.stamp = rospy.Time.now()
        # for realsense
        gyro_msg.angular_velocity.x = data.qvel[4]
        gyro_msg.angular_velocity.y = data.qvel[5]
        gyro_msg.angular_velocity.z = data.qvel[3]
        realsense_gyro_callback(gyro_msg, peripherals_state)

        acc_msg = Imu()
        acc_msg.header.stamp = rospy.Time.now()
        # for realsense
        acc_msg.linear_acceleration.x = data.qacc[1]
        acc_msg.linear_acceleration.y = data.qacc[2]
        acc_msg.linear_acceleration.z = data.qacc[0]
        realsense_acc_callback(acc_msg, peripherals_state)

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

        viewer.render()
        rate.sleep()

if __name__ == "__main__":
    rospy.init_node("mevius")

    robot_state = RobotState()
    peripheral_state = PeripheralState()
    robot_command = RobotCommand()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="do simulation")
    args = parser.parse_args()

    rospy.Subscriber("/mevius_command", String, ros_command_callback, (robot_state, robot_command), queue_size=1)
    rospy.Subscriber("/camera/odom/sample", Odometry, realsense_vel_callback, peripheral_state, queue_size=1)
    rospy.Subscriber("/camera/gyro/sample", Imu, realsense_gyro_callback, peripheral_state, queue_size=1)
    rospy.Subscriber("/camera/accel/sample", Imu, realsense_acc_callback, peripheral_state, queue_size=1)
    rospy.Subscriber("/spacenav/joy", Joy, spacenav_joy_callback, peripheral_state, queue_size=1)
    rospy.Subscriber("/virtual/joy", Joy, virtual_joy_callback, peripheral_state, queue_size=1)
    main_controller_thread = threading.Thread(target=main_controller, args=(robot_state, robot_command, peripheral_state))
    if not args.sim:
        can_communication_thread = threading.Thread(target=can_communication, args=(robot_state, robot_command, peripheral_state))
    else:
        sim_communication_thread = threading.Thread(target=sim_communication, args=(robot_state, robot_command, peripheral_state))

    main_controller_thread.start()
    if not args.sim:
        can_communication_thread.start()
    else:
        sim_communication_thread.start()

    main_controller_thread.join()
    if not args.sim:
        can_communication_thread.join()
    else:
        sim_communication_thread.join()

