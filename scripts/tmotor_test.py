import argparse
import sys
import select
import time
import rospy
import numpy as np
from tmotor_lib import CanMotorController

def setZeroPosition(motor):
    pos, _, _ = motor.set_zero_position()
    while abs(np.rad2deg(pos)) > 0.5:
        pos, vel, cur = motor.set_zero_position()
        print("Position: {}, Velocity: {}, Torque: {}".format(np.rad2deg(pos), np.rad2deg(vel), cur))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", '-d', type=str, default="can0", help="can interface name")
    parser.add_argument("--ids", '-i', type=int, nargs="+", default=None, help="motor ids to control")
    parser.add_argument("--motor_type", type=str, default="AK70_10_V1p1", help="motor type")
    parser.add_argument("--task", '-t', type=str, default="sense", help="[pos, sense, bilateral, real, mevius]")
    parser.add_argument("--value", '-v', type=int, default=None, help="value used for task")
    parser.add_argument("--kp", type=float, default=3.0, help="p gain for position control")
    parser.add_argument("--kd", type=float, default=1.0, help="d gain for position control")
    parser.add_argument("--hz", type=int, default=1000, help="hz to control motor")
    parser.add_argument("--time", type=float, default=3.0, help="time to control [sec]")
    args = parser.parse_args()

    print("# using Socket {} for can communication".format(args.device))
    print("# motor ids: {}".format(args.ids))
    assert args.ids is not None, "please input motor ids"
    if args.task == "bilateral":
        assert len(args.ids) == 2, "please input two motor ids for bilateral control"

    ids = args.ids
    motors = {}
    for id in ids:
        motor_dir = 1
        if args.task == "mevius":
            if id in [1, 4, 5, 6, 11, 12]:
                motor_dir = -1
        motors[id] = CanMotorController(args.device, id, motor_dir=motor_dir, motor_type=args.motor_type)

    print("Enabling Motors..")
    for motor_id, motor_controller in motors.items():
        pos, vel, cur, tem = motor_controller.enable_motor()
        print("Motor {} Status: Pos: {}, Vel: {}, Torque: {}, Temp: {}".format(motor_id, pos, vel, cur, tem))

    time.sleep(1)

    rospy.init_node("tmotor_test")
    rate = rospy.Rate(args.hz)

    pos_vec = []
    vel_vec = []
    cur_vec = []
    tem_vec = []
    for motor_id, motor_controller in motors.items():
        pos, vel, cur, tem = motor_controller.send_deg_command(0, 0, 0.0, 0.0, 0)
        pos_vec.append(pos)
        vel_vec.append(vel)
        cur_vec.append(cur)
        tem_vec.append(tem)

    start_time = time.time()

    if args.task == "pos":
        for deg in np.linspace(0.0, 360.0, int(args.time*args.hz/len(args.ids))):
            for motor_id, motor_controller in motors.items():
                pos, vel, cur, tem = motor_controller.send_deg_command(deg, 0, args.kp, args.kd, 0)
                print("Moving Motor {} Position: {}, Velocity: {}, Torque: {}, Temp: {}".format(motor_id, pos, vel, cur, tem))
            rate.sleep()
            end_time = time.time()
            print("Time taken: {}".format(end_time - start_time))
            start_time = end_time
    elif args.task == "sense":
        while not rospy.is_shutdown():
            for i, (motor_id, motor_controller) in enumerate(motors.items()):
                pos, vel, cur, tem = motor_controller.send_deg_command(0, 0, 0.0, 0.0, 0)
                print("Moving Motor {} Position: {}, Velocity: {}, Torque: {}, Temp: {}".format(motor_id, pos, vel, cur, tem))
            rate.sleep()
            # if enter is pressed, break with non-blocking
            if select.select([sys.stdin,],[],[],0.0)[0]:
                input_char = sys.stdin.read(1)
                if input_char == '\n':
                    break
    elif args.task == "bilateral":
        while not rospy.is_shutdown():
            for i, (motor_id, motor_controller) in enumerate(motors.items()):
                 command_cur = max(-4.0, min(0.02*(pos_vec[(i+1)%2]-pos_vec[i])+0.01*(vel_vec[(i+1)%2]-vel_vec[i]), 4.0))
                 pos, vel, cur, tem = motor_controller.send_deg_command(0, 0, 0.0, 0.0, command_cur)
                 print("Moving Motor {} Position: {}, Velocity: {}, Torque: {}, Temp: {}".format(motor_id, pos, vel, cur, tem))
                 pos_vec[i] = pos
                 vel_vec[i] = vel
                 cur_vec[i] = cur
                 tem_vec[i] = tem
            end_time = time.time()
            print("Time taken: {}".format(end_time - start_time))
            start_time = end_time
            rate.sleep()
            # if enter is pressed, break with non-blocking
            if select.select([sys.stdin,],[],[],0.0)[0]:
                input_char = sys.stdin.read(1)
                if input_char == '\n':
                    break
    elif args.task == "real":
        initial_pos = []
        for motor_id, motor_controller in motors.items():
            motor_controller.set_angle_range(-90, 90, deg=True)
            motor_controller.set_angle_offset(45, deg=True)
            initial_pos.append(motor_controller.current_pos)
        for deg in np.linspace(0.0, 360.0, int(args.time*args.hz/len(args.ids))):
            for i, (motor_id, motor_controller) in enumerate(motors.items()):
                pos, vel, cur, tem = motor_controller.send_deg_command(initial_pos[i]+deg, 0, args.kp, args.kd, 0)
                print("Moving Motor {} Position: {}, Velocity: {}, Torque: {}, Temp: {}".format(motor_id, pos, vel, cur, tem))
            rate.sleep()
            end_time = time.time()
            print("Time taken: {}".format(end_time - start_time))
        for deg in np.linspace(0.0, -360.0, int(args.time*args.hz/len(args.ids))):
            for i, (motor_id, motor_controller) in enumerate(motors.items()):
                pos, vel, cur, tem = motor_controller.send_deg_command(initial_pos[i]+deg, 0, args.kp, args.kd, 0)
                print("Moving Motor {} Position: {}, Velocity: {}, Torque: {}, Temp: {}".format(motor_id, pos, vel, cur, tem))
            rate.sleep()
            end_time = time.time()
            print("Time taken: {}".format(end_time - start_time))
            start_time = end_time
        print("end")
    elif args.task == "mevius":
        initial_offset = [-0.1, 1.4, -2.9, 0.1, 1.4, -2.9, -0.1, 1.4, -2.9, 0.1, 1.4, -2.9]
        final_angle = [0.0, 0.8, -1.6, 0.0, 0.8, -1.6, -0.0, 0.8, -1.6, 0.0, 0.8, -1.6]
        for i, (motor_id, motor_controller) in enumerate(motors.items()):
            motor_controller.set_angle_offset(initial_offset[motor_id-1], deg=False)
        move_time = 5.0
        for deg_rate in np.linspace(0.0, 1.0, int(move_time*args.hz/len(args.ids))):
            for i, (motor_id, motor_controller) in enumerate(motors.items()):
                command = initial_offset[motor_id-1]+deg_rate*(final_angle[motor_id-1]-initial_offset[motor_id-1])
                pos, vel, cur, tem = motor_controller.send_rad_command(command, 0, args.kp, args.kd, 0)
                print("Moving Motor {} Command: {}, Position: {}, Velocity: {}, Torque: {}, Temp: {}".format(motor_id, command, pos, vel, cur, tem))
            rate.sleep()
            end_time = time.time()
            print("Time taken: {}".format(end_time - start_time))
            if select.select([sys.stdin,],[],[],0.0)[0]:
                input_char = sys.stdin.read(1)
                if input_char == '\n':
                    break
        input()
        for deg_rate in np.linspace(0.0, 1.0, int(move_time*args.hz/len(args.ids))):
            for i, (motor_id, motor_controller) in enumerate(motors.items()):
                command = final_angle[motor_id-1]+deg_rate*(initial_offset[motor_id-1]-final_angle[motor_id-1])
                pos, vel, cur, tem = motor_controller.send_rad_command(command, 0, args.kp, args.kd, 0)
                print("Moving Motor {} Command: {}, Position: {}, Velocity: {}, Torque: {}, Temp: {}".format(motor_id, command, pos, vel, cur, tem))
            rate.sleep()
            end_time = time.time()
            print("Time taken: {}".format(end_time - start_time))
            if select.select([sys.stdin,],[],[],0.0)[0]:
                input_char = sys.stdin.read(1)
                if input_char == '\n':
                    break
        input()
    else:
        import ipdb
        ipdb.set_trace()

    time.sleep(1)

    print("Disabling Motors...")
    for motor_id, motor_controller in motors.items():
        pos, vel, cur, tem = motor_controller.disable_motor()
        time.sleep(0.2)
        print("Motor {} Status: Pos: {}, Vel: {}, Torque: {}".format(motor_id, pos, vel, cur))


if __name__ == "__main__":
    main()
