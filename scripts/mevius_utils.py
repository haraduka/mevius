import os
import pprint
import urdf_parser_py.urdf as urdf
import numpy as np
import torch
from isaacgym_torch_utils import quat_apply, quat_rotate_inverse, get_axis_params
from legged_gym_math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
import parameters as P


def get_urdf_joint_params(urdf_path, joint_names):
    if isinstance(urdf_path, str) and urdf_path.endswith('.urdf'):
        robot_urdf = open(urdf_path).read()

    robot = urdf.Robot.from_xml_string(robot_urdf)
    # joint_params = {}
    joint_params = [None]*len(robot.joints)

    for joint in robot.joints:
        if joint.type == 'revolute' or joint.type == 'continuous':
            if joint.limit:
                index = joint_names.index(joint.name)
                joint_params[index] = (joint.limit.lower, joint.limit.upper, joint.limit.effort, joint.limit.velocity)

    return joint_params

def test_get_urdf_joint_params():
    urdf_fullpath = os.path.join(os.path.dirname(__file__), "../models/mevius.urdf")

    joint_names = [
            'BL_collar', 'BL_hip', 'BL_knee',
            'BR_collar', 'BR_hip', 'BR_knee',
            'FL_collar', 'FL_hip', 'FL_knee',
            'FR_collar', 'FR_hip', 'FR_knee',
            ]
    pprint.pprint(get_urdf_joint_params(urdf_fullpath, joint_names))

def read_torch_policy(policy_path):
    policy = torch.jit.load(policy_path)
    policy.eval()
    return policy

def test_read_torch_policy():
    policy_path = os.path.join(os.path.dirname(__file__), "../models/policy.pt")
    policy = read_torch_policy(policy_path)
    input_data = torch.randn(1, 46 + 208)
    hidden = torch.zeros(2, 1, 128)

    with torch.no_grad():
        actions, hidden_, exte_pred, priv_pred = policy(input_data, hidden)
        print(actions.numpy()[0])

def get_policy_observation(
        base_quat_,
        base_ang_vel_,
        command_,
        dof_pos_,
        dof_vel_,
        is_standing_,
        phases_,
        dphases_,
        height_scan_,
        ):
    default_dof_pos = torch.tensor(P.DEFAULT_ANGLE, dtype=torch.float32)
    forward_vec = torch.tensor([1., 0., 0.], dtype=torch.float, requires_grad=False).reshape(1, -1)
    gravity_vec = torch.tensor(get_axis_params(-1., 2), dtype=torch.float, requires_grad=False).reshape(1, -1)
    base_quat = torch.tensor(base_quat_, dtype=torch.float, requires_grad=False).reshape(1, -1)
    base_ang_vel = torch.tensor(base_ang_vel_[:], dtype=torch.float, requires_grad=False).reshape(1, -1)
    command = torch.tensor(command_, dtype=torch.float, requires_grad=False).reshape(1, -1)
    dof_pos = torch.tensor(dof_pos_, dtype=torch.float, requires_grad=False).reshape(1, -1)
    dof_vel = torch.tensor(dof_vel_, dtype=torch.float, requires_grad=False).reshape(1, -1)
    phases = torch.cat([torch.sin(phases_), torch.cos(phases_)], dim=-1)

    projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)

    if P.commands.heading_command:
        forward = quat_apply(base_quat, forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        command[:, 2] = torch.clip(0.5*wrap_to_pi(command[:, 3] - heading), -1., 1.)

    obs = torch.cat((
        base_ang_vel, # 3D
        projected_gravity, # 3D
        command[:, :3], # 3D
        (dof_pos - default_dof_pos), # 12D
        dof_vel, # 12D
        is_standing_,
        phases,
        dphases_,
        height_scan_,
        ), dim=-1)
    return obs

def get_policy_output(policy, obs, hidden):
    with torch.no_grad():
        actions, hidden_, exte_pred, priv_pred = policy(obs, hidden)
    actions = torch.clip(actions, -P.control.action_clipping, P.control.action_clipping)
    return actions, hidden_, exte_pred, priv_pred

def test_get_policy_output():
    policy_path = os.path.join(os.path.dirname(__file__), "../models/policy.pt")
    policy = read_torch_policy(policy_path)
    base_quat = [0.0, 0.0, 0.0, 1.0]
    base_ang_vel = [0.0, 0.0, 0.0]
    command = [0.0, 0.0, 0.0, 0.0]
    dof_pos = P.DEFAULT_ANGLE
    dof_vel = [0.0]*12
    is_standing = [1.0]
    phases = [0.0]*4
    dphases = [0.0]*4
    height_scan = [0.0]*208
    hidden = torch.zeros(2, 1, 128)
    obs = get_policy_observation(base_quat, base_ang_vel, command, dof_pos, dof_vel, is_standing, phases, dphases, height_scan)
    print(obs.numpy()[0])
    print(get_policy_output(policy, obs, hidden))


if __name__ == '__main__':
    print("# test_get_urdf_joint_params")
    test_get_urdf_joint_params()
    print("# test_read_torch_policy")
    test_read_torch_policy()
    print("# test_get_policy_output")
    test_get_policy_output()

