import os
import pprint
import urdf_parser_py.urdf as urdf
import numpy as np
import torch
from isaacgym_torch_utils import quat_apply, quat_rotate_inverse, get_axis_params
from legged_gym_math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
import parameters as P


# copied from legged_gym
class normalization:
    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        height_measurements = 5.0
    clip_observations = 100.
    clip_actions = 100.

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
    input_data = torch.randn(1, 48)

    with torch.no_grad():
        output = policy(input_data)
        print(output.numpy()[0])

def get_policy_observation(base_quat_, base_lin_vel_, base_ang_vel_, command_, dof_pos_, dof_vel_, actions_):
    obs_scales = normalization.obs_scales
    default_dof_pos = torch.tensor(P.DEFAULT_ANGLE, dtype=torch.float32)

    forward_vec = torch.tensor([1., 0., 0.], dtype=torch.float, requires_grad=False).reshape(1, -1)
    gravity_vec = torch.tensor(get_axis_params(-1., 2), dtype=torch.float, requires_grad=False).reshape(1, -1)
    base_quat = torch.tensor(base_quat_, dtype=torch.float, requires_grad=False).reshape(1, -1)
    base_lin_vel = torch.tensor(base_lin_vel_[:], dtype=torch.float, requires_grad=False).reshape(1, -1)
    base_ang_vel = torch.tensor(base_ang_vel_[:], dtype=torch.float, requires_grad=False).reshape(1, -1)
    command = torch.tensor(command_, dtype=torch.float, requires_grad=False).reshape(1, -1)
    dof_pos = torch.tensor(dof_pos_, dtype=torch.float, requires_grad=False).reshape(1, -1)
    dof_vel = torch.tensor(dof_vel_, dtype=torch.float, requires_grad=False).reshape(1, -1)
    actions = torch.tensor(actions_, dtype=torch.float, requires_grad=False).reshape(1, -1)
    command_scale = torch.tensor([obs_scales.lin_vel, obs_scales.lin_vel, obs_scales.ang_vel], requires_grad=False)

    # base_lin_vel = quat_rotate_inverse(base_quat, base_lin_vel)
    # base_ang_vel = quat_rotate_inverse(base_quat, base_ang_vel)
    projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
    # print("----------------------------------")
    # print(base_quat.numpy()[0])
    # print(base_lin_vel.numpy()[0])
    # print(base_ang_vel.numpy()[0])
    # print(projected_gravity.numpy()[0])

    if P.commands.heading_command:
        forward = quat_apply(base_quat, forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        command[:, 2] = torch.clip(0.5*wrap_to_pi(command[:, 3] - heading), -1., 1.)

    # if the norm of command is lower than 0.03, set it to one
    is_standing = torch.tensor([[torch.norm(command[:, :3]) < 0.03]])

    obs = torch.cat((base_lin_vel * obs_scales.lin_vel, # 3D
                     base_ang_vel  * obs_scales.ang_vel, # 3D
                     projected_gravity, # 3D
                     command[:, :3] * command_scale, # 3D
                     (dof_pos - default_dof_pos) * obs_scales.dof_pos, # 12D
                     dof_vel * obs_scales.dof_vel, # 12D
                     actions, #12D
                     # is_standing,
                     ), dim=-1)
    return obs

def get_policy_output(policy, obs):
    clip_obs = normalization.clip_observations
    obs = torch.clip(obs, -clip_obs, clip_obs)

    with torch.no_grad():
        actions = policy(obs)

    clip_actions = normalization.clip_actions
    actions = torch.clip(actions, -clip_actions, clip_actions)
    return actions.numpy()[0] # reference angle [rad]

def test_get_policy_output():
    policy_path = os.path.join(os.path.dirname(__file__), "../models/policy.pt")
    policy = read_torch_policy(policy_path)
    base_quat = [0.0, 0.0, 0.0, 1.0]
    base_lin_vel = [0.0, 0.0, 0.0]
    base_ang_vel = [0.0, 0.0, 0.0]
    command = [0.0, 0.0, 0.0, 0.0]
    dof_pos = P.DEFAULT_ANGLE
    dof_vel = [0]*12
    actions = [0]*12
    obs = get_policy_observation(base_quat, base_lin_vel, base_ang_vel, command, dof_pos, dof_vel, actions)
    print(obs.numpy()[0])
    print(get_policy_output(policy, obs))


if __name__ == '__main__':
    print("# test_get_urdf_joint_params")
    test_get_urdf_joint_params()
    print("# test_read_torch_policy")
    test_read_torch_policy()
    print("# test_get_policy_output")
    test_get_policy_output()

