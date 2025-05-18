import torch
from typing import Tuple
from isaacgym_torch_utils import quat_conjugate, quat_mul, quat_rotate, quat_rotate_inverse

@torch.jit.script
def quat_axis(axis:torch.Tensor, angle:torch.Tensor)->torch.Tensor:
    """
    Calculate the quaternion from the axis and angle
    Args:
        axis: axis tensor : torch.Tensor(N, 3)
        angle: angle tensor : torch.Tensor(N,)
    Returns:
        q: quaternion tensor : torch.Tensor(N, 4)
    """
    assert isinstance(axis, torch.Tensor), f"axis must be torch.Tensor but got {type(axis)}"
    assert isinstance(angle, torch.Tensor), f"angle must be torch.Tensor but got {type(angle)}"

    assert axis.shape[1] == 3, f"axis must have 3 columns but got {axis.shape[1]}"
    assert len(angle.shape) == 1, f"angle must be 1D tensor but got {len(angle.shape)}"
    assert axis.shape[0] == angle.shape[0], f"axis and angle must have the same number of rows but got {axis.shape[0]} and {angle.shape[0]}"


    n = axis.shape[0]
    device = axis.device
    q = torch.zeros(n, 4, device=device)
    axis_norm = torch.clip(torch.norm(axis, dim=1, keepdim=True), min=1e-5)
    q[:, :3] = torch.sin(angle.unsqueeze(-1)/2) * axis / axis_norm
    q[:, 3] = torch.cos(angle/2)
    return q

@torch.jit.script
def ik_2d_leg(l1:float, l2:float, pos_tensor_2d:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse kinematics for 2D leg
    Args:
        l1: length of the first link : float
        l2: length of the second link : float
        pos_tensor_2d: 2D position of the end effector : torch.Tensor(N, 2)
    Returns:
        is_ik_solved_tensor: torch.Tensor(N,) : boolean tensor indicating if the end effector is reachable  
        q_tensor_2d : torch.Tensor(N, 2) : joint angles of the two links
    """
    # type validation
    assert isinstance(l1, float), f"l1 must be float but got {type(l1)}"
    assert isinstance(l2, float), f"l2 must be float but got {type(l2)}"
    assert isinstance(pos_tensor_2d, torch.Tensor), f"pos_tensor_2d must be torch.Tensor but got {type(pos_tensor_2d)}"

    # shape validation
    assert pos_tensor_2d.shape[1] == 2, f"pos_tensor_2d must have 2 columns but got {pos_tensor_2d.shape[1]}"

    n = pos_tensor_2d.shape[0]
    device = pos_tensor_2d.device
    is_ik_solved_tensor = torch.zeros(n, dtype=torch.bool, device=device)
    q_tensor_2d = torch.zeros_like(pos_tensor_2d)

    # check if the end effector is reachable
    r_sq = torch.sum(torch.square(pos_tensor_2d), dim=1)
    r_max = (l1 + l2)**2
    r_min = (l1 - l2)**2
    is_ik_solved_tensor = torch.logical_and(r_sq <= r_max, r_sq >= r_min)

    cosq2 = (r_sq - l1**2 - l2**2) / (2 * l1 * l2)
    q_tensor_2d[:,1]= -torch.acos(cosq2)
    q2_abs = torch.abs(q_tensor_2d[:,1])
    q_tensor_2d[:,0] = -torch.atan2(pos_tensor_2d[:,1], pos_tensor_2d[:,0]) + torch.atan2(l2*torch.sin(q2_abs), l1 + l2*torch.cos(q2_abs)) - torch.pi/2

    # replace q_tensor_2d elements with zero where the end effector is not reachable
    q_tensor_2d[~is_ik_solved_tensor] = 0
    return is_ik_solved_tensor, q_tensor_2d

@torch.jit.script
def ik_3d_leg(l1:float, l2:float, l3:float, pos_tensor_3d:torch.Tensor, boundary_limit_soft:float=0.01)->Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse kinematics for 3D leg
    Args:
        l1: length of the first link : float
        l2: length of the second link : float
        l3: length of the third link : float
        pos_tensor_3d: 3D position of the end effector : torch.Tensor(N, 3)
    Returns:
        is_ik_solved_tensor: torch.Tensor(N,) : boolean tensor indicating if the end effector is reachable  
        q_tensor_3d: torch.Tensor(N, 3) : joint angles of the three links(hip, knee, ankle)
    """ 
    # type validation
    assert isinstance(l1, float), f"l1 must be float but got {type(l1)}"
    assert isinstance(l2, float), f"l2 must be float but got {type(l2)}"
    assert isinstance(l3, float), f"l3 must be float but got {type(l3)}"
    assert isinstance(pos_tensor_3d, torch.Tensor), f"pos_tensor_3d must be torch.Tensor but got {type(pos_tensor_3d)}"

    # shape validation
    assert pos_tensor_3d.shape[1] == 3, f"pos_tensor_3d must have 3 columns but got {pos_tensor_3d.shape[1]}"

    n = pos_tensor_3d.shape[0]
    device = pos_tensor_3d.device

    is_ik_solved_tensor = torch.ones(n, dtype=torch.bool, device=device)
    q_tensor_3d = torch.zeros_like(pos_tensor_3d)

    boundary_limit_soft = boundary_limit_soft ** 2 # allow small error (default=1mm) in reachability check and make it reachable by setting the target_pos to the boundary

    # check if the end effector is reachable
    r_sq = torch.sum(torch.square(pos_tensor_3d), dim=1)
    r_max = l1**2 + (l2+l3)**2
    r_min = l1**2 + (l2-l3)**2
    r_yz = torch.sqrt(torch.square(pos_tensor_3d[:,1]) + torch.square(pos_tensor_3d[:,2]))

    exceed_max = torch.logical_and(r_sq > r_max, r_sq < r_max+boundary_limit_soft)
    exceed_min = torch.logical_and(r_sq < r_min, r_sq > r_min-boundary_limit_soft)
    exceed_yz = torch.logical_and(r_yz < l1, r_yz > l1-boundary_limit_soft)

    max_bounded_pos = torch.clone(pos_tensor_3d)*(r_max**0.5-boundary_limit_soft**0.5)/torch.sqrt(r_sq).unsqueeze(-1)
    min_bounded_pos = torch.clone(pos_tensor_3d)*(r_min**0.5+boundary_limit_soft**0.5)/torch.sqrt(r_sq).unsqueeze(-1)
    yz_bounded_pos = torch.clone(pos_tensor_3d)*l1/r_yz.unsqueeze(-1)

    pos_tensor_3d = torch.where(exceed_max.unsqueeze(-1), max_bounded_pos, pos_tensor_3d)
    pos_tensor_3d = torch.where(exceed_min.unsqueeze(-1), min_bounded_pos, pos_tensor_3d)
    pos_tensor_3d = torch.where(exceed_yz.unsqueeze(-1), yz_bounded_pos, pos_tensor_3d)


    r_sq = torch.sum(torch.square(pos_tensor_3d), dim=1)
    r_yz = torch.sqrt(torch.square(pos_tensor_3d[:,1]) + torch.square(pos_tensor_3d[:,2]))
    is_ik_solved_tensor = torch.logical_and(torch.logical_and(r_sq <= r_max, r_sq >= r_min), r_yz >= l1)

    if (~is_ik_solved_tensor).any():
        print("not reachable for pos=", pos_tensor_3d[~is_ik_solved_tensor])


    # calculate q1(hip)
    a = pos_tensor_3d[:, 1]*l1
    b = pos_tensor_3d[:,2]*torch.sqrt(torch.square(pos_tensor_3d[:,1]) + torch.square(pos_tensor_3d[:,2]) - l1**2)
    c = torch.square(pos_tensor_3d[:,1]) + torch.square(pos_tensor_3d[:,2])

    cos1_positive = (a + b) / c
    cos1_negative = (a - b) / c
    cos1 = torch.max(cos1_positive, cos1_negative)
    # sin1 satisfies y*cos1 + z*sin1 = l1
    # in order to avoid zero division on z, select from sin1_positive and sin1_negative which satisfies upper equation
    sin1_positive = torch.sqrt(torch.max(torch.zeros_like(cos1), 1.0 - torch.square(cos1)))
    sin1_negative = -sin1_positive
    sin1 = torch.where(torch.abs(pos_tensor_3d[:,1]*cos1 + pos_tensor_3d[:,2]*sin1_positive - l1) < 1e-5, sin1_positive, sin1_negative)
    q_tensor_3d[:,0] = torch.atan2(sin1, cos1).squeeze()

    # calculate q2(knee), q3(ankle)
    y_hat = -pos_tensor_3d[:,1]*torch.sin(q_tensor_3d[:,0]) + pos_tensor_3d[:,2]*torch.cos(q_tensor_3d[:,0])
    pos_tensor_2d = torch.stack([pos_tensor_3d[:,0], y_hat], dim=1) #(N, 2)
    _, q_tensor_2d = ik_2d_leg(l2, l3, pos_tensor_2d)
    q_tensor_3d[:,1:] = q_tensor_2d

    # replace q_tensor_3d elements with zero where the end effector is not reachable
    q_tensor_3d[~is_ik_solved_tensor] = 0
    if (torch.isnan(q_tensor_3d).any()):
        print("nan is detected")

    return is_ik_solved_tensor, q_tensor_3d

@torch.jit.script
def fk_3d_leg(l1:float, l2:float, l3:float, q_tensor_3d:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward kinematics for 3D leg
    Args:
        l1: length of the first link : float
        l2: length of the second link : float
        l3: length of the third link : float
        q_tensor_3d: joint angles of the three links(hip, knee, ankle) : torch.Tensor(N, 3)
    Returns:
        hip_pos: torch.Tensor(N, 3) : position of the hip joint
        knee_pos: torch.Tensor(N, 3) : position of the knee joint
        ankle_pos: torch.Tensor(N, 3) : position of the ankle joint
        foot_pos: torch.Tensor(N, 3) : position of the foot
    """
    # type validation
    assert isinstance(l1, float), f"l1 must be float but got {type(l1)}"
    assert isinstance(l2, float), f"l2 must be float but got {type(l2)}"
    assert isinstance(l3, float), f"l3 must be float but got {type(l3)}"
    assert isinstance(q_tensor_3d, torch.Tensor), f"q_tensor_3d must be torch.Tensor but got {type(q_tensor_3d)}"

    # shape validation
    assert q_tensor_3d.shape[1] == 3, f"q_tensor_3d must have 3 columns but got {q_tensor_3d.shape[1]}"

    n = q_tensor_3d.shape[0]
    device = q_tensor_3d.device

    hip_pos = torch.zeros(n, 3, device=device)

    # calculate the position of the end effector
    hip_to_knee = torch.tensor([[0.0, l1, 0.0]], device=device, dtype=torch.float32).repeat(n, 1)
    knee_to_ankle = torch.tensor([[0.0, 0.0, -l2]], device=device, dtype=torch.float32).repeat(n, 1)
    ankle_to_foot = torch.tensor([[0.0, 0.0, -l3]], device=device, dtype=torch.float32).repeat(n, 1)

    hip_quat = quat_axis(torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(n, 1), q_tensor_3d[:, 0])
    hip2knee_quat = quat_axis(torch.tensor([[0.0, 1.0, 0.0]], device=device).repeat(n, 1), q_tensor_3d[:, 1])
    knee2ankle_quat = quat_axis(torch.tensor([[0.0, 1.0, 0.0]], device=device).repeat(n, 1), q_tensor_3d[:, 2])

    # knee_quat = quat_mul(hip2knee_quat, hip_quat)
    # ankle_quat = quat_mul(knee2ankle_quat, knee_quat)
    knee_quat = quat_mul(hip_quat, hip2knee_quat)
    ankle_quat = quat_mul(knee_quat, knee2ankle_quat)

    knee_pos = hip_pos + quat_rotate(hip_quat, hip_to_knee)
    ankle_pos = knee_pos + quat_rotate(knee_quat, knee_to_ankle)
    foot_pos = ankle_pos + quat_rotate(ankle_quat, ankle_to_foot)

    return hip_pos, knee_pos, ankle_pos, foot_pos


# test
if __name__ == "__main__":
    def _test_ik():
        l1 = 0.0776
        l2 = 0.25
        l3 = 0.235
        pos_tensor_3d = torch.tensor([[0.0, 0.09, -0.48]], dtype=torch.float32)
        is_ik_solved_tensor, q_tensor_3d = ik_3d_leg(l1, l2, l3, pos_tensor_3d)
        print("is_ik_solved_tensor: ", is_ik_solved_tensor)
        print("q_tensor_3d: ", q_tensor_3d)

    def _test_fk():
        n = 10
        range_max = [torch.pi/4, torch.pi/2, 0]
        range_min = [-torch.pi/4, -torch.pi/4, -torch.pi]
        random_q_tensor_3d = torch.zeros(n, 3)
        for i in range(3):
            random_q_tensor_3d[:, i] = torch.rand(n) * (range_max[i] - range_min[i]) + range_min[i]

        l1 = 0.0776
        l2 = 0.25
        l3 = 0.235
        hip_pos_tensor, knee_pos_tensor, ankle_pos_tensor, foot_pos_tensor = fk_3d_leg(l1, l2, l3, random_q_tensor_3d)

        # numpy version
        import numpy as np
        hip_pos_numpy_tensor = torch.zeros_like(hip_pos_tensor)
        knee_pos_numpy_tensor = torch.zeros_like(knee_pos_tensor)
        ankle_pos_numpy_tensor = torch.zeros_like(ankle_pos_tensor)
        foot_pos_numpy_tensor = torch.zeros_like(foot_pos_tensor)

        for i, q_tensor_3d in enumerate(random_q_tensor_3d):
            hip_angle, knee_angle, ankle_angle = q_tensor_3d.numpy()
            hip_rot_mat = np.array([[1, 0, 0], [0, np.cos(hip_angle), -np.sin(hip_angle)], [0, np.sin(hip_angle), np.cos(hip_angle)]])
            knee_rot_mat = np.array([[np.cos(knee_angle), 0, np.sin(knee_angle)], [0, 1, 0], [-np.sin(knee_angle), 0, np.cos(knee_angle)]])
            ankle_rot_mat = np.array([[np.cos(ankle_angle), 0, np.sin(ankle_angle)], [0, 1, 0], [-np.sin(ankle_angle), 0, np.cos(ankle_angle)]])

            hip_pos = np.array([0, 0, 0])
            knee_pos = hip_pos + hip_rot_mat @ np.array([0, l1, 0])
            ankle_pos = knee_pos + hip_rot_mat @ knee_rot_mat @ np.array([0, 0, -l2])
            foot_pos = ankle_pos + hip_rot_mat @ knee_rot_mat @ ankle_rot_mat @ np.array([0, 0, -l3])

            hip_pos_numpy_tensor[i] = torch.tensor(hip_pos)
            knee_pos_numpy_tensor[i] = torch.tensor(knee_pos)
            ankle_pos_numpy_tensor[i] = torch.tensor(ankle_pos)
            foot_pos_numpy_tensor[i] = torch.tensor(foot_pos)

        print("hip_diff: ", torch.mean(torch.norm(hip_pos_tensor - hip_pos_numpy_tensor, dim=1)))
        print("knee_diff: ", torch.mean(torch.norm(knee_pos_tensor - knee_pos_numpy_tensor, dim=1)))
        print("ankle_diff: ", torch.mean(torch.norm(ankle_pos_tensor - ankle_pos_numpy_tensor, dim=1)))
        print("foot_diff: ", torch.mean(torch.norm(foot_pos_tensor - foot_pos_numpy_tensor, dim=1)))

    def _test_ik_fk():
        n = 10000
        device_cuda = torch.device("cuda")
        device_cpu = torch.device("cpu")
        range_max = [torch.pi/4, torch.pi/2, 0]
        range_min = [-torch.pi/4, -torch.pi/4, -torch.pi]
        random_q_tensor_3d = torch.zeros(n, 3, device=device_cuda)
        for i in range(3):
            random_q_tensor_3d[:, i] = torch.rand(n, device=device_cuda) * (range_max[i] - range_min[i]) + range_min[i]

        l1 = 0.0776
        l2 = 0.25
        l3 = 0.235
        hip_pos, knee_pos, ankle_pos, foot_pos = fk_3d_leg(l1, l2, l3, random_q_tensor_3d)
        is_ik_solved_tensor, q_tensor_3d = ik_3d_leg(l1, l2, l3, foot_pos)
        if torch.any(~is_ik_solved_tensor):
            print("ik not solbed for ", torch.nonzero(~is_ik_solved_tensor).squeeze())
        norm = torch.mean(torch.norm(random_q_tensor_3d - q_tensor_3d, dim=1))
        print("mean_diff: ", norm)

    _test_ik()
    _test_fk()
    _test_ik_fk()
