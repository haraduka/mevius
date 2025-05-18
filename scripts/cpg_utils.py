import os
import pprint
import numpy as np
import torch
from analytical_ik import ik_3d_leg
import mevius_utils
import parameters as P


class CpgUtils:
    def __init__(self):
        self.num_envs = 1
        self.default_base_freq = 1.8
        self.foot_height = 0.13
        self.foot_default_positions = [[-0.10155,  0.0960, -0.3663],
                                       [-0.10155, 0.0960, -0.3663],
                                       [-0.00745,  0.0960, -0.3647],
                                       [-0.00745, 0.0960, -0.3647]]
        self.foot_default_positions = torch.tensor(self.foot_default_positions, requires_grad=False).view(4, 3)
        self.l1 = 0.0776
        self.l2 = 0.25
        self.l3 = 0.235
        self.default_dof_pos = torch.tensor(P.DEFAULT_ANGLE[:])

        self.foot_pos_ref = torch.zeros(self.num_envs, 4, 3, requires_grad=False)
        self.joint_ref = torch.zeros(self.num_envs, 12, requires_grad=False)
        self._base_freq = torch.ones(self.num_envs, 1, requires_grad=False) * self.default_base_freq
        self._phases = torch.zeros(self.num_envs, 4, requires_grad=False)
        self._dphases = torch.zeros(self.num_envs, 4, requires_grad=False)

        # reset
        self._phases[:] = 0.0
        self._base_freq[:] = 0.0
        self._reset_dphase()

    @property
    def phases(self):
        return self._phases * 2 * torch.pi

    @property
    def dphases(self):
        return self._dphases * 2 * torch.pi / P.control.decimation

    @property
    def base_phase_step(self):
        return self._base_freq * P.control.dt * P.control.decimation

    def _round_phase(self):
        self._phases = self._phases % 1.0

    def _reset_dphase(self):
        self._dphases[:] = self.base_phase_step

    def reset_trot(self):
        self._phases[:] = 0.0
        self._phases[:, 1:3] = 0.5
        self._base_freq[:] = self.default_base_freq
        self._reset_dphase()

    def reset_stance(self):
        self._phases[:] = 0.0
        self._base_freq[:] = self.default_base_freq
        self._reset_dphase()

    def _get_cubic_height(self):
        height = torch.zeros_like(self._phases)
        t = self._phases * 4.0 # 0.0 <= t < 4.0
        mask_swing = self._phases < 0.5 # 0.0 <= t < 2.0, swing phase
        mask_lift = (t < 1.0) & mask_swing # 0.0 <= t < 1.0, lift phase
        mask_drop = (t >= 1.0) & mask_swing # 1.0 <= t < 2.0, drop phase
        height += mask_lift * (t * t * (-2 * t + 3))
        t -= 1
        height += mask_drop * (t * t * (2 * t - 3) + 1)
        return height * self.foot_height

    def _update_base_freq(self, is_standing):
        if is_standing:
            self._base_freq[:] = 0.0
        else:
            self._base_freq[:] = self.default_base_freq

    def _update_phase(self, dphase):
        self._dphases = self.base_phase_step + dphase * P.control.decimation / (2 * torch.pi)
        self._phases += self._dphases
        self._round_phase()

    def compute_actions(self, actions_, is_standing):
        actions = torch.clip(actions_, -P.control.action_clipping, P.control.action_clipping)
        scaled_actions = actions * P.control.action_scale
        self._update_base_freq(is_standing)
        # dphase = torch.zeros(1, 4)
        dphase = scaled_actions[:, :4]
        self._update_phase(dphase)
        heights = self._get_cubic_height()

        self.foot_pos_ref[:] = self.foot_default_positions
        self.foot_pos_ref[:, :, 2] += heights

        for i in range(4):
            _, joint_ref = ik_3d_leg(self.l1, self.l2, self.l3, self.foot_pos_ref[:, i])
            if i in [1, 3]:
                joint_ref[:, 0] = -joint_ref[:, 0]
            self.joint_ref[:, i*3:(i+1)*3] = joint_ref
        self.joint_ref += scaled_actions[:, 4:]
        return self.joint_ref - self.default_dof_pos

if __name__ == "__main__":
    cpg = CpgUtils()
    angle_list = []
    for i in range(50):
        angle = cpg.compute_actions(torch.zeros(cpg.num_envs, 16), True)[0]
        angle = angle.detach().cpu().numpy()
        angle_list.append(angle)
    for i in range(300):
        angle = cpg.compute_actions(torch.zeros(cpg.num_envs, 16), False)[0]
        angle = angle.detach().cpu().numpy()
        angle_list.append(angle)
    angle_list = np.array(angle_list)
    print(angle_list.shape)

    # write to file
    path = "/tmp/angle_list.pkl"
    with open(path, "wb") as f:
        torch.save(angle_list, f)

