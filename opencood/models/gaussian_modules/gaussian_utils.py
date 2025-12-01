import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from typing import List, Optional, NamedTuple


SIGMOID_MAX = 9.21024
LOGIT_MAX = 0.9999

class GaussianPrediction(NamedTuple):
    means: Tensor
    scales: Tensor
    rotations: Tensor
    opacities: Tensor
    semantics: Tensor
    original_means: Tensor = None
    delta_means: Tensor = None
    mask: Tensor = None


def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -9.21, 9.21)
    return torch.sigmoid(tensor)


def safe_inverse_sigmoid(tensor):
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))


def cartesian(anchor, pc_range, use_sigmoid=True):
    if use_sigmoid:
        xyz = safe_sigmoid(anchor[..., :3])
    else:
        xyz = anchor[..., :3].clamp(min=1e-6, max=1 - 1e-6)
    xxx = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    yyy = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    zzz = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack([xxx, yyy, zzz], dim=-1)

    return xyz


def get_meshgrid(ranges, grid, reso):
    if isinstance(reso, float):
        reso = [reso] * 3
    xxx = torch.arange(grid[0], dtype=torch.float) * reso[0] + 0.5 * reso[0] + ranges[0]
    yyy = torch.arange(grid[1], dtype=torch.float) * reso[1] + 0.5 * reso[1] + ranges[1]
    zzz = torch.arange(grid[2], dtype=torch.float) * reso[2] + 0.5 * reso[2] + ranges[2]

    xxx = xxx[:, None, None].expand(*grid)
    yyy = yyy[None, :, None].expand(*grid)
    zzz = zzz[None, None, :].expand(*grid)

    xyz = torch.stack([
        xxx, yyy, zzz
    ], dim=-1).numpy()
    return xyz  # x, y, z, 3


def reverse_cartesian(xyz, pc_range, use_sigmoid=True):
    xxx = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    yyy = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    zzz = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
    unitxyz = torch.stack([xxx, yyy, zzz], dim=-1)
    if use_sigmoid:
        anchor = safe_inverse_sigmoid(unitxyz)
    else:
        anchor = unitxyz.clamp(min=1e-6, max=1-1e-6)
    return anchor


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def get_rotation_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    assert quaternion.shape[-1] == 4, "Quaternion must have last dim size 4 (format: [w, x, y, z])"
    quaternion = F.normalize(quaternion, dim=-1)

    w, x, y, z = quaternion.unbind(dim=-1)  # Each has shape (...,)

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rot = torch.empty(*quaternion.shape[:-1], 3, 3, dtype=quaternion.dtype, device=quaternion.device)

    rot[..., 0, 0] = 1 - 2 * (yy + zz)
    rot[..., 0, 1] = 2 * (xy - wz)
    rot[..., 0, 2] = 2 * (xz + wy)

    rot[..., 1, 0] = 2 * (xy + wz)
    rot[..., 1, 1] = 1 - 2 * (xx + zz)
    rot[..., 1, 2] = 2 * (yz - wx)

    rot[..., 2, 0] = 2 * (xz - wy)
    rot[..., 2, 1] = 2 * (yz + wx)
    rot[..., 2, 2] = 1 - 2 * (xx + yy)

    return rot


def get_quaternion_from_rotation(R: torch.Tensor) -> torch.Tensor:
    batch_dims = R.shape[:-2]
    R = R.view(-1, 3, 3)

    m00 = R[:, 0, 0]
    m01 = R[:, 0, 1]
    m02 = R[:, 0, 2]
    m10 = R[:, 1, 0]
    m11 = R[:, 1, 1]
    m12 = R[:, 1, 2]
    m20 = R[:, 2, 0]
    m21 = R[:, 2, 1]
    m22 = R[:, 2, 2]

    qw = torch.empty(R.shape[0], device=R.device, dtype=R.dtype)
    qx = torch.empty_like(qw)
    qy = torch.empty_like(qw)
    qz = torch.empty_like(qw)

    cond = (m00 > m11) & (m00 > m22)
    cond2 = (m11 > m22)

    idx = cond
    S = torch.sqrt(1.0 + m00[idx] - m11[idx] - m22[idx] + 1e-8) * 2
    qw[idx] = (m21[idx] - m12[idx]) / S
    qx[idx] = 0.25 * S
    qy[idx] = (m01[idx] + m10[idx]) / S
    qz[idx] = (m02[idx] + m20[idx]) / S

    idx = ~cond & cond2
    S = torch.sqrt(1.0 + m11[idx] - m00[idx] - m22[idx] + 1e-8) * 2
    qw[idx] = (m02[idx] - m20[idx]) / S
    qx[idx] = (m01[idx] + m10[idx]) / S
    qy[idx] = 0.25 * S
    qz[idx] = (m12[idx] + m21[idx]) / S

    idx = ~cond & ~cond2 & (m22 >= m00) & (m22 >= m11)
    S = torch.sqrt(1.0 + m22[idx] - m00[idx] - m11[idx] + 1e-8) * 2
    qw[idx] = (m10[idx] - m01[idx]) / S
    qx[idx] = (m02[idx] + m20[idx]) / S
    qy[idx] = (m12[idx] + m21[idx]) / S
    qz[idx] = 0.25 * S

    idx = ~cond & ~cond2 & ~((m22 >= m00) & (m22 >= m11))
    S = torch.sqrt(1.0 + m00[idx] + m11[idx] + m22[idx] + 1e-8) * 2
    qw[idx] = 0.25 * S
    qx[idx] = (m21[idx] - m12[idx]) / S
    qy[idx] = (m02[idx] - m20[idx]) / S
    qz[idx] = (m10[idx] - m01[idx]) / S

    quat = torch.stack([qw, qx, qy, qz], dim=-1)
    quat = F.normalize(quat, dim=-1)

    return quat.view(*batch_dims, 4)


def scale_rotation_to_covariance(scale, rotation):
    B = scale.shape[0]
    R = get_rotation_from_quaternion(rotation)  # (..., 3, 3)
    S = torch.zeros(B, 3, 3, dtype=scale.dtype, device=scale.device)
    S[..., 0, 0] = scale[..., 0]
    S[..., 1, 1] = scale[..., 1]
    S[..., 2, 2] = scale[..., 2]
    M = torch.matmul(S, R)  # ← wrong order
    Cov = torch.matmul(M.transpose(-1, -2), M)
    return Cov


def covariance_to_scale_rotation(cov):
    eigvals, eigvecs = torch.linalg.eigh(cov)  # (..., 3), (..., 3, 3)
    scale = eigvals.clamp(min=1e-6, max=1e2).sqrt()
    det = torch.linalg.det(eigvecs)  # shape (B,)
    neg_mask = det < 0  # shape (B,)
    sign = torch.ones_like(eigvecs)
    sign[neg_mask, :, 0] = -1
    eigvecs = eigvecs * sign
    rotation = get_quaternion_from_rotation(eigvecs.transpose(-1, -2))
    return scale, rotation


# # Copyright (c) OpenMMLab. All rights reserved.


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
