#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

############################################################################################

# recover Rotation matrix from vector r
def r_2_R(r, device):
    r_t = torch.transpose(r.reshape(2, 3), 0, 1).to(device)
    a1 = r_t[:, 0]
    a2 = r_t[:, 1]
    b1 = a1 / torch.norm(a1, p=2)
    b2 = a2-torch.dot(b1, a2)*b1 / torch.norm(a2-torch.dot(b1, a2)*b1, p=2)               # N(a2−(b1· a2)b1)
    b3 = torch.cross(b1, b2)    

    return torch.transpose(torch.cat((b1, b2, b3), 0).reshape(3, 3), 0, 1)                # b3 = b1 × b2

def getWorld2View2_tensor(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    # Create a 4x4 identity matrix as a torch tensor
    Rt = torch.eye(4, dtype=torch.float32)

    # Set the upper-left 3x3 block of Rt to the transpose of R
    Rt[:3, :3] = R.t()

    # Set the first three elements of the last column of Rt to t
    Rt[:3, 3] = t

    # Set Rt[3, 3] to 1.0
    Rt[3, 3] = 1.0

    # Compute the inverse of C2W using torch's inverse function
    C2W = torch.inverse(Rt)

    # Extract the camera center from C2W
    cam_center = C2W[:3, 3]

    # Apply translation and scaling to the camera center
    cam_center = (cam_center + translate) * scale

    # Update the camera center in C2W
    C2W[:3, 3] = cam_center

    # Compute the inverse of the updated C2W
    Rt = torch.inverse(C2W)

    # Convert the result to a 32-bit floating-point tensor
    return Rt.float()

def getProjectionMatrix_tensor(znear, zfar, fovX, fovY):
    """
    Calculates the projection matrix for a given set of parameters.

    Args:
        znear (float): The distance to the near clipping plane.
        zfar (float): The distance to the far clipping plane.
        fovX (float): The horizontal field of view angle in radians.
        fovY (float): The vertical field of view angle in radians.

    Returns:
        torch.Tensor: The projection matrix as a 4x4 tensor.
    """
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P