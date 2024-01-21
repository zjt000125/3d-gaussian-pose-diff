import math
import numpy as np
import torch
from torch import nn
from utils.graphics_utils import getWorld2View2_tensor, getProjectionMatrix_tensor, r_2_R
from scene.dataset_readers import fetchPly, fetchPly_from_ouput, storePly

# # FoVx = np.array(0.56)
# # FoVy = np.array(0.96)

# FoVx = torch.tensor(0.56).cuda()
# FoVy = torch.tensor(0.96).cuda()

# print(type(FoVx), type(FoVy))
# print(FoVx, FoVy)

# tanfovx = math.tan(FoVx * 0.5)
# tanfovy = math.tan(FoVy * 0.5)

# print(type(tanfovx), type(tanfovy))
# print(tanfovx, tanfovy)

# def r_2_R(r):
#     r_t = torch.transpose(r.reshape(2, 3), 0, 1)
#     a1 = r_t[:, 0]
#     a2 = r_t[:, 1]
#     b1 = a1 / torch.norm(a1, p=2)
#     b2 = a2-torch.dot(b1, a2)*b1 / torch.norm(a2-torch.dot(b1, a2)*b1, p=2)               # N(a2−(b1· a2)b1)
#     b3 = torch.cross(b1, b2)    

#     return torch.transpose(torch.cat((b1, b2, b3), 0).reshape(3, 3), 0, 1)                # b3 = b1 × b2

# # Generate random angles for rotation
# angle_x = torch.rand(1) * 2 * math.pi  # Random angle around x-axis
# angle_y = torch.rand(1) * 2 * math.pi  # Random angle around y-axis
# angle_z = torch.rand(1) * 2 * math.pi  # Random angle around z-axis

# # Create rotation matrices for each axis
# rotation_matrix_x = torch.tensor([
#     [1, 0, 0],
#     [0, math.cos(angle_x), -math.sin(angle_x)],
#     [0, math.sin(angle_x), math.cos(angle_x)]
# ], dtype=torch.float32)

# rotation_matrix_y = torch.tensor([
#     [math.cos(angle_y), 0, math.sin(angle_y)],
#     [0, 1, 0],
#     [-math.sin(angle_y), 0, math.cos(angle_y)]
# ], dtype=torch.float32)

# rotation_matrix_z = torch.tensor([
#     [math.cos(angle_z), -math.sin(angle_z), 0],
#     [math.sin(angle_z), math.cos(angle_z), 0],
#     [0, 0, 1]
# ], dtype=torch.float32)

# # Combine the rotation matrices to get the final random rotation matrix
# random_rotation_matrix = torch.mm(rotation_matrix_z, torch.mm(rotation_matrix_y, rotation_matrix_x))

# print(random_rotation_matrix)

# # R = torch.tensor([i for i in range(9)]).reshape(3, 3)

# # print(R)

# r = torch.transpose(random_rotation_matrix[:, :2], 0, 1).flatten()

# print(r)

# R = r_2_R(r)

# print(R)

# R = torch.tensor([i for i in range(9)]).reshape(3, 3).float()

# print(R)

# r = nn.Parameter(torch.transpose(R[:, :2], 0, 1).flatten().requires_grad_(True))
        
# # Use vector r to compute the rotation matrix R
# R = r_2_R(r, 'cpu')

# print(r)

# print(R)

path_output = './data/lego/point_cloud.ply'
path_input = './data/lego/points3D.ply'

pcd = fetchPly_from_ouput(path_output)

print(pcd)

