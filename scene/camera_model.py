import torch
from torch import nn
import numpy as np

from utils.graphics_utils import getWorld2View2_tensor, getProjectionMatrix_tensor, r_2_R
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, PILtoTorch

'''
            class CameraInfo(NamedTuple):
            uid: int
            R: np.array # shape (3, 3)
            T: np.array # shape (3,)
            FovY: np.array
            FovX: np.array
            image: np.array
            image_path: str
            image_name: str
            width: int
            height: int
'''
class Camera_tensor(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0, camera_device = "cuda:0"
                 ):
        super(Camera_tensor, self).__init__()

        self.camera_device = camera_device

        self.uid = uid
        self.colmap_id = colmap_id # uid in CameraInfo

        self.R = R    # torch.tensor
        self.T = T    # torch.tensor

        self.T = nn.Parameter(self.T.to(camera_device).requires_grad_(True))
        
        # from vector r to build the computation graph
        self.r = nn.Parameter(torch.transpose(R[:, :2], 0, 1).flatten().to(camera_device).requires_grad_(True))
        
        # # Use vector r to compute the rotation matrix R
        # self.R = r_2_R(self.r, camera_device)
        
        self.FoVx = FoVx    
        self.FoVy = FoVy    

        self.image_name = image_name

        # try:
        #     self.camera_device = torch.device(camera_device)
        # except Exception as e:
        #     print(e)
        #     print(f"[Warning] Custom device {camera_device} failed, fallback to default cuda device" )
        #     self.camera_device = torch.device("cuda:0")

        # move the parameter to the device

        self.original_image = image.clamp(0.0, 1.0).to(camera_device)

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(camera_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=camera_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # self.world_view_transform = getWorld2View2_tensor(self.R, self.T, self.trans, self.scale).transpose(0, 1).to(camera_device)
        # self.projection_matrix = getProjectionMatrix_tensor(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(camera_device)
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]

    def forward(self):
        # Use vector r to compute the rotation matrix R
        self.R = r_2_R(self.r, self.camera_device)

        # compute the transformation and projection matrix
        self.world_view_transform = getWorld2View2_tensor(self.R, self.T, self.trans, self.scale).transpose(0, 1).to(self.camera_device)
        self.projection_matrix = getProjectionMatrix_tensor(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.camera_device)       # pay attention to the transpose
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        return self.world_view_transform, self.full_proj_transform, self.camera_center

# camerainfo_list is a list of sorted CameraInfo objects
# without the scale operation and 4-channel input compared with the original verison
class CameraModel(nn.Module):
    def __init__(self, args, resolution, camerainfo_list=[], trans=torch.tensor([0.0, 0.0, 0.0]), resolution_scale=1.0):
        super(CameraModel, self).__init__()

        self.camera_tensor_list = []
        
        self.camera_device = args.camera_device

        self.image_width = 0
        self.image_height = 0

        for i, camerainfo in enumerate(camerainfo_list):
            # print(camerainfo.image)
            # print(type(camerainfo.image))
            # print(resolution)
            # print(resolution_scale)
            gt_image, gt_alpha_mask = self.resize_image(resolution, camerainfo.image, resolution_scale)

            self.image_width = gt_image.shape[2]
            self.image_height = gt_image.shape[1]

            self.camera_tensor_list.append(Camera_tensor(colmap_id=camerainfo.uid, R=torch.from_numpy(camerainfo.R).float(), T=torch.from_numpy(camerainfo.T).float(), 
                  FoVx=camerainfo.FovX, FoVy=camerainfo.FovY, 
                  image=gt_image, gt_alpha_mask=gt_alpha_mask,
                  image_name=camerainfo.image_name, uid=i, 
                  trans=trans, scale=resolution_scale, camera_device=self.camera_device))

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = resolution_scale
        self.length = len(camerainfo_list)

        self.optimizer = None
    
    '''
    modified from ../utils/camera_utils.loadCam(args, id, cam_info, resolution_scale)
    '''
    def resize_image(self, resolution, image, resolution_scale=1.0):
        orig_w, orig_h = image.size

        if resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/(resolution_scale * resolution)), round(orig_h/(resolution_scale * resolution))
        else:  # should be a type that converts to float
            if resolution == -1:
                if orig_w > 1600:
                    global WARNED
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / resolution

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))

        resized_image_rgb = PILtoTorch(image, resolution)

        gt_image = resized_image_rgb[:3, ...]
        loaded_mask = None
        
        # For 4 channel images, the 4th channel is the alpha mask, loaded as loaded_mask
        if resized_image_rgb.shape[1] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
        
        # print(gt_image.shape, loaded_mask.shape)
        return gt_image, loaded_mask
    

    def get_cameras(self, uid_list=[]):
        camera_tensor_selected_list = []
        for uid in uid_list:
            camera_tensor_selected_list.append(self.camera_tensor_list[uid])
        
        return camera_tensor_selected_list
    
    # initialize the learning rate of different parameters, and pass the parameters to optimizer
    def training_setup(self, training_args, uid_list=[]):
        l = []
        for uid in uid_list:
            camera_tensor_selected = self.camera_tensor_list[uid]
            # print(camera_tensor_selected.T)
            # print(camera_tensor_selected.r)
            l.append({'params': [camera_tensor_selected.T], 'lr': training_args.t_lr_init, "name": "T_"+str(camera_tensor_selected.uid)})
            l.append({'params': [camera_tensor_selected.r], 'lr': training_args.r_lr_init, "name": "r_"+str(camera_tensor_selected.uid)})
        
        # define the optimizer and pass the parameters to optimizer
        self.optimizer = torch.optim.Adam(l, lr=1e-3, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        pass
    
    # def update_pose(self, uid_list=[]):
    #     for uid in uid_list:
    #         camera_tensor_selected = self.camera_tensor_list[uid]
    #         # Use vector r to compute the rotation matrix R
    #         camera_tensor_selected.R = r_2_R(camera_tensor_selected.r, self.camera_device)

    #         camera_tensor_selected.world_view_transform = getWorld2View2_tensor(camera_tensor_selected.R, camera_tensor_selected.T, camera_tensor_selected.trans, camera_tensor_selected.scale).transpose(0, 1)
    #         camera_tensor_selected.projection_matrix = getProjectionMatrix_tensor(znear=camera_tensor_selected.znear, zfar=camera_tensor_selected.zfar, fovX=camera_tensor_selected.FoVx, fovY=camera_tensor_selected.FoVy).transpose(0,1)
    #         camera_tensor_selected.full_proj_transform = (camera_tensor_selected.world_view_transform.unsqueeze(0).bmm(camera_tensor_selected.projection_matrix.unsqueeze(0))).squeeze(0)
    #         camera_tensor_selected.camera_center = camera_tensor_selected.world_view_transform.inverse()[3, :3]

    #         return camera_tensor_selected.R.cpu(), camera_tensor_selected.T.data.cpu()

