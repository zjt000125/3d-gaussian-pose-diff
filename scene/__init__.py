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

import os
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, OptimizationCamerasParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

from scene.camera_model import CameraModel

class Scene:

    gaussians : GaussianModel
    
    '''
    scene = Scene(lp.extract(args), GaussianModel(dataset.sh_degree))
    '''
    def __init__(self, args : ModelParams, opt_c : OptimizationCamerasParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found transforms.json file, assuming customer data set!")
            scene_info = sceneLoadTypeCallbacks["Customer"](args.source_path, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
        
        # # shuffle cameras
        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        #     random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        # load cameras in different resolutions
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            # modify to initilize CameraModel objects
            self.train_cameras[resolution_scale] = CameraModel(opt_c, args.resolution, scene_info.train_cameras, trans=torch.tensor([0.0, 0.0, 0.0]), resolution_scale=resolution_scale)   # scene_info.train_cameras is a list of shuffled CameraInfo objects
            # print("Loading Test Cameras")
            # self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

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

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        # return a list of Camera objects in a specific resolution
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]