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
import torch
# from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, OptimizationCamerasParams
import json
import pdb
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

'''
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
opt_c = OptimizationCamerasParams(parser)
training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, 
        args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
'''

def optimize_incrementally(gaussians, scene, cameramodel, background, tb_writer, dataset, opt, opt_c, pipe, testing_iterations, saving_iterations, checkpoint_iterations, debug_from):
    first_iter = 0
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Incrementally Training Progress")
    
    iteration = 1
    for iteration_c in range(first_iter, cameramodel.length-1):
        # get two camera objects
        viewpoint_cam1, viewpoint_cam2 = cameramodel.get_cameras([iteration_c, iteration_c+1])

        # optimize the local gaussian in the first view
        # get the first camera object
        viewpoint_cam = viewpoint_cam1
        # TODO
        cameramodel.training_setup(opt_c, [iteration_c])
        for iteration_one_view in range(first_iter+1, opt_c.iterations_one_view+1):
            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            # pipe.debug = True

            bg = torch.rand((3), device="cuda:0") if opt.random_background else background

            # pdb.set_trace()

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.to('cuda:0')
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

            iter_end.record()

            # print('Gradient of the pose:')
            # print(cameramodel.optimizer.param_groups[0]['params'][0].grad)

            # print('Gradient of others in the first camera:')
            # # print(viewpoint_cam.r.grad)
            # # print(viewpoint_cam.T.grad)
            # # print(viewpoint_cam.R.grad)
            # # print(viewpoint_cam.world_view_transform.grad)
            # # print(viewpoint_cam.full_proj_transform.grad)

            # print(viewpoint_cam.r)
            # print(viewpoint_cam.T)
            # print(viewpoint_cam.R)
            # print(viewpoint_cam.world_view_transform)
            # print(viewpoint_cam.full_proj_transform)

            # # TODO: test if rendering the viewpoint2 can be done here
            # loss_pose = 0

            # if iteration_one_view % 200 == 0:
            #     # try:

            #     viewpoint_cam_test = viewpoint_cam2
            #     # print('Camera pose optimization: ', iteration_pose)

            #     bg = torch.rand((3), device="cuda:0") if opt.random_background else background

            #     # torch.cuda.empty_cache()
            #     # print(pipe.debug)
            #     render_pkg = render(viewpoint_cam_test, gaussians, pipe, bg)
            #     image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            #     # except:
            #     #     import pdb; pdb.set_trace()
            #     import cv2
            #     print("writing image ............")
            #     with torch.no_grad():
            #         cv2.imwrite("test.png", image.cpu().numpy().transpose(1, 2, 0) * 255)


        
            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and (iteration-1) % opt_c.iterations_one_view == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
                # add 1 to iteration
                iteration = iteration + 1
        
        # # optimize the pose of the second camera based on the gaussian
        # print('The initial pose of the second camera:')
        # print('T: ', viewpoint_cam2.T.data.cpu())
        # print('R: ', viewpoint_cam2.R.cpu())
        
        # print(iteration_c)

        cameramodel.training_setup(opt_c, [iteration_c+1])
        # get the second camera object
        viewpoint_cam = viewpoint_cam2
        # pipe.debug = True
        for iteration_pose in range(first_iter+1, opt_c.iterations_pose+1):

            # print('Camera pose optimization: ', iteration_pose)

            bg = torch.rand((3), device="cuda:0") if opt.random_background else background

            # torch.cuda.empty_cache()
            # print(pipe.debug)
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.to('cuda:0')
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            
            # if(iteration_pose == 2):
            #     import cv2
            #     print("writing image ............")
            #     with torch.no_grad():
            #         cv2.imwrite("image.png", image.cpu().numpy().transpose(1, 2, 0) * 255)
            #         cv2.imwrite('image_gt.png', gt_image.cpu().numpy().transpose(1, 2, 0) * 255)

            # print('loss for the pose:', loss.item())

            # with torch.no_grad():
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            # # print(cameramodel.optimizer.param_groups[0]['params'])
            
            # print('Gradient of the pose:')
            # print(cameramodel.optimizer.param_groups[0]['params'][0].grad)

            # print('Gradient of others:')
            # print(viewpoint_cam.r.grad)
            # print(viewpoint_cam.T.grad)
            # print(viewpoint_cam.R.grad)
            # print(viewpoint_cam.world_view_transform.grad)
            # print(viewpoint_cam.full_proj_transform.grad)

            # # Optimizer step
            # cameramodel.optimizer.step()
            # # gaussians.optimizer.step()
            # cameramodel.optimizer.zero_grad(set_to_none = False)
            # # gaussians.optimizer.zero_grad(set_to_none = True)
            
            # # update other parameters in the camera after updating r vector
            # # cameramodel.update_pose([iteration_c+1])
            # print('The second camera pose optimization:')
            # print('T: ', viewpoint_cam.T.data.cpu())
            # print('R: ', viewpoint_cam.R.cpu())

            # # print(viewpoint_cam.r)
            # # print(viewpoint_cam.T)





def optimize_joitly():
    pass

'''
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
training(lp.extract(args), op.extract(args), opt_c.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, 
        args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
'''
def training(dataset, opt, opt_c, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0

    # define Tensorboard writer
    tb_writer = prepare_output_and_logger(dataset)
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, opt_c, gaussians)
    gaussians.training_setup(opt)

    # load the checkpoint
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    # background
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda:0")

    cameramodel = scene.getTrainCameras()

    # TODO start the training incrementally
    optimize_incrementally(gaussians, scene, cameramodel, background, tb_writer, dataset, opt, opt_c, pipe, testing_iterations, saving_iterations, checkpoint_iterations, debug_from)

    # iter_start = torch.cuda.Event(enable_timing = True)
    # iter_end = torch.cuda.Event(enable_timing = True)

    # ema_loss_for_log = 0.0
    # progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    # first_iter += 1
    # for iteration in range(first_iter, opt.iterations + 1):        
    #     # if network_gui.conn == None:
    #     #     network_gui.try_connect()
    #     # while network_gui.conn != None:
    #     #     try:
    #     #         net_image_bytes = None
    #     #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
    #     #         if custom_cam != None:
    #     #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
    #     #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
    #     #         network_gui.send(net_image_bytes, dataset.source_path)
    #     #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
    #     #             break
    #     #     except Exception as e:
    #     #         network_gui.conn = None

    #     iter_start.record()

    #     gaussians.update_learning_rate(iteration)

    #     # Every 1000 its we increase the levels of SH up to a maximum degree
    #     if iteration % 1000 == 0:
    #         gaussians.oneupSHdegree()

        
    #     if not viewpoint_stack:
    #         # get a list of Camera objects in a specific resolution(here use the default resolution 1.0)
    #         viewpoint_stack = scene.getTrainCameras().copy()
        
    #     # Pick a random Camera object from the list
    #     viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

    #     # Render
    #     if (iteration - 1) == debug_from:
    #         pipe.debug = True

    #     bg = torch.rand((3), device="cuda") if opt.random_background else background

    #     render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
    #     image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

    #     # Loss
    #     gt_image = viewpoint_cam.original_image.cuda()
    #     Ll1 = l1_loss(image, gt_image)
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    #     loss.backward()

    #     iter_end.record()

    #     with torch.no_grad():
    #         # Progress bar
    #         ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
    #         if iteration % 10 == 0:
    #             progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
    #             progress_bar.update(10)
    #         if iteration == opt.iterations:
    #             progress_bar.close()

    #         # Log and save
    #         training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
    #         if (iteration in saving_iterations):
    #             print("\n[ITER {}] Saving Gaussians".format(iteration))
    #             scene.save(iteration)

    #         # Densification
    #         if iteration < opt.densify_until_iter:
    #             # Keep track of max radii in image-space for pruning
    #             gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
    #             gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

    #             if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
    #                 size_threshold = 20 if iteration > opt.opacity_reset_interval else None
    #                 gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
    #             if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
    #                 gaussians.reset_opacity()

    #         # Optimizer step
    #         if iteration < opt.iterations:
    #             gaussians.optimizer.step()
    #             gaussians.optimizer.zero_grad(set_to_none = True)

    #         if (iteration in checkpoint_iterations):
    #             print("\n[ITER {}] Saving Checkpoint".format(iteration))
    #             torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Get the current process ID
    pid = os.getpid()

    # Print the process ID
    print("The PID of the current Python process is:", pid)

    # python train.py
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    op_c = OptimizationCamerasParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    '''
    The sys.argv[1:] part is getting all command line arguments passed to the script 
    (excluding the script name itself, which is the first argument sys.argv[0]). 
    The parse_args method then parses these arguments based on the arguments that 
    were added to the ArgumentParser object.
    '''
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # params = {}
    # for group in parser._action_groups:
    #     group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
    #     params[group.title] = group_dict

    # with open('./params_group.json', mode='w') as f:
    #     json.dump(params, f, indent=4)

    # with open('./params.json', mode='w') as f:
    #     json.dump(args.__dict__, f, indent=4)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    
    # The set_detect_anomaly function enables or disables the anomaly detection mode in Autograd.
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # start training
    training(lp.extract(args), op.extract(args), op_c.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

    params = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        params[group.title] = group_dict

    with open('./params_group.json', mode='w') as f:
        json.dump(params, f, indent=4)

    with open('./params.json', mode='w') as f:
        json.dump(args.__dict__, f, indent=4)


# # 9D embed:
        
# def pose_to_d9(pose: torch.Tensor) -> torch.Tensor:
#     """Converts rotation matrix to 9D representation. 

#     We take the two first ROWS of the rotation matrix, 
#     along with the translation vector.
#     ATTENTION: row or vector needs to be consistent from pose_to_d9 and r6d2mat
#     """
#     nbatch = pose.shape[0]
#     R = pose[:, :3, :3]  # [N, 3, 3]
#     t = pose[:, :3, -1]  # [N, 3]

#     r6 = R[:, :2, :3].reshape(nbatch, -1)  # [N, 6]

#     d9 = torch.cat((t, r6), -1)  # [N, 9]
#     # first is the translation vector, then two first ROWS of rotation matrix

#     return d9

# import torch.nn.functional as F

# def r6d2mat(d6: torch.Tensor) -> torch.Tensor:
#     """
#     Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
#     using Gram--Schmidt orthogonalisation per Section B of [1].
#     Args:
#         d6: 6D rotation representation, of size (*, 6). Here corresponds to the two
#             first two rows of the rotation matrix. 
#     Returns:
#         batch of rotation matrices of size (*, 3, 3)
#     [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
#     On the Continuity of Rotation Representations in Neural Networks.
#     IEEE Conference on Computer Vision and Pattern Recognition, 2019.
#     Retrieved from http://arxiv.org/abs/1812.07035
#     """

#     a1, a2 = d6[..., :3], d6[..., 3:]
#     b1 = F.normalize(a1, dim=-1)
#     b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
#     b2 = F.normalize(b2, dim=-1)
#     b3 = torch.cross(b1, b2, dim=-1)
#     return torch.stack((b1, b2, b3), dim=-2)  # corresponds to row