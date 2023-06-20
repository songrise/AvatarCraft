#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
Render 360 views of a Da-posed human.
Render 360 views of a posed human.

Examples:
python render_360.py --scene_dir ./data/seattle --use_cuda=no --white_bkg=yes --rays_per_batch=2048 --trajectory_resolution=40 --weights_path ./out/seattle_human/checkpoint.pth.tar --render_h=72 --render_w=128 --mode canonical_360 --can_posenc rotate
python render_360.py --scene_dir ./data/seattle --use_cuda=no --white_bkg=yes --rays_per_batch=2048 --trajectory_resolution=40 --weights_path ./out/seattle_human/checkpoint.pth.tar --render_h=72 --render_w=128 --mode posed_360 --can_posenc rotate
'''
import os
import argparse

import imageio
import torch
import cv2

import numpy as np

from cameras.captures import ResizedPinholeCapture
from cameras.pinhole_camera import PinholeCamera

from utils import render_utils, utils
from options import options
from utils.constant import CANONICAL_ZOOM_FACTOR, CANONICAL_CAMERA_DIST_VAL, WHITE_BKG, BLACK_BKG, NOISE_BKG, NSR_BOUND, CAN_HEAD_OFFSET, CAN_HEAD_CAMERA_DIST
import pickle
import einops
import models.instant_nsr as instant_nsr


# CANONICAL_CAMERA_DIST_VAL = 1.3 # for larger human in paper
CANONICAL_CAMERA_DIST_VAL = 1.7  # for supplementary video


def main_canonical_360(opt):
    """
    360 degree views of a canonical avatar.
    """

    # load model
    nerf = instant_nsr.NeRFNetwork()
    nerf.load_state_dict(torch.load(opt.weights_path))

    center, up = np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    body_poses, _ = render_utils.default_360_path(
        center, up, CANONICAL_CAMERA_DIST_VAL, opt.trajectory_resolution)
    # head poses
    head_offset = np.array([0.0, 1.0, 0.0]).astype(np.float32)
    head_offset = head_offset * CAN_HEAD_OFFSET
    head_poses, _ = render_utils.default_360_path(
        center + head_offset, up, CAN_HEAD_CAMERA_DIST, opt.trajectory_resolution)
    # render multiple views of the body and head
    for pose_name, render_poses in zip(['body', 'head'], [body_poses, head_poses]):
        log_extrinsics = []
        log_imgs = []
        log_depths = []
        for i, rp in enumerate(render_poses):
            can_cap = ResizedPinholeCapture(
                PinholeCamera(
                    opt.render_w,
                    opt.render_h,
                    CANONICAL_ZOOM_FACTOR * opt.render_w,
                    CANONICAL_ZOOM_FACTOR * opt.render_h,
                    opt.render_w / 2.0,
                    opt.render_h / 2.0,
                ),
                rp,
                tgt_size=opt.render_size
            )
            if opt.implicit_model == "instant_nsr":
                # for nsr, it must run on gpu
                rays_o, rays_d = render_utils.cap2rays(can_cap)
                rays_o, rays_d = rays_o.cuda(), rays_d.cuda()
                nerf = nerf.cuda().eval()
                out, _, extra_out = render_utils.render_instantnsr_naive(
                    nerf, rays_o, rays_d, opt.batch_size,
                    requires_grad=False, bkg_key=WHITE_BKG if opt.white_bkg else BLACK_BKG,
                    return_torch=True, perturb=False, return_raw=True, render_can=True)

                out = out.detach().cpu().numpy()
                out = einops.repeat(out, '(h w) c -> h w c',
                                    h=opt.render_h, w=opt.render_w)
                if opt.log_extra:
                    depth = extra_out['depth'].detach().cpu().numpy()
                    depth = einops.repeat(
                        depth, '(h w) 1 -> h w 1', h=opt.render_h, w=opt.render_w)
                    mask = depth < 4e-1
                    depth[mask] = 0.45
                    # normalize to 0-1
                    depth = (depth - depth.min())/(depth.max()-depth.min())
                    depth = depth * 255
                    depth = depth.astype(np.uint8)
                    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
                    depth[mask.repeat(3, axis=2)] = 0
                    log_depths.append(depth)
                    save_path_depth = os.path.join(
                        './demo', 'canonical_360', opt.exp_name, f'{opt.exp_name}_{pose_name}_can_{str(i).zfill(4)}_depth.png')

                    if not os.path.isdir(os.path.dirname(save_path_depth)):
                        os.makedirs(os.path.dirname(save_path_depth))
                    # todo dump raw depth
                    # save_path_pkl = os.path.join('./demo', 'canonical_360', opt.exp_name, f'{opt.exp_name}_{pose_name}_can_{str(i).zfill(4)}_depth.pkl')
                    # with open(save_path_pkl, 'wb') as f:
                    #     pickle.dump(depth, f)

                    # depth = utils.integerify_img(depth)
                    imageio.imsave(save_path_depth, depth)

            log_extrinsics.append(can_cap.cam_pose.camera_to_world)
            save_path = os.path.join('./demo', 'canonical_360', opt.exp_name,
                                     f'{opt.exp_name}_{pose_name}_can_{str(i).zfill(4)}.png')
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            out = utils.integerify_img(out)
            log_imgs.append(out)
            imageio.imsave(save_path, out)
            print(f'image saved: {save_path}')
        # save gif for visualization
        imageio.mimsave(os.path.join('./demo', f'canonical_360', opt.exp_name,
                        f'{opt.exp_name}_{pose_name}_can.gif'), log_imgs, fps=15)
        print(
            f'gif saved: {os.path.join("./demo", "canonical_360", opt.exp_name, f"{opt.exp_name}_{pose_name}_can.gif")}')
        if opt.log_extra:
            # save camera
            intrisic_path = os.path.join(
                './demo', 'canonical_360', opt.exp_name, f'{opt.exp_name}_{pose_name}_intrinsic.pkl')
            with open(intrisic_path, 'wb') as f:
                # share the same intrinsic matrix for all poses
                pickle.dump(can_cap.intrinsic_matrix, f)
            extrinsic_path = os.path.join(
                './demo', 'canonical_360', opt.exp_name, f'{opt.exp_name}_{pose_name}_extrinsic.pkl')
            all_extrinsics = np.stack(log_extrinsics, axis=0)  # N, 3, 4
            with open(extrinsic_path, 'wb') as f:
                pickle.dump(all_extrinsics, f)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options.set_general_option(parser)
    opt, _ = parser.parse_known_args()

    options.set_nerf_option(parser)
    options.set_pe_option(parser)
    options.set_render_option(parser)
    options.set_trajectory_option(parser)

    parser.add_argument('--image_dir', required=False,
                        type=str, default=None, help='image directory')
    parser.add_argument('--out_dir', default='./out',
                        type=str, help='weights dir')
    parser.add_argument('--offset_scale', default=1.0,
                        type=float, help='scale the predicted offset')
    parser.add_argument('--geo_threshold', default=-1, type=float, help='')
    parser.add_argument('--normalize', default=True,
                        type=options.str2bool, help='')
    parser.add_argument('--bkg_range_scale', default=3,
                        type=float, help='extend near/far range for background')
    parser.add_argument('--human_range_scale', default=1.5,
                        type=float, help='extend near/far range for human')
    parser.add_argument('--offset_scale_type',
                        default='linear', type=str, help='no/linear/tanh')
    parser.add_argument('--exp_name', default='exp',
                        type=str, help='experiment name')
    parser.add_argument('--implicit_model', default="instant_nsr", choices=[
                        "neus", "nerf", "instant_nsr"], type=str, help="choose the implicit model")
    parser.add_argument('--log_extra', default=False, type=options.str2bool,
                        help='whether to log extra info (depth, camera extrinsic Nx4x4, camera intrinsic 3x3)')
    parser.add_argument('--batch_size', type=int,
                        help="maximum number of rays to be rendered together", default=4096)

    opt = parser.parse_args()
    assert opt.geo_threshold == -1, 'please use auto geo_threshold'
    if opt.render_h is None:
        opt.render_size = None
    else:
        opt.render_size = (opt.render_h, opt.render_w)

    options.print_opt(opt)
    main_canonical_360(opt)
