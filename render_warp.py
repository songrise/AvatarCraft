# -*- coding : utf-8 -*-
# @FileName  : render_test_views.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : May 16, 2023
# @Github    : https://github.com/songrise
# @Description: render with pose and/or shape control

import os
import argparse
import joblib

import imageio
import torch
import numpy as np
from utils.constant import CANONICAL_ZOOM_FACTOR, CANONICAL_CAMERA_DIST_VAL, WHITE_BKG, BLACK_BKG, NSR_BOUND, SMPL_SCALE

from utils import render_utils, utils, ray_utils
from utils.SMPLDataset import SMPLDataset
from options import options
import einops
import models.instant_nsr as instant_nsr
from models.smpl import SMPL


def main(opt):

    pose_seq, shape_from, shape_to = None, None, None
    if opt.poseseq_path is not None:
        with open(opt.poseseq_path, 'rb') as f:
            pose_seq = np.load(f).astype(np.float32)
    if opt.shape_from_path is not None:
        with open(opt.shape_from_path, 'rb') as f:
            shape_from = np.load(f).astype(np.float32)
    else:
        shape_from = np.zeros((1, 10))
        shape_from[0, 1] = 2.0
    if opt.shape_to_path is not None:
        with open(opt.shape_to_path, 'rb') as f:
            shape_to = np.load(f).astype(np.float32)
    else:
        shape_to = np.zeros((1, 10))
        shape_to[0, 1] = -2.0

    world_verts, Ts, n_frames = calc_local_trans(
        rest_pose="data/stand_pose.npy",  # !HARDCODED Jan 10:
        render_type=opt.render_type,
        poses=pose_seq,
        shape_from=shape_from,
        shape_to=shape_to,
        max_frames=opt.max_frames
    )

    nerf = instant_nsr.NeRFNetwork()
    nerf.load_state_dict(torch.load(opt.weights_path))
    nerf = nerf.cuda().eval()

    preds = []

    # this dataset is 512x512
    dataloader = SMPLDataset("data/smpl_da_512")
    poses = dataloader.poses  # n_cap x 4 x 4
    _, uvs, faces = utils.read_obj(
        os.path.join('data/smplx/smpl_uv.obj')
    )

    # here are some camera index used in visualizing the results in our paper
    center_view = 58
    side_view = 33
    control_side_view = 39  # side view for control shape
    # here are some mannually selected camera pose index for AMASS SFU dataset that are used in our paper and video (animation),
    # when you render animate, replace the `render_view` to better visualize the results.
    comparison_view = 51
    kick_front_view = 81
    xinjiang_front_view = traditional_front_view = 64
    roll_front_view = 33
    rope_front_view = 54
    vault_2_center_view = 33

    render_view = center_view
    for i in range(n_frames):
        pose = poses[render_view, ...]
        subsample_rate = int(512 / opt.resolution)
        rays_o, rays_d = dataloader.gen_rays_pose(
            pose, subsample_rate)  # only use single camera pose
        # cap = scene[view_name]
        rays_o, rays_d = rays_o.cuda(), rays_d.cuda()
        rays_o, rays_d = einops.rearrange(
            rays_o, "h w c -> (h w) c"), einops.rearrange(rays_d, "h w c -> (h w) c")
        img, _, __ = render_utils.render_instantnsr_naive(
            nerf,
            rays_o,
            rays_d,
            # 4096,
            64*128,
            requires_grad=False,
            bkg_key=WHITE_BKG if opt.white_bkg else BLACK_BKG,
            return_torch=True,
            perturb=False,
            return_raw=True,
            render_can=False,
            verts=world_verts[i],
            faces=faces,
            Ts=Ts[i],
            num_steps=32,
            upsample_steps=32,
            bound=NSR_BOUND
        )
        img = img.detach().cpu()
        img = einops.repeat(img, '(h w) c -> h w c',
                            h=opt.resolution, w=opt.resolution)
        img = utils.integerify_img(img)
        save_path = os.path.join(
            './demo', 'test_views', opt.exp_name, f'{opt.exp_name}_{str(i).zfill(4)}.png')
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        imageio.imsave(save_path, img)
        print(f'image saved: {save_path}')
        preds.append(imageio.imread(save_path))
    if opt.render_type == 'interp_shape':
        # reverse the sequence for smoother animation
        preds = preds + preds[::-1]
    imageio.mimsave(os.path.join('./demo', 'test_views',
                    opt.exp_name, f'{opt.exp_name}.gif'), preds, fps=10)
    print(
        f'gif saved: {os.path.join("./demo", "test_views", opt.exp_name, f"{opt.exp_name}.gif")}')


def calc_local_trans(scale=1, rest_pose=None, render_type: str = "animate", poses=None, shape_from=None, shape_to=None, n_interp=10, max_frames=100):
    """
    generate a sequence of SMPL mesh and inverse transformation for warping the canonical NeuS.
    """

    if rest_pose is not None:
        with open(rest_pose, 'rb') as f:
            rest_pose = np.load(f).astype(np.float32)

    device = torch.device('cpu')
    body_model = SMPL(
        'data/smplx/smpl',
        gender='neutral',
        device=device
    )

    world_verts = []
    Ts = []

    rest_pose[:, 0] = 0.0  # manually set the root joint to be 0

    # interpolation
    zero_shape = np.zeros((1, 10)).astype(np.float32)
    zero_pose = np.zeros((1, 72)).astype(np.float32)
    # how many interval for interpolate betas

    target_shapes = np.linspace(shape_from, shape_to, n_interp)

    if render_type == "animate":
        n_frame = min(max_frames, poses.shape[0])
        target_shapes = np.zeros((n_frame, 1, 10))
    elif render_type == "interp_shape":
        n_frame = min(max_frames, target_shapes.shape[0])
        poses = np.zeros((n_frame, 72))
    else:
        raise NotImplementedError

    for i in range(n_frame):
        # assert 0 <= frame_id < len(caps)

        # the "da(大) pose" used in NeuMan
        da_smpl = np.zeros((1, 72)).astype(np.float32)
        da_smpl = da_smpl.reshape(-1, 3)
        da_smpl[1] = np.array([0, 0, 1.0])
        da_smpl[2] = np.array([0, 0, -1.0])
        da_smpl = da_smpl.reshape(1, -1)  # (1, 72)

        # transformation from t-pose to articulation pose
        _, T_t2pose, __ = body_model.verts_transformations(
            return_tensor=False,
            poses=poses[i, ...][None],
            betas=zero_shape,
            concat_joints=True
        )

        # transformation for t-pose to the canonical da-pose
        v0, T_t2rest, _ = body_model.verts_transformations(
            return_tensor=False,
            poses=da_smpl,
            betas=zero_shape,
            concat_joints=True
        )

        vt, _, _ = body_model.verts_transformations(
            return_tensor=False,
            poses=da_smpl,
            betas=target_shapes[i],
            concat_joints=True
        )

        delta_v = v0 - vt  # vertex displacement caused by beta-blend shape in SMPL

        delta_v = delta_v.squeeze()
        T_shape = np.concatenate([np.eye(4)[None, ...]
                                 for _ in range(6890+24)])
        T_shape = ray_utils.batch_add_translation(T_shape, delta_v)

        T_scale = np.eye(4) / SMPL_SCALE
        T_rest2pose = T_t2pose @ np.linalg.inv(
            T_shape) @ np.linalg.inv(T_t2rest)
        T_rest2scene = T_rest2pose
        T_rest2scene_nerf = T_rest2scene @ T_scale
        s = np.eye(4)
        s[:3, :3] *= scale
        T_rest2scene = s @ T_rest2scene

        rest_pose_verts, rest_pose_joints = body_model.forward(
            return_tensor=False,
            return_joints=True,
            poses=da_smpl,
            betas=zero_shape,
        )
        temp_world_verts = np.einsum('BNi, Bi->BN', T_rest2scene, ray_utils.to_homogeneous(
            np.concatenate([rest_pose_verts, rest_pose_joints], axis=0)))[:, :3].astype(np.float32)
        temp_world_verts = temp_world_verts[:6890, :]
        # utils.save_mesh(temp_world_verts, body_model.faces,"test_Warp.ply")

        Ts.append(T_rest2scene_nerf)
        world_verts.append(temp_world_verts)
    return world_verts, Ts, n_frame


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
    parser.add_argument('--num_offset_nets', default=1,
                        type=int, help='how many offset networks')
    parser.add_argument('--offset_scale_type',
                        default='linear', type=str, help='no/linear/tanh')
    parser.add_argument('--exp_name', default='exp',
                        type=str, help='experiment name')
    parser.add_argument('--implicit_model', default="instant_nsr", choices=[
                        "neus", "nerf", "instant_nsr"], type=str, help="choose the implicit model")
    # render args
    parser.add_argument('--poseseq_path', default=None,
                        type=str, help="path to the pose sequence")
    parser.add_argument('--render_type', default='animate', type=str, choices=[
                        'animate', 'interp_shape'], help="control pose (animate) or shape (interp_shape))")
    parser.add_argument('--shape_from_path', type=str,
                        help='path to the shape0 sequence, used for interpolation')
    parser.add_argument('--shape_to_path', type=str,
                        help='path to the shape1 sequence, used for interpolation')
    parser.add_argument('--max_frames', default=20, type=int,
                        help='max number of frames to render')
    parser.add_argument('--resolution', default=256, type=int,
                        help='rendering resolution, 128x128, 256x256, 512x512', choices=[128, 256, 512])

    opt = parser.parse_args()

    main(opt)
