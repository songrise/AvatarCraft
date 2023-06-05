# Code based on nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf.py
# License from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch/blob/master/LICENSE


import random
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PointLights, TexturesVertex,
    PerspectiveCameras
)
from PIL import Image

from cameras.camera_pose import CameraPose
from cameras.captures import ResizedPinholeCapture
from cameras.pinhole_camera import PinholeCamera
from utils import ray_utils, debug_utils
from utils.constant import DEFAULT_GEO_THRESH,CANONICAL_ZOOM_FACTOR, CANONICAL_CAMERA_DIST_VAL, WHITE_BKG, BLACK_BKG, NOISE_BKG, CHESSBOARD_BKG

from geometry import transformations
import einops
import open3d as o3d
from torchvision import transforms
from scipy.stats import beta
import json
import cv2 as cv

import nerf_pytorch.run_nerf as run_nerf
import gc
########DEBUG


trans_t = lambda t: np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]])

rot_phi = lambda phi: np.array([
    [1, 0,           0,            0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi),  0],
    [0, 0,           0,            1]])

rot_theta = lambda th: np.array([
    [np.cos(th), 0, -np.sin(th), 0],
    [0,          1, 0,           0],
    [np.sin(th), 0, np.cos(th),  0],
    [0,          0, 0,           1]])


def pose_spherical(theta, phi, radius, add_noise, noise_scale = 1.0):
    if add_noise:
        # radius += np.random.normal(0, 0.05) * noise_scale
        #! Nov 06: noise to focal length, could only be closer
        radius += np.random.uniform(-0.2, 0) * noise_scale
        phi += np.random.uniform(-15, 15) * noise_scale
        theta += np.random.normal(0, 1) * noise_scale

    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    transf = np.array([
        [1, 0,  0,  0],
        [0, -1, 0,  0],
        [0, 0,  -1, 0],
        [0, 0,  0,  1.],
    ])
    c2w = c2w @ transf
    return CameraPose.from_camera_to_world(c2w, unstable=True)



def describe_view(angles, body_part:str = "body"):
    descs = []
    for angle in angles:
        if -180 <= angle <= -150 or 150 <= angle <= 180:
            descs.append(f"front view of the {body_part} of the")
        elif -30 <= angle <= 30:
            descs.append(f"back view of the {body_part} of the")
        else:
            descs.append(f"side view of the {body_part} of the")

    return descs



def load_transform(path):
    """
    load transform from json file
    """
    with open(path) as f:
        transforms = json.load(f)
        transforms = [np.array(v["transform_matrix"])[:3,:4] for v in transforms["frames"]]
    return transforms


def path_from_transform(transforms):
    """
    neus style transform to NeuMan style pose
    """
    pose = [CameraPose.from_camera_to_world(t) for t in transforms]

    return pose, None


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def default_360_path(center, up, dist, res=40, rad=360, add_noise=False):
    """
    Args:
    add_noise: if True, add random noise to the camera pose, used in stylization training
    """
    up2 = np.array([0, 0, 1])
    axis = np.cross(up, up2)
    angle = transformations.angle_between_vectors(up, up2)
    rot = transformations.rotation_matrix(-angle, axis)
    trans = transformations.translation_matrix(center)
    angles = np.linspace(-rad / 2, rad / 2, res + 1)[:-1]

    poses = [pose_spherical(angle, 0, dist, add_noise,  noise_scale = 1.0) 
                for angle in angles]

    poses = [CameraPose.from_camera_to_world(trans @ rot @ p.camera_to_world) for p in poses]
    desc = describe_view(angles)
    return poses, desc


def style_360_path(center, up, dist, res=40, rad=360, add_noise=False, noise_scale = 1.0, style_head = False, head_offset = 0.0, body_part:str = "body", head_rate = 0.0, head_dist = 0.5):
    """
    Generate camera poses for style training.
    sample more capture in front and back of a human, to ensure semantic infomation for CLIP score.

    Args:
        add_noise: if True, add random noise to the camera pose, used in stylization training
        style_head: if True, add extra camera pose for head stylization
        head_offset: offset of head from the center of the body
    """


    up2 = np.array([0, 0, 1])
    axis = np.cross(up, up2)
    angle = transformations.angle_between_vectors(up, up2)
    rot = transformations.rotation_matrix(-angle, axis)
    trans = transformations.translation_matrix(center)


    #! Nov 06: only sample front and back, [-45, 45], [135, 180], [-180, -135]
    style_angles = np.concatenate([np.linspace(-180, -120, res//4),
                                    np.linspace(120, 180, res//4),
                                    np.linspace(-60, 60, res//2),])

    poses = [pose_spherical(angle, 0, dist, add_noise, noise_scale) 
                for angle in style_angles]

    poses = [CameraPose.from_camera_to_world(trans @ rot @ p.camera_to_world) for p in poses]
    # generate 
    desc = describe_view(style_angles, body_part)
    if style_head and head_rate > 0.0:
        res = int(res * head_rate)
        up2 = np.array([0, 0, 1])
        axis = np.cross(up, up2)
        angle = transformations.angle_between_vectors(up, up2)
        rot = transformations.rotation_matrix(-angle, axis)
        trans = transformations.translation_matrix(center + up * head_offset)

        # style_angles = np.linspace(-rad / 2, rad / 2, res + 1)[:-1]

        #! Nov 06: only sample front and back, [-45, 45], [135, 180], [-180, -135]
        style_angles = np.concatenate([np.linspace(-180, -120, res//2),
                                        np.linspace(120, 180, res//2),
                                        ])

        face_poses = [pose_spherical(angle, 0, head_dist, True, 1.0) 
                    for angle in style_angles]

        face_poses = [CameraPose.from_camera_to_world(trans @ rot @ p.camera_to_world) for p in face_poses]
        # generate 
        face_desc = describe_view(style_angles, "face")
        return poses + face_poses, desc + face_desc
    return poses, desc



def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkg=True):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    device = raw.device
    _raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = (z_vals[..., 1:] - z_vals[..., :-1])
    dists = torch.cat([dists, torch.Tensor([1e10], ).expand(dists[..., :1].shape).to(device)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape, device=device) * raw_noise_std
    alpha = _raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, device=device), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkg:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_vanilla(coarse_net, cap, fine_net=None, rays_per_batch=32768, samples_per_ray=64, importance_samples_per_ray=128, white_bkg=True, near_far_source='bkg', return_depth=False, ablate_nerft=False):
    device = next(coarse_net.parameters()).device

    def build_batch(origins, dirs, near, far):
        _bs = origins.shape[0]
        ray_batch = {
            'origin':    torch.from_numpy(origins).float().to(device),
            'direction': torch.from_numpy(dirs).float().to(device),
            'near':      torch.tensor([near] * _bs, dtype=torch.float32)[..., None].to(device),
            'far':       torch.tensor([far] * _bs, dtype=torch.float32)[..., None].to(device)
        }
        return ray_batch

    with torch.set_grad_enabled(False):
        origins, dirs = ray_utils.shot_all_rays(cap)
        total_rays = origins.shape[0]
        total_rgb_map = []
        total_depth_map = []
        for i in range(0, total_rays, rays_per_batch):
            print(f'{i} / {total_rays}')
            ray_batch = build_batch(
                origins[i:i + rays_per_batch],
                dirs[i:i + rays_per_batch],
                cap.near[near_far_source],
                cap.far[near_far_source]
            )
            if ablate_nerft:
                cur_time = cap.frame_id['frame_id'] / cap.frame_id['total_frames']
                coarse_time = torch.ones(origins[i:i + rays_per_batch].shape[0], samples_per_ray, 1, device=device) * cur_time
                fine_time = torch.ones(origins[i:i + rays_per_batch].shape[0], samples_per_ray + importance_samples_per_ray, 1, device=device) * cur_time
            else:
                coarse_time, fine_time = None, None
            _pts, _dirs, _z_vals = ray_utils.ray_to_samples(ray_batch, samples_per_ray, device=device, append_t=coarse_time)
            out = coarse_net(
                _pts,
                _dirs
            )
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(out, _z_vals, _dirs[:, 0, :], white_bkg=white_bkg)

            if fine_net is not None:
                _pts, _dirs, _z_vals = ray_utils.ray_to_importance_samples(ray_batch, _z_vals, weights, importance_samples_per_ray, device=device, append_t=fine_time)
                out = fine_net(
                    _pts,
                    _dirs
                )
                rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(out, _z_vals, _dirs[:, 0, :], white_bkg=white_bkg)

            total_rgb_map.append(rgb_map)
            total_depth_map.append(depth_map)
        total_rgb_map = torch.cat(total_rgb_map).reshape(*cap.shape, -1).detach().cpu().numpy()
        total_depth_map = torch.cat(total_depth_map).reshape(*cap.shape).detach().cpu().numpy()
    if return_depth:
        return total_rgb_map, total_depth_map
    return total_rgb_map

# def pose2cap(scene, pose, hw):
#     can_cap = ResizedPinholeCapture(
#             PinholeCamera(
#                 scene.captures[0].pinhole_cam.width,
#                 scene.captures[0].pinhole_cam.height,
#                 CANONICAL_ZOOM_FACTOR * scene.captures[0].pinhole_cam.width,
#                 CANONICAL_ZOOM_FACTOR * scene.captures[0].pinhole_cam.width,
#                 scene.captures[0].pinhole_cam.width / 2.0,
#                 scene.captures[0].pinhole_cam.height / 2.0,
#             ),
#             pose,
#             # tgt_size=scene.captures[0].pinhole_cam.shape
#             tgt_size = hw
#         )
#     return can_cap

def pose2cap(hw, pose):
    h, w = hw
    can_cap = ResizedPinholeCapture(
            PinholeCamera(
                w,
                h,
                CANONICAL_ZOOM_FACTOR * w,
                CANONICAL_ZOOM_FACTOR * w,
                w / 2.0,
                h / 2.0
            ),
            pose,
            tgt_size = hw
        )
    return can_cap


def K_Rt_2cap(K,pose):
    H = K[0,2]*2
    W = K[1,2]*2
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    can_cap = ResizedPinholeCapture(
            PinholeCamera(
                W,
                H,
                fx,
                fy,
                cx,
                cy,
            ),
            pose,
            # tgt_size=scene.captures[0].pinhole_cam.shape
            tgt_size = (H, W)
        )
    return can_cap

def cap2rays(cap):
    """
    Args:
        cap: Capture, containing both intrinsic and extrinsic parameters.
    Returns:
        origins: [HxW, 3]
        dirs: [HxW, 3]
    """

    device = "cuda" #!HARDCODED Sep 25
    coords = np.argwhere(np.ones(cap.shape))[:, ::-1]
    origins, dirs = ray_utils.shot_rays(cap, coords)
    origins, dirs = torch.from_numpy(origins).to(device), torch.from_numpy(dirs).to(device)
    return origins, dirs




def render_nerf_naive(net,rays_o, rays_d, rays_per_batch=6400,
            requires_grad=False,return_torch=True,white_bkg=True,
            render_can=False,return_depth=False,return_mask=False):
    """
    naive way to render all rays provided.
    net: the nerf model
    Return:
        rgb_map: [H, W, C]
    """
    def build_batch(origins, dirs, near, far):
        if isinstance(origins, torch.Tensor):
            ray_batch = {
                'origin':    (origins).float().to(device),
                'direction': (dirs).float().to(device),
                'near':      (near[..., None]).float().to(device),
                'far':       (far[..., None]).float().to(device)
            }
        else:
            ray_batch = {
                'origin':    torch.from_numpy(origins).float().to(device),
                'direction': torch.from_numpy(dirs).float().to(device),
                'near':      torch.from_numpy(near[..., None]).float().to(device),
                'far':       torch.from_numpy(far[..., None]).float().to(device)
            }
        return ray_batch

    device = rays_o.device
    origins, dirs = rays_o, rays_d
    total_rays = rays_o.shape[0]
    total_rgb_map = []
    total_depth_map = []
    total_acc_map = []
    # for i in tqdm.trange(0, total_rays, rays_per_batch, desc='rendering rays '):
    with torch.set_grad_enabled(requires_grad):
        near = torch.ones(origins.shape[0]).to(device)
        far = torch.ones(origins.shape[0]).to(device) * 4

        for i in range(0, total_rays, rays_per_batch):        
            ray_batch = build_batch(
                origins[i:i + rays_per_batch],
                dirs[i:i + rays_per_batch],
                near[i:i + rays_per_batch],
                far[i:i + rays_per_batch],
                )
            _pts, _dirs, _z_vals = ray_utils.ray_to_samples(ray_batch, 128, device=device)
            can_pts = _pts
            can_dirs = _dirs

            can_pts = can_pts.to(device).float()
            can_dirs = can_dirs.to(device).float()
            # out = net.coarse_human_net(can_pts, can_dirs)
            out = net.coarse_human_net(can_pts, can_dirs)
            _rgb_map, _, _acc_map, _, _depth_map = raw2outputs(out, _z_vals, _dirs[:, 0, :], white_bkg=white_bkg)

            total_rgb_map.append(_rgb_map)
            total_depth_map.append(_depth_map)
            total_acc_map.append(_acc_map)

        total_rgb_map = torch.cat(total_rgb_map).unsqueeze(0)
        total_depth_map = torch.cat(total_depth_map).unsqueeze(0)
        total_acc_map = torch.cat(total_acc_map).unsqueeze(0)
    if not return_torch:
        total_rgb_map = total_rgb_map.detach().cpu().numpy()
        total_depth_map = total_depth_map.detach().cpu().numpy()
        total_acc_map = total_acc_map.detach().cpu().numpy()
    if return_depth and return_mask:
        return total_rgb_map, total_depth_map, total_acc_map
    if return_depth:
        return total_rgb_map, total_depth_map
    if return_mask:
        return total_rgb_map, total_acc_map
    return total_rgb_map




def render_neus_naive(net,rays_o,rays_d,rays_per_batch=6400,
            requires_grad=False,return_torch=True,white_bkg=False,
            render_can=False, perturb = 1.0,return_raw = False):
    """
    naive way to render all rays provided. This is used to pretrain
    a Neus on canonical scene
    net: the nerf model
    Return:
        rgb_map: [H, W, C]
    """


    device = rays_o.device
    origins, dirs = rays_o, rays_d
    total_rays = rays_o.shape[0]
    total_rgb_map = []
    total_depth_map = []
    total_acc_map = []
    total_eikonal = 0.0
    # for i in tqdm.trange(0, total_rays, rays_per_batch, desc='rendering rays '):
    render_kwargs = {
        "perturb_overwrite": perturb,
        "n_importance_overwrite": 3,
        "cos_anneal_ratio": 1.0 #todo add code to do anneal
    }

    with torch.set_grad_enabled(requires_grad):


        for i in range(0, total_rays, rays_per_batch):        
  
            rays_o_batch = origins[i:i + rays_per_batch]
            rays_d_batch = dirs[i:i + rays_per_batch]
            # near_batch = near[i:i + rays_per_batch]
            # far_batch = far[i:i + rays_per_batch]
            near, far = 1.0, 4.0
            #!neus output here
            background_rgb = None
            if white_bkg:
                background_rgb = torch.ones([1, 3]).to(device)
            out = net.coarse_human_net.render(rays_o_batch, rays_d_batch, near, far, background_rgb = background_rgb, **render_kwargs)
            #########convert raw output to rgb using code from render_core()
            _rgb_map = out["color_fine"]
            eikonal_val = out["gradient_error"]          
            total_eikonal = total_eikonal + eikonal_val
            total_rgb_map.append(_rgb_map)

        total_rgb_map = torch.cat(total_rgb_map).unsqueeze(0)

    if not return_torch:
        total_rgb_map = total_rgb_map.detach().cpu().numpy()
        total_depth_map = total_depth_map.detach().cpu().numpy()
        total_acc_map = total_acc_map.detach().cpu().numpy()
    if return_raw: # this assumes only one batch is evaluated
        return total_rgb_map, total_eikonal, out
    return total_rgb_map, total_eikonal

def render_instantnsr_naive(net,rays_o,rays_d,rays_per_batch=6400,
            requires_grad=False,return_torch=True, bkg_key:int = WHITE_BKG,
            render_can:bool=False, perturb:bool = True, return_raw:bool = False,
            verts = None, faces = None, Ts = None, num_steps:int = 64, 
            upsample_steps = 64, bound:float = 1.6):
    """
    naive way to render all rays provided. 
    Args:
        net: the nerf model
        rays_o: [H*W, 3]
        rays_d: [H*W, 3]
        rays_per_batch: number of rays to render per batch
        requires_grad: whether to require gradient for the forward pass
        return_torch: whether to return torch tensor or numpy array
        bkg_key: which background to use, see select_background()
        render_can: whether to render the canonical avatar
        perturb: whether to perturb the ray sampling
        return_raw: whether to return raw output from Neus
        verts: the verts of SMPL model, used to warp the canonical avatar, only used when render_can is False
        faces: the faces of SMPL model, used to warp the canonical avatar, only used when render_can is False
        Ts: the inverse transformation of verts, used to warp the canonical avatar, only used when render_can is False
        num_steps: number of coarse steps to sample pts when rendering the avatar
        upsample_steps: number of fine steps to sample pts when rendering the avatar
    Return:
        rgb_map: [H, W, C]
    """


    device = rays_o.device
    origins, dirs = rays_o, rays_d
    total_rays = rays_o.shape[0]
    total_rgb_map = []
    total_depth_map = []
    total_weight_sum_map = []
    total_normal_map = []
    extra_out = {}
    total_eikonal = 0.0



    with torch.set_grad_enabled(requires_grad):

        for i in range(0, total_rays, rays_per_batch):        
  
            rays_o_batch = origins[i:i + rays_per_batch]
            rays_d_batch = dirs[i:i + rays_per_batch]
            #!neus output here
            background_rgb = select_background(rays_o_batch.shape, bkg_key).to(device)
            #!HARDCODED Dec 09: 
            rays_o_batch, rays_d_batch = rays_o_batch.unsqueeze(0), rays_d_batch.unsqueeze(0)
            out = net.render(rays_o_batch, rays_d_batch, num_steps = num_steps, 
                        upsample_steps = upsample_steps, bound = bound, staged=False, bg_color = background_rgb,
                        cos_anneal_ratio = 1.0, normal_epsilon_ratio = 0.0, render_can = render_can,
                        verts = verts, faces = faces, Ts = Ts, perturb = perturb)

            _rgb_map = out['rgb']
            _weight_sum_map = out['weight_sum']
            _depth_map = out['depth']
            _normal_map = out['normal']
            eikonal_val = out["gradient_error"]       

            total_eikonal = total_eikonal + eikonal_val
            total_rgb_map.append(_rgb_map)
            total_depth_map.append(_depth_map)
            total_weight_sum_map.append(_weight_sum_map)
            total_normal_map.append(_normal_map)

        total_rgb_map = torch.cat(total_rgb_map).squeeze(0) #[N_patch, patch_size, 3]
        total_rgb_map = total_rgb_map.reshape(-1,3) #[N_patch*patch_size, 3]
        total_depth_map = torch.cat(total_depth_map).squeeze(0) #[N_patch, patch_size, 1]
        total_depth_map = total_depth_map.reshape(-1,1) #[N_patch*patch_size, 1]
        total_weight_sum_map = torch.cat(total_weight_sum_map).squeeze(0) #[N_patch, patch_size, 1]
        total_weight_sum_map = total_weight_sum_map.reshape(-1,1) #[N_patch*patch_size, 1]
        total_normal_map = torch.cat(total_normal_map).squeeze(0) #[N_patch, patch_size, 3]
        total_normal_map = total_normal_map.reshape(-1,3) #[N_patch*patch_size, 3]

        extra_out["depth"] = total_depth_map
        extra_out["weight_sum"] = total_weight_sum_map
        extra_out["normal"] = total_normal_map
    if not return_torch:
        total_rgb_map = total_rgb_map.detach().cpu().numpy()
        extra_out["depth"] = total_depth_map.detach().cpu().numpy()
        extra_out["weight_sum"] = total_weight_sum_map.detach().cpu().numpy()
        extra_out["normal"] = total_normal_map.detach().cpu().numpy()
    if return_raw:
        return total_rgb_map, total_eikonal, extra_out
    return total_rgb_map, total_eikonal




def render_hybrid_avatar(net, cap, posed_verts, faces, Ts, rays_per_batch=32768, samples_per_ray=64, importance_samples_per_ray=128, white_bkg=True, geo_threshold=DEFAULT_GEO_THRESH, return_depth=False, scene_scale:float = 1):
    """
    Render avatar(instant-nsr) with background(NeRF).
    """
    device = next(net.parameters()).device

    def build_batch(origins, dirs, near, far):
        if isinstance(origins, torch.Tensor):
            ray_batch = {
                'origin':    (origins).float().to(device),
                'direction': (dirs).float().to(device),
                'near':      (near[..., None]).float().to(device),
                'far':       (far[..., None]).float().to(device)
            }
        else:
            ray_batch = {
                'origin':    torch.from_numpy(origins).float().to(device),
                'direction': torch.from_numpy(dirs).float().to(device),
                'near':      torch.from_numpy(near[..., None]).float().to(device),
                'far':       torch.from_numpy(far[..., None]).float().to(device)
            }
        return ray_batch

    with torch.set_grad_enabled(False):
        coords = np.argwhere(np.ones(cap.shape))[:, ::-1]
        origins, dirs = ray_utils.shot_rays(cap, coords)
        total_rays = origins.shape[0]
        total_rgb_map = []
        total_depth_map = []
        total_acc_map = []
        for i in range(0, total_rays, rays_per_batch):
            print(f'{i} / {total_rays}')
            rgb_map = np.zeros_like(origins[i:i + rays_per_batch])
            depth_map = np.zeros_like(origins[i:i + rays_per_batch, 0])
            acc_map = np.zeros_like(origins[i:i + rays_per_batch, 0])
            bkg_ray_batch = build_batch(
                origins[i:i + rays_per_batch],
                dirs[i:i + rays_per_batch],
                np.array([cap.near['bkg']] * origins[i:i + rays_per_batch].shape[0]) / scene_scale,
                np.array([cap.far['bkg']] * origins[i:i + rays_per_batch].shape[0]) / scene_scale,
            )
            bkg_pts, bkg_dirs, bkg_z_vals = ray_utils.ray_to_samples(bkg_ray_batch, samples_per_ray, device=device)
            bkg_pts = bkg_pts * scene_scale
            bkg_out = net.coarse_bkg_net(
                bkg_pts,
                bkg_dirs
            )
            if net.fine_bkg_net is not None:
                _, _, _, bkg_weights, _ = raw2outputs(bkg_out, bkg_z_vals, bkg_dirs[:, 0, :], white_bkg=white_bkg)
                bkg_pts, bkg_dirs, bkg_z_vals = ray_utils.ray_to_importance_samples(bkg_ray_batch, bkg_z_vals, bkg_weights, importance_samples_per_ray, device=device)
                bkg_pts = bkg_pts * scene_scale
                bkg_out = net.fine_bkg_net(
                    bkg_pts,
                    bkg_dirs
                )
            # this near and far should be the same as the one calculated in instant-nsr:run()
            temp_near, temp_far = ray_utils.geometry_guided_near_far(origins[i:i + rays_per_batch], dirs[i:i + rays_per_batch], posed_verts, geo_threshold=geo_threshold)
            if (temp_near >= temp_far).any():
                # no fuse
                # render bkg colors
                coarse_bkg_rgb_map, _, coarse_bkg_acc_map, weights, coarse_bkg_depth_map = raw2outputs(
                    bkg_out,
                    bkg_z_vals,
                    bkg_dirs[:, 0, :],
                    white_bkg=white_bkg
                )
                rgb_map = coarse_bkg_rgb_map.detach().cpu().numpy()
                depth_map = coarse_bkg_depth_map.detach().cpu().numpy()
                # normalize depth map
                depth_map = depth_map * scene_scale / cap.far['bkg']
                # depth_map = np.zeros_like(depth_map)
                acc_map = np.zeros_like(depth_map)[:,None]
                acc_map = acc_map.repeat(3, axis=-1)
            if (temp_near < temp_far).any():
                # avatar
                rays_o_batch = origins[i:i + rays_per_batch]
                rays_d_batch = dirs[i:i + rays_per_batch]

                #!neus output here
                background_rgb = select_background(rays_o_batch.shape, 0).to(device)
                rays_o_batch, rays_d_batch = torch.from_numpy(rays_o_batch).float().to(device), torch.from_numpy(rays_d_batch).float().to(device)
                #!HARDCODED Dec 09: 
                rays_o_batch, rays_d_batch = rays_o_batch.unsqueeze(0), rays_d_batch.unsqueeze(0)
                out = net.coarse_human_net.render(rays_o_batch, rays_d_batch, num_steps = 64, 
                            upsample_steps = 0, bound = 1.6, staged=False, bg_color = background_rgb,
                            cos_anneal_ratio = 1.0, normal_epsilon_ratio = 0.0, render_can = False,
                            verts = posed_verts, faces = faces, Ts = Ts, perturb = 0)
                human_rgb_map = out['rgb'].squeeze().detach().cpu().numpy()
                human_acc_map = out["weight_sum"].detach().cpu().numpy() 
                human_depth_map = out["depth"].squeeze().detach().cpu().numpy()
                # human_depth_map[human_depth_map<1e-2] = 1e5
                human_depth_map = human_depth_map #/ cap.far['human']
                human_depth_map[human_acc_map[:,0]<0.9] = 1.0 # set background depth to 1.0
                human_depth_map[human_acc_map[:,0]>0.9] = 0.29
                human_acc_map = human_acc_map.repeat(3, axis = -1)  # [Nrays, 3]
                # alpha-blend background and avatar according to the ray opacity
                # rgb_map = human_rgb_map * human_acc_map + (1. - human_acc_map)*rgb_map
                # First,  only solid part of human should be rendered
                human_rgb_map = human_rgb_map * human_acc_map
                # Then, only visible part should be rendered
                rgb_map = composite_by_depth(rgb_map, human_rgb_map, depth_map, human_depth_map)
                depth_map = np.minimum(depth_map, human_depth_map)
                # depth_map = human_depth_map
                acc_map = human_acc_map
            total_rgb_map.append(rgb_map)
            total_depth_map.append(depth_map)
            total_acc_map.append(acc_map)
        total_rgb_map = np.concatenate(total_rgb_map).reshape(*cap.shape, -1)
        total_depth_map = np.concatenate(total_depth_map).reshape(*cap.shape)
        # debug_utils.dump_tensor(total_depth_map, 'total_depth_map.pkl')
        total_acc_map = np.concatenate(total_acc_map).reshape(*cap.shape,-1)
        # debug_utils.dump_tensor(total_acc_map, 'total_acc_map.pkl')
    if return_depth:
        return total_rgb_map, total_depth_map
    return total_rgb_map

def render_hybrid_avatar_llff(net, scene_kwargs, cap, posed_verts, faces, Ts, rays_per_batch=32768, samples_per_ray=64, importance_samples_per_ray=128, white_bkg=True, geo_threshold=DEFAULT_GEO_THRESH, return_depth=False, scene_scale:float = 1):
    """
    Render avatar(instant-nsr) with background(NeRF).
    """
    device = next(net.parameters()).device

 

    with torch.set_grad_enabled(False):
        coords = np.argwhere(np.ones(cap.shape))[:, ::-1]
        origins, dirs = ray_utils.shot_rays(cap, coords)
        total_rays = origins.shape[0]
        total_rgb_map = []
        total_depth_map = []
        total_acc_map = []
        # no fuse
        # render bkg colors
        H, W = cap.shape[0], cap.shape[1]
        K, c2w = torch.from_numpy(cap.intrinsic_matrix).float().to(device), torch.from_numpy(cap.extrinsic_matrix).float().to(device)
        if False:
            # move the camera
            T_move = np.array([[np.cos(-np.pi/12), 0, np.sin(-np.pi/12), -0.2],
                              [0, 1, 0, -0.1],
                              [-np.sin(-np.pi/12), 0, np.cos(-np.pi/12), 0.0],
                              [0, 0, 0, 1]])
            # T_move = np.array([[1, 0, 0, 0.],
            #                     [0, 1, 0, 0.],
            #                     [0, 0, 1, 0.2],
            #                     [0, 0, 0, 1]])
            c2w = torch.cat([c2w, torch.tensor([0,0,0,1]).float().to(device)[None,:]], dim=0)
            c2w = torch.from_numpy(T_move).float().to(device) @ c2w
            c2w = c2w[:3,:4]
        bkg_rgb, bkg_disp, bkg_acc, bkg_depth, _ = run_nerf.render(H, W, K, chunk=8192*4, c2w=c2w, **scene_kwargs)
        bkg_rgb_map = bkg_rgb.detach().cpu().numpy()
        bkg_depth_map = bkg_depth.detach().cpu().numpy()
        # normalize depth map

        # bkg_depth_map = bkg_depth_map * scene_scale 
        # depth_map = np.zeros_like(depth_map)
        # acc_map = np.zeros_like(depth_map)[:,None]
        # acc_map = acc_map.repeat(3, axis=-1)

        for i in range(0, total_rays, rays_per_batch):
            print(f'{i} / {total_rays}')
            rgb_map = np.zeros_like(origins[i:i + rays_per_batch])
            depth_map = np.zeros_like(origins[i:i + rays_per_batch, 0])
            acc_map = np.zeros_like(origins[i:i + rays_per_batch, 0])

            # this near and far should be the same as the one calculated in instant-nsr:run()
            temp_near, temp_far = ray_utils.geometry_guided_near_far(origins[i:i + rays_per_batch], dirs[i:i + rays_per_batch], posed_verts, geo_threshold=geo_threshold)

            rays_o_batch = origins[i:i + rays_per_batch]
            rays_d_batch = dirs[i:i + rays_per_batch]

            #!neus output here
            background_rgb = select_background(rays_o_batch.shape, 0).to(device)
            rays_o_batch, rays_d_batch = torch.from_numpy(rays_o_batch).float().to(device), torch.from_numpy(rays_d_batch).float().to(device)
            #!HARDCODED Dec 09: 
            rays_o_batch, rays_d_batch = rays_o_batch.unsqueeze(0), rays_d_batch.unsqueeze(0)
            out = net.coarse_human_net.render(rays_o_batch, rays_d_batch, num_steps = 64, 
                        upsample_steps = 0, bound = 1.6, staged=False, bg_color = background_rgb,
                        cos_anneal_ratio = 1.0, normal_epsilon_ratio = 0.0, render_can = False,
                        verts = posed_verts, faces = faces, Ts = Ts, perturb = 0, use_mesh_guide = True)
            human_rgb_map = out['rgb'].squeeze().detach().cpu().numpy()
            human_acc_map = out["weight_sum"].detach().cpu().numpy() 
            human_depth_map = out["depth"].squeeze().detach().cpu().numpy()
            # human_depth_map[human_depth_map<1e-2] = 1e5
            human_depth_map = human_depth_map #/ cap.far['human']
            human_depth_map[human_acc_map[:,0]<0.95] = 1.0 # set background depth to 1.0
            human_depth_map[human_acc_map[:,0]>0.95] = 0.43
            human_acc_map = human_acc_map.repeat(3, axis = -1)  # [Nrays, 3]
            acc_map = human_acc_map
            total_rgb_map.append(human_rgb_map)
            total_depth_map.append(human_depth_map)
            total_acc_map.append(human_acc_map)
        human_rgb_map = np.concatenate(total_rgb_map).reshape(*cap.shape, -1)
        human_depth_map = np.concatenate(total_depth_map).reshape(*cap.shape)
        human_acc_map = np.concatenate(total_acc_map).reshape(*cap.shape, -1)

        # alpha-blend background and avatar according to the ray opacity
        # rgb_map = human_rgb_map * human_acc_map + (1. - human_acc_map)*bkg_rgb_map
        # First,  only solid part of human should be rendered
        human_rgb_map = human_rgb_map * human_acc_map
        # Then, only visible part should be rendered
        rgb_map = composite_by_depth(bkg_rgb_map, human_rgb_map, bkg_depth_map, human_depth_map)
        depth_map = np.minimum(bkg_depth_map, human_depth_map)
        # depth_map = human_depth_map
        # debug_utils.dump_tensor(total_depth_map, 'total_depth_map.pkl')
        # total_acc_map = np.concatenate(total_acc_map).reshape(*cap.shape,-1)
        # debug_utils.dump_tensor(total_acc_map, 'total_acc_map.pkl')
    if return_depth:
        return rgb_map, depth_map
    return total_rgb_map

def render_hybrid_nerf_multi_persons(bkg_model, cap, human_models, posed_verts, faces, Ts, rays_per_batch=32768, samples_per_ray=64, importance_samples_per_ray=128, white_bkg=True, geo_threshold=DEFAULT_GEO_THRESH, return_depth=False):
    device = next(bkg_model.parameters()).device

    def build_batch(origins, dirs, near, far):
        if isinstance(origins, torch.Tensor):
            ray_batch = {
                'origin':    (origins).float().to(device),
                'direction': (dirs).float().to(device),
                'near':      (near[..., None]).float().to(device),
                'far':       (far[..., None]).float().to(device)
            }
        else:
            ray_batch = {
                'origin':    torch.from_numpy(origins).float().to(device),
                'direction': torch.from_numpy(dirs).float().to(device),
                'near':      torch.from_numpy(near[..., None]).float().to(device),
                'far':       torch.from_numpy(far[..., None]).float().to(device)
            }
        return ray_batch
    with torch.set_grad_enabled(False):
        coords = np.argwhere(np.ones(cap.shape))[:, ::-1]
        origins, dirs = ray_utils.shot_rays(cap, coords)
        total_rays = origins.shape[0]
        total_rgb_map = []
        total_depth_map = []
        for i in range(0, total_rays, rays_per_batch):
            print(f'{i} / {total_rays}')
            bkg_ray_batch = build_batch(
                origins[i:i + rays_per_batch],
                dirs[i:i + rays_per_batch],
                np.array([cap.near['bkg']] * origins[i:i + rays_per_batch].shape[0]),
                np.array([cap.far['bkg']] * origins[i:i + rays_per_batch].shape[0]),
            )
            bkg_pts, bkg_dirs, bkg_z_vals = ray_utils.ray_to_samples(bkg_ray_batch, samples_per_ray, device=device)
            bkg_out = bkg_model.coarse_bkg_net(
                bkg_pts,
                bkg_dirs
            )
            if bkg_model.fine_bkg_net is not None:
                _, _, _, bkg_weights, _ = raw2outputs(bkg_out, bkg_z_vals, bkg_dirs[:, 0, :], white_bkg=white_bkg)
                bkg_pts, bkg_dirs, bkg_z_vals = ray_utils.ray_to_importance_samples(bkg_ray_batch, bkg_z_vals, bkg_weights, importance_samples_per_ray, device=device)
                bkg_out = bkg_model.fine_bkg_net(
                    bkg_pts,
                    bkg_dirs
                )
            human_out_dict = {
                'out': [],
                'z_val': [],
            }
            for _net, _posed_verts, _faces, _Ts in zip(human_models, posed_verts, faces, Ts):
                temp_near, temp_far = ray_utils.geometry_guided_near_far(origins[i:i + rays_per_batch], dirs[i:i + rays_per_batch], _posed_verts, geo_threshold)
                # generate ray samples
                num_empty_rays = bkg_pts.shape[0]
                empty_human_out = torch.zeros([num_empty_rays, samples_per_ray, 4], device=device)
                empty_human_z_vals = torch.stack([torch.linspace(cap.far['bkg'] * 2, cap.far['bkg'] * 3, samples_per_ray)] * num_empty_rays).to(device)
                if (temp_near < temp_far).any():
                    human_ray_batch = build_batch(
                        origins[i:i + rays_per_batch][temp_near < temp_far],
                        dirs[i:i + rays_per_batch][temp_near < temp_far],
                        temp_near[temp_near < temp_far],
                        temp_far[temp_near < temp_far]
                    )
                    human_pts, human_dirs, human_z_vals = ray_utils.ray_to_samples(human_ray_batch, samples_per_ray, device=device)
                    can_pts, can_dirs, _ = ray_utils.warp_samples_to_canonical_cpu(
                        human_pts.cpu().numpy(),
                        _posed_verts,
                        _faces,
                        _Ts
                    )
                    can_pts = torch.from_numpy(can_pts).to(device).float()
                    can_dirs = torch.from_numpy(can_dirs).to(device).float()
                    human_out = _net.coarse_human_net(can_pts, can_dirs)
                    empty_human_out[temp_near < temp_far] = human_out
                    empty_human_z_vals[temp_near < temp_far] = human_z_vals
                human_out_dict['out'].append(empty_human_out)
                human_out_dict['z_val'].append(empty_human_z_vals)
            coarse_total_zvals, coarse_order = torch.sort(torch.cat([bkg_z_vals, torch.cat(human_out_dict['z_val'], 1)], -1), -1)
            coarse_total_out = torch.cat([bkg_out, torch.cat(human_out_dict['out'], 1)], 1)
            _b, _n, _c = coarse_total_out.shape
            coarse_total_out = coarse_total_out[
                torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
                coarse_order.view(_b, _n, 1).repeat(1, 1, _c),
                torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
            ]
            rgb_map, _, _, _, depth_map = raw2outputs(
                coarse_total_out,
                coarse_total_zvals,
                bkg_dirs[:, 0, :],
                white_bkg=white_bkg,
            )
            total_rgb_map.append(rgb_map)
            total_depth_map.append(depth_map)
        total_rgb_map = torch.cat(total_rgb_map).reshape(*cap.shape, -1).detach().cpu().numpy()
        total_depth_map = torch.cat(total_depth_map).reshape(*cap.shape).detach().cpu().numpy()
        if return_depth:
            return total_rgb_map, total_depth_map
        return total_rgb_map


def phong_renderer_from_pinhole_cam(cam, device='cpu'):
    focal_length = torch.tensor([[cam.fx, cam.fy]])
    principal_point = torch.tensor([[cam.width - cam.cx, cam.height - cam.cy]])  # In PyTorch3D, we assume that +X points left, and +Y points up and +Z points out from the image plane.
    image_size = torch.tensor([[cam.height, cam.width]])
    cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point, in_ndc=False, image_size=image_size, device=device)
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    raster_settings = RasterizationSettings(
        image_size=(cam.height, cam.width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )
    return silhouette_renderer


def overlay_smpl(img, verts, faces, cap):
    device = verts.device
    renderer = phong_renderer_from_pinhole_cam(cap.pinhole_cam, device=device)
    R = torch.from_numpy(cap.cam_pose.rotation_matrix[:3, :3].T)[None].to(device)
    T = torch.from_numpy(cap.cam_pose.translation_vector)[None].to(device)
    mesh_col = torch.ones_like(verts)[None].to(device)
    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=TexturesVertex(verts_features=mesh_col)
    )
    silhouette = renderer(meshes_world=mesh, R=R, T=T)
    silhouette = torch.rot90(silhouette[0].cpu().detach(), k=2).numpy()
    silhouette = Image.fromarray(np.uint8(silhouette * 255))
    bkg = Image.fromarray(np.concatenate([img, np.ones_like(img[..., 0:1]) * 255], axis=-1))
    overlay = Image.alpha_composite(bkg, silhouette)
    return np.array(overlay)[..., :3]

def select_background(shape, key) -> torch.Tensor:
    """
    randomly geneate a back ground
    :param shape: (height * width, 3)
    :return: background rgb, (height * width, 3)
    """
    #!HARDCODED Oct 02: assume 4 types
    key = key%4
    if key == WHITE_BKG:
        background = torch.ones(shape)

    elif key == BLACK_BKG:
        background = torch.zeros(shape)
    elif key == NOISE_BKG:
        # gaussian noise
        background = torch.ones(shape[0])
        background = torch.nn.init.normal_(background, mean=0.5, std=0.1)
        background = torch.clamp(background, 0, 1)
        background = einops.repeat(background, 'n -> n c', c=3)
    elif key == CHESSBOARD_BKG:
        # avatarclip chessboard
        #!HARDCODED Oct 18: assume sqrt is integer
        H, W = int(np.sqrt(shape[0])), int(np.sqrt(shape[0]))
        chess_board = torch.zeros([H, W, 1]) + 0.2
        # do not randomize the chessboard size
        chess_length = H // 10
        # chess_length = H // np.random.choice(np.arange(10,20))
        i, j = np.meshgrid(np.arange(H, dtype=np.int32), np.arange(W, dtype=np.int32), indexing='xy')
        div_i, div_j = i // chess_length, j // chess_length
        white_i, white_j = i[(div_i + div_j) % 2 == 0], j[(div_i + div_j) % 2 == 0]
        chess_board[white_i, white_j] = 0.8
        blur_fn = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
        background= blur_fn(chess_board.unsqueeze(0).permute(0, 3, 1, 2)).squeeze(0).permute(1, 2, 0)
        background = einops.repeat(background, "h w 1-> (h w) 3")
    return background

def sparse_ray_sampling(rays_o:torch.Tensor,rays_d:torch.Tensor, stride:int = 1):
    """
    Sparsely sample rays in given bunch of rays.

    Args:
        rays_o: [H, W ,3]
        rays_d: [H, W ,3]
        max_ray_num: int

    Returns:
        rays_o: [h, w, 3]
        rays_d: [h, w, 3]
    """
    assert stride > 0
    if stride == 1:
        return rays_o, rays_d
    max_offset = stride - 1
    #begin at random top-left pixel
    x_off, y_off = random.randint(0, max_offset), random.randint(0, max_offset) 
    # x_off, y_off = 0, 0
    rays_o, rays_d = rays_o[x_off::stride,y_off::stride,...],rays_d[x_off::stride,y_off::stride,...]

    return rays_o, rays_d

def visualize_poses(poses):
    """
    visualize poses
    :param poses: (batch_size, 24, 3)
    :return: None
    """
    poses_raw = []
    for p in poses:
        poses_raw.append(p.cpu().numpy())
    pcd = o3d.geometry.PointCloud()
    #extract translation part only

def composite_by_depth(img_a, img_b, depth_a, depth_b):
    """
    Composite two images by depth.
    img: Ndarray [N,3]
    depth: Ndarray [N]
    """
    assert img_a.shape == img_b.shape
    assert img_a.shape[0] == depth_a.shape[0]
    mask_a = (depth_a < depth_b)
    mask_b = (depth_a >= depth_b)
    #[N,] -> [N,1]
    mask_a = mask_a[...,None]
    mask_b = mask_b[...,None]
    return img_a * mask_a + img_b * mask_b
