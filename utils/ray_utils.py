# Code based on nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
# License from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch/blob/master/LICENSE


import numpy as np
import torch
import igl

from geometry.pcd_projector import PointCloudProjectorNp
from utils.constant import DEFAULT_GEO_THRESH, PERTURB_EPSILON


def shot_ray(cap, x, y):
    z = np.array([[1]])
    xy = np.array([[x, y]])
    pcd_3d = PointCloudProjectorNp.pcd_2d_to_pcd_3d(xy, z, cap.intrinsic_matrix, cam2world=cap.cam_pose.camera_to_world)[0].astype(np.float32)
    orig = cap.cam_pose.camera_center_in_world
    dir = pcd_3d - orig
    dir = dir / np.linalg.norm(dir)
    return orig, dir




def shot_rays(cap, xys):
    z = np.ones((xys.shape[0], 1))
    # r = cap.cam_pose.rotation_matrix
    # t = cap.cam_pose.translation_vector
    c2w = cap.cam_pose.camera_to_world
    # rotate 10 degree around y axis


    pcd_3d = PointCloudProjectorNp.pcd_2d_to_pcd_3d(xys, z, cap.intrinsic_matrix, cam2world=c2w).astype(np.float32)
    orig = np.stack([cap.cam_pose.camera_center_in_world] * xys.shape[0])
    dir = pcd_3d - orig
    dir = dir / np.linalg.norm(dir, axis=1, keepdims=True)
    return orig, dir


def shot_all_rays(cap):
    c2w = cap.cam_pose.camera_to_world
    temp_pcd = PointCloudProjectorNp.img_to_pcd_3d(np.ones(cap.size), cap.intrinsic_matrix, img=None, cam2world=c2w)
    dirs = temp_pcd - cap.cam_pose.camera_center_in_world
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    origs = np.stack([cap.cam_pose.camera_center_in_world] * dirs.shape[0], axis=0)
    return origs, dirs


def to_homogeneous(pts):
    if isinstance(pts, torch.Tensor):
        return torch.cat([pts, torch.ones_like(pts[..., 0:1])], axis=-1)
    elif isinstance(pts, np.ndarray):
        return np.concatenate([pts, np.ones_like(pts[..., 0:1])], axis=-1)

# def warp_samples_to_canonical(pts, verts, faces, T):
#     if pts.is_cuda and verts.is_cuda and faces.is_cuda:
#         return warp_samples_to_canonical_cuda(pts, verts, faces, T)
#     else:
#         return warp_samples_to_canonical_cpu(pts, verts, faces, T)


def warp_samples_to_canonical(pts, verts, faces, T, threshold=0.2):
    """
    warp points from world space to canonical space, guided by the SMPL mesh
    """
    assert len(pts.shape) == 3, 'pts should have shape [num_rays, num_samples, 3]'
    assert pts.shape[-1] == 3
    num_rays, num_samples, _ = pts.shape
    pts = pts.reshape(-1, 3)
    dist2, f_id, closest = igl.point_mesh_squared_distance(pts, verts, faces[:, :3])
    
    # dist2: list of smallest squared distances
    # mask according to the threshold
    mask = dist2 < threshold

    closest_tri = verts[faces[:, :3][f_id]]
    barycentric = igl.barycentric_coordinates_tri(closest, closest_tri[:, 0, :].copy(), closest_tri[:, 1, :].copy(), closest_tri[:, 2, :].copy())
    T_interp = (T[faces[:, :3][f_id]] * barycentric[..., None, None]).sum(axis=1)
    T_interp_inv = np.linalg.inv(T_interp)


    can_pts = (T_interp_inv @ to_homogeneous(pts)[..., None])[:, :3, 0]
    # reshape back and compute ray directions in canonical space
    can_pts = can_pts.reshape(num_rays, num_samples, 3)
    closest = closest.reshape(num_rays, num_samples, 3)
    can_dirs = can_pts[:, 1:] - can_pts[:, :-1]
    can_dirs = np.concatenate([can_dirs, can_dirs[:, -1:]], axis=1)
    can_dirs = can_dirs / np.linalg.norm(can_dirs, axis=2, keepdims=True)

    return can_pts, can_dirs, closest, mask

# @torch.no_grad()
# def warp_samples_to_canonical_cuda(pts, verts, faces, T, threshold=0.005):
#     """
#     warp points from world space to canonical space, guided by the SMPL mesh
#     """
#     assert len(pts.shape) == 3, 'pts should have shape [num_rays, num_samples, 3]'
#     assert pts.shape[-1] == 3
#     num_rays, num_samples, _ = pts.shape
#     if not isinstance(T, torch.Tensor):
#         T = torch.from_numpy(T).float().to(pts.device)
#     # pts = pts.reshape(-1, 3)
#     # seems due to bug in torch-knn, verts and pts must be in minibatch format
#     pts = pts.permute(1, 0, 2).contiguous()
#     verts = verts.repeat(num_samples, 1, 1) 
#     if True:
#         if len(pts.shape) == 2 and len(verts.shape) == 2:
#             pts, verts = pts.unsqueeze(0), verts.unsqueeze(0)
#         knn = KNN(k=4, transpose_mode=True)
#         # dist2, f_id = knn(pts, verts)

#         # ref = torch.rand(1, 524288, 3).cuda()
#         # query = torch.rand(1, 6890, 5).cuda()

#         dist, indx = knn(verts, pts)


#     mask = dist.mean(axis=2) < threshold

#     # normalize distance
#     dist = dist / dist.sum(axis=2, keepdims=True)

#     # interpolate T, weighted by distance to each vertex
#     T_knn = T[indx]
#     T_interp = (T_knn * dist[..., None, None]).sum(axis=2)
#     T_interp_inv = torch.inverse(T_interp).reshape(-1, 4, 4)

#     pts = pts.reshape(-1, 3)
#     mask = mask.reshape(-1)


#     can_pts = (T_interp_inv @ to_homogeneous(pts)[..., None])[:, :3, 0]
#     # reshape back and compute ray directions in canonical space
#     can_pts = can_pts.reshape(num_rays, num_samples, 3)
#     # closest = closest.reshape(num_rays, num_samples, 3)
#     can_dirs = can_pts[:, 1:] - can_pts[:, :-1]
#     can_dirs = torch.cat([can_dirs, can_dirs[:, -1:]], dim=1)
#     can_dirs = can_dirs / torch.norm(can_dirs, dim=2, keepdim=True)

#     return can_pts, can_dirs, None, mask

def warp_samples_to_canonical_diff(pts, verts, faces, T):
    signed_dist, f_id, closest = igl.signed_distance(pts, verts.detach().cpu().numpy(), faces[:, :3])

    # differentiable barycentric interpolation
    closest_tri = verts[faces[:, :3][f_id]]
    closest = torch.from_numpy(closest).float().to(verts.device)
    v0v1 = closest_tri[:, 1] - closest_tri[:, 0]
    v0v2 = closest_tri[:, 2] - closest_tri[:, 0]
    v1v2 = closest_tri[:, 2] - closest_tri[:, 1]
    v2v0 = closest_tri[:, 0] - closest_tri[:, 2]
    v1p = closest - closest_tri[:, 1]
    v2p = closest - closest_tri[:, 2]
    N = torch.cross(v0v1, v0v2)
    denom = torch.bmm(N.unsqueeze(dim=1), N.unsqueeze(dim=2)).squeeze()
    C1 = torch.cross(v1v2, v1p)
    u = torch.bmm(N.unsqueeze(dim=1), C1.unsqueeze(dim=2)).squeeze() / denom
    C2 = torch.cross(v2v0, v2p)
    v = torch.bmm(N.unsqueeze(dim=1), C2.unsqueeze(dim=2)).squeeze() / denom
    w = 1 - u - v
    barycentric = torch.stack([u, v, w], dim=1)

    T_interp = (T[faces[:, :3][f_id]] * barycentric[..., None, None]).sum(axis=1)
    T_interp_inv = torch.inverse(T_interp)

    return T_interp_inv, f_id, signed_dist


def ray_to_samples(ray_batch,
                   samples_per_ray,
                   lindisp=False,
                   perturb=0.,
                   device='cpu',
                   append_t=None
                   ):
    '''
    reference: https://github.com/yenchenlin/nerf-pytorch
    '''
    rays_per_batch = ray_batch['origin'].shape[0]
    rays_o, rays_d = ray_batch['origin'], ray_batch['direction']  # [rays_per_batch, 3] each
    near, far = ray_batch['near'], ray_batch['far']  # [-1,1]
    assert near.shape[0] == far.shape[0] == rays_per_batch

    t_vals = torch.linspace(0., 1., steps=samples_per_ray, device=device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.clip(
            torch.rand(z_vals.shape, device=device),
            min=PERTURB_EPSILON,
            max=1-PERTURB_EPSILON
        )

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [rays_per_batch, samples_per_ray, 3]
    dirs = torch.stack([rays_d] * samples_per_ray, axis=1)
    if append_t is not None:
        pts = torch.cat([pts, append_t.to(device)], dim=-1)
    return pts, dirs, z_vals


def ray_to_importance_samples(ray_batch,
                              z_vals,
                              weights,
                              importance_samples_per_ray,
                              device='cpu',
                              including_old=True,
                              append_t=None
                              ):
    rays_o, rays_d = ray_batch['origin'].to(device), ray_batch['direction'].to(device)

    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], importance_samples_per_ray, det=True, device=device)
    z_samples = z_samples.detach()
    if including_old:
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    else:
        z_vals = z_samples
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    total_samples = pts.shape[1]
    dirs = torch.stack([rays_d] * total_samples, axis=1)
    if append_t is not None:
        pts = torch.cat([pts, append_t.to(device)], dim=-1)
    return pts, dirs, z_vals


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, device='cpu'):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def geometry_guided_near_far(orig, dir, vert, geo_threshold):
    if isinstance(orig, torch.Tensor):
        return geometry_guided_near_far_torch(orig, dir, vert, geo_threshold)
    elif isinstance(orig, np.ndarray):
        return geometry_guided_near_far_np(orig, dir, vert, geo_threshold)


def geometry_guided_near_far_torch(orig, dir, vert, geo_threshold=DEFAULT_GEO_THRESH):
    if not isinstance(vert, torch.Tensor):
        vert = torch.tensor(vert, dtype=torch.float32).to(orig.device)
    num_vert = vert.shape[0]
    num_rays = orig.shape[0]
    orig_ = torch.repeat_interleave(orig[:, None, :], num_vert, 1)
    dir_ = torch.repeat_interleave(dir[:, None, :], num_vert, 1)
    vert_ = torch.repeat_interleave(vert[None, ...], num_rays, 0)
    orig_v = vert_ - orig_
    z0 = torch.einsum('ij,ij->i', orig_v.reshape(-1, 3), dir_.reshape(-1, 3)).reshape(num_rays, num_vert)
    dz = torch.sqrt(geo_threshold**2 - (torch.norm(orig_v, dim=2)**2 - z0**2))
    near = z0 - dz
    near[near != near] = float('inf')
    near = near.min(dim=1)[0]
    far = z0 + dz
    far[far != far] = float('-inf')
    far = far.max(dim=1)[0]
    return near, far


def geometry_guided_near_far_np(orig, dir, vert, geo_threshold=DEFAULT_GEO_THRESH):
    num_vert = vert.shape[0]
    num_rays = orig.shape[0]
    orig_ = np.repeat(orig[:, None, :], num_vert, 1)
    dir_ = np.repeat(dir[:, None, :], num_vert, 1)
    vert_ = np.repeat(vert[None, ...], num_rays, 0)
    orig_v = vert_ - orig_
    z0 = np.einsum('ij,ij->i', orig_v.reshape(-1, 3), dir_.reshape(-1, 3)).reshape(num_rays, num_vert)
    dz = np.sqrt(geo_threshold**2 - (np.linalg.norm(orig_v, axis=2)**2 - z0**2))
    near = np.nan_to_num(z0-dz, nan=np.inf).min(axis=1)
    far = np.nan_to_num(z0+dz, nan=-np.inf).max(axis=1)
    return near, far




def sphere_coord(theta, phi, r = 1.0):
    return np.array([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    ])

def near_far_from_bound(rays_o, rays_d, bound, type='cube'):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)

    if type == 'sphere':
        near = radius - bound # [B, N, 1]
        far = radius + bound

    elif type == 'cube':
        # TODO: if bound < radius, some rays may not intersect with the bbox. (should set near = far = inf ... or far < near)
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        near = torch.clamp(near, min=0.05)

    return near, far

def batch_add_translation(Ts, transl):
    """
    add the translation part to the transformation matrix
    [R | t   +  [x | y | z]   =   [R |  t]
     0 | 1]                        0 | 1]
    Ts: [B, 4, 4]
    transl: [B, 3]
    """
    assert Ts.shape[0] == transl.shape[0] 
    delta_T = np.zeros_like(Ts)
    delta_T[:, :3, 3] = transl
    Ts = Ts + delta_T
    return Ts