import torch
import json
import imageio
import numpy as np
import os
import cv2 as cv
from scipy import ndimage

class SMPLDataset:
    def __init__(self, path):
        super(SMPL_Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')

        self.data_dir = path
        with open(os.path.join(self.data_dir, 'transforms_train.json'), 'r') as fp:
            meta = json.load(fp)
        
        self.images = []
        self.poses = []
        self.images_lis = []
        
        for frame in meta['frames']:
            fname = os.path.join(self.data_dir, frame['file_path'] + '.png')
            self.images.append(imageio.imread(fname))
            self.images_lis.append(fname)
            self.poses.append(np.array(frame['transform_matrix']))
        
        self.n_images = len(self.images)
        self.images = (np.array(self.images) / 255.).astype(np.float32)
        self.images = self.images[:, :, ::-1]
        self.images = torch.from_numpy(self.images.copy()).cpu()
        self.masks = torch.zeros_like(self.images)
        self.masks[self.images != 0] = 1.0

        self.poses = np.array(self.poses).astype(np.float32)
        self.poses = torch.from_numpy(self.poses).to(self.device)

        self.H, self.W = self.images[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.W / np.tan(.5 * camera_angle_x)
        # self.render_poses = torch.stack([pose_spherical(90, angle, 2.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        self.image_pixels = self.H * self.W

        self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
        self.object_bbox_max = np.array([ 1.01,  1.01,  1.01])

        self.K = np.array([
            [self.focal, 0,          0.5*self.W],
            [0,          self.focal, 0.5*self.H],
            [0,          0,          1         ]
        ])
        self.K = torch.from_numpy(self.K).cpu()

        print('Load data: End')
    
    def gen_rays_silhouettes(self, pose, max_ray_num, mask):
        if mask.sum() == 0:
            return self.gen_rays_pose(pose, resolution_level=4)
        struct = ndimage.generate_binary_structure(2, 2)
        dilated_mask = ndimage.binary_dilation(mask, structure=struct, iterations=10).astype(np.int32)
        current_ratio = dilated_mask.sum() / float(mask.shape[0] * mask.shape[1])
        W = H = min(self.H, int(np.sqrt(max_ray_num / current_ratio)))
        tx = torch.linspace(0, self.W - 1, W)
        ty = torch.linspace(0, self.H - 1, H)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.t()
        pixels_y = pixels_y.t()
        p = torch.stack([(pixels_x - self.K[0][2]) / self.K[0][0],
                         -(pixels_y - self.K[1][2]) / self.K[1][1],
                         -torch.ones_like(pixels_x)], -1).float()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.sum(rays_v[..., None, :] * pose[:3, :3], -1)
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        resized_dilated_mask = torch.nn.functional.interpolate(
            torch.from_numpy(dilated_mask).reshape(256, 256, 1).permute(2, 0, 1).unsqueeze(0).float(), size=(H, W)).squeeze()
        masked_rays_v = rays_v[resized_dilated_mask > 0]
        masked_rays_o = rays_o[resized_dilated_mask > 0]

        return masked_rays_o, masked_rays_v, W, resized_dilated_mask > 0

    def gen_rays_pose(self, pose, resolution_level=1):
        """
        Generate rays at world space given pose.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, int(self.W // l))
        ty = torch.linspace(0, self.H - 1, int(self.H // l))
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.t().to(self.device)
        pixels_y = pixels_y.t().to(self.device)
        p = torch.stack([(pixels_x - self.K[0][2]) / self.K[0][0],
                         -(pixels_y - self.K[1][2]) / self.K[1][1],
                         -torch.ones_like(pixels_x).to(self.device)], -1).float()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.sum(rays_v[..., None, :] * pose[:3, :3], -1)
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o, rays_v

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, int(self.W // l))
        ty = torch.linspace(0, self.H - 1, int(self.H // l))
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.t()
        pixels_y = pixels_y.t()
        p = torch.stack([(pixels_x - self.K[0][2]) / self.K[0][0],
                         -(pixels_y - self.K[1][2]) / self.K[1][1],
                         -torch.ones_like(pixels_x)], -1).float()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = rays_v.to(self.device)
        # rays_v = torch.matmul(self.poses[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = torch.sum(rays_v[..., None, :] * self.poses[img_idx, :3, :3], -1)
        rays_o = self.poses[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o, rays_v

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([(pixels_x - self.K[0][2]) / self.K[0][0],
                         -(pixels_y - self.K[1][2]) / self.K[1][1],
                         -torch.ones_like(pixels_x)], -1).float()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        # rays_v = torch.matmul(self.poses[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_v = torch.sum(rays_v[..., None, :] * self.poses[img_idx, :3, :3], -1)
        rays_o = self.poses[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def near_far_from_sphere(self, rays_o, rays_d, is_sphere=False):
        # if not is_sphere:
        #     return 0.5, 3
        # else:
        #     return 0.5, 1
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1
        near[near < 0] = 0
        far = mid + 1
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        img = img[:, ::-1, :]
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
