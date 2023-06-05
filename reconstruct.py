# -*- coding : utf-8 -*-
# @FileName  : stylize_canonical.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Sep 25, 2022
# @Github    : https://github.com/songrise
# @Description: Use neus to reconstruct the canonical avatar, 
import os
import argparse

import imageio
import torch
import torch.nn.functional as F

from models import  instant_nsr
from utils import render_utils, utils, debug_utils
from utils.SMPLDataset import SMPLDataset
from options import options
from utils.constant import WHITE_BKG, BLACK_BKG, NSR_BOUND
import einops

#! Sep 25:  Overwrite for larger human
####DEBUG Variables######
CANONICAL_CAMERA_DIST_VAL = 1.6
all_gt = []
STYLE_CANONICAL = True



def main_reconstruct(opt):

    utils.print_notification('reconstruct the canonical view of a human.')
    utils.fix_randomness(42)
    device = torch.device('cuda' if opt.use_cuda else 'cpu')
    #! Sep 28: Check for canonical setting

    nerf = instant_nsr.NeRFNetwork()
    nerf = nerf.to(device)
    print('Model setup done.')

    ###prepare data####

    #! Sep 25: load poses from pickled file
    #! Oct 01: generate cam pose in runtime
    H, W = opt.render_h, opt.render_w



    #! Dec 09: instant-nsr opt
    optimizer = torch.optim.Adam(nerf.parameters(), lr=5e-4, betas=(0.9, 0.99), eps=1e-15)
    #!HARDCODED Oct 02: decay to half 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.lr//20)

    if opt.data_format == "neuman":
      raise NotImplementedError
    elif opt.data_format == 'neus':
        
        dataloader = SMPLDataset(opt.data_path)
        gt_rgb = dataloader.images #n_cap x H x W x 3
        poses = dataloader.poses   #n_cap x 4 x 4
        H, W = gt_rgb.shape[1], gt_rgb.shape[2]

        gt_rgb_orig = gt_rgb.reshape(-1,3).to(device) #[n_cap x H x W , 3]
        all_rays_o, all_rays_d = [], []
        for i in range(poses.shape[0]):
            rays_o, rays_d = dataloader.gen_rays_pose(poses[i,...])
            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)
        all_rays_o_orig = torch.stack(all_rays_o).to(device).reshape(-1,3) #[n_cap x H x W , 3]
        all_rays_d_orig = torch.stack(all_rays_d).to(device).reshape(-1,3) #[n_cap x H x W , 3]


       


    batch_size = 1600
    #!#######Main Train loop########
    global_step = 0
    print('Start training...')
    for epoch in range(opt.epochs):
        #permute rays
        perm = torch.randperm(all_rays_o_orig.shape[0])
        all_rays_o = all_rays_o_orig[perm]
        all_rays_d = all_rays_d_orig[perm]
        gt_rgb = gt_rgb_orig[perm]
        for i in range(0,all_rays_o.shape[0],batch_size):

            rays_o, rays_d = all_rays_o[i:i+batch_size], all_rays_d[i:i+batch_size]

            with torch.enable_grad():

                rgb_gt_patch = gt_rgb[i:i+batch_size,...]

                loss = 0.0

                if opt.implicit_model == 'instant_nsr':
                    rgb_pred_patch, eikonal_loss, _ = render_utils.render_instantnsr_naive(nerf, rays_o = rays_o, rays_d = rays_d,
                        render_can=STYLE_CANONICAL, requires_grad=True,  bkg_key= WHITE_BKG if opt.white_bkg else BLACK_BKG,
                        return_torch = True, rays_per_batch = batch_size, perturb = 1.0,return_raw=True, bound = NSR_BOUND) 

                elif opt.implicit_model == 'neus':
                    raise NotImplementedError
                rgb_pred_patch = rgb_pred_patch.squeeze(0)
  
                optimizer.zero_grad()

                loss = loss + F.smooth_l1_loss(rgb_pred_patch, rgb_gt_patch,reduction='mean')
                loss = loss + eikonal_loss * 0.1

                if i % 32000 == 0:
                    print ("loss: ", loss.item())
                loss.backward()
                optimizer.step()
 

            #!####Logging and saving#####
            if global_step == 0 or (global_step+1) % opt.i_val == 0:
                # log rgb
                #todo revise val_cap gen logic to avoid repeated calc
                if opt.data_format == "neuman":
                    raise NotImplementedError
                elif opt.data_format == "neus":
                    rays_o_val, rays_d_val = dataloader.gen_rays_at(62)
                    rays_o_val, rays_d_val = rays_o_val.reshape(-1,3), rays_d_val.reshape(-1,3) #[H*W, 3]


                #! Dec 09: instant nsr val
                if opt.implicit_model == 'instant_nsr':
                    with torch.no_grad():
                        rgb_val, _, __ = render_utils.render_instantnsr_naive(nerf, rays_o = rays_o_val, rays_d = rays_d_val,
                        render_can=STYLE_CANONICAL, requires_grad=False,  bkg_key= WHITE_BKG if opt.white_bkg else BLACK_BKG,
                        return_torch = True, rays_per_batch = 4096, perturb = True, return_raw=True) 

                        rgb_val = rgb_val.squeeze(0).cpu()
                        rgb_val = einops.rearrange(rgb_val, '(h w) c -> h w c', h=H, w=W)
                elif opt.implicit_model == 'neus':
                    raise NotImplementedError

                save_path = os.path.join('./style', 'canonical_360', opt.exp_name, f'{opt.exp_name}_{str(global_step+1).zfill(4)}.png')
                if not os.path.isdir(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                imageio.imsave(save_path, rgb_val)
                print(f'image saved: {save_path}')
            # log weights
            if (global_step+1) % opt.i_save == 0:
                weights_path = os.path.join('./style', 'canonical_360', opt.exp_name, f'{opt.exp_name}_{str(global_step+1).zfill(4)}.pth.tar')
                if not os.path.isdir(os.path.dirname(weights_path)):
                    os.makedirs(os.path.dirname(weights_path))
                torch.save(nerf.state_dict(), weights_path)
                print(f'weights saved: {weights_path}')

            # log mesh
            if (global_step+1) % opt.i_mesh == 0:
                mesh_path = os.path.join('./style', 'canonical_360', opt.exp_name, f'{opt.exp_name}_{str(global_step+1).zfill(4)}.ply')
                if opt.implicit_model == 'neus':
                    raise NotImplementedError
                elif opt.implicit_model == 'instant_nsr':
                    #!HARDCODED Nov 15: https://github.com/Totoro97/NeuS/blob/main/models/dataset.py#L90
                    #TODO Dec 09: extract_geo change bound impl for instant-nsr, also change for others later.
                    vert, face = nerf.extract_geometry(NSR_BOUND, 512, device=device)
                utils.save_mesh(vert, face, mesh_path)

            global_step += 1
        scheduler.step()
        print("Current learning rate: {}".format(scheduler.get_last_lr()))
        debug_utils.dump_tensor(all_gt, 'all_gt.pkl')
    utils.print_notification(f'Finished training {opt.exp_name}.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options.set_general_option(parser)
    opt, _ = parser.parse_known_args()

    options.set_nerf_option(parser)
    options.set_pe_option(parser)
    options.set_render_option(parser)
    options.set_trajectory_option(parser)
    parser.add_argument('--scene_dir', required=True, type=str, help='scene directory')
    parser.add_argument('--image_dir', required=False, type=str, default=None, help='image directory')
    parser.add_argument('--out_dir', default='./out', type=str, help='weights dir')
    parser.add_argument('--offset_scale', default=1.0, type=float, help='scale the predicted offset')
    parser.add_argument('--geo_threshold', default=-1, type=float, help='')
    parser.add_argument('--normalize', default=True, type=options.str2bool, help='')
    parser.add_argument('--bkg_range_scale', default=3, type=float, help='extend near/far range for background')
    parser.add_argument('--human_range_scale', default=1.5, type=float, help='extend near/far range for human')
    parser.add_argument('--mode', required=True, choices=['canonical_360', 'posed_360', 'style'], default = 'style',type=str, help='rendering mode')
    parser.add_argument('--num_offset_nets', default=1, type=int, help='how many offset networks')
    parser.add_argument('--offset_scale_type', default='linear', type=str, help='no/linear/tanh')
    parser.add_argument('--data_format', default = "neuman", choices=["neuman","neus"], type=str, help='dataset type')
    parser.add_argument('--data_path', default="data/da_09", type = str)
    #########Stylization###########
    #nerf related
    parser.add_argument('--n_sample', default=64, type=int, help='number of pts to sample along a ray')
    parser.add_argument('--model_config', default='config/instant_nsr.json', type=str, help='model config file')
    parser.add_argument('--implicit_model', default='instant_nsr', type=str, choices = ["neus", "instant_nsr"], help='implicit model')
    #! Oct 05: to set up hw, use --render_h and --render_w instead.
    # parser.add_argument('--img_hw', default = [160, 160], type=list, help='image height and width')
    #clip loss related
    parser.add_argument('--canonical_path',required=True, default=None, type=str, help='path to canonical image')

    #regularization related

    parser.add_argument("--exp_name", type=str, default="recon", help="name of the experiment")
    parser.add_argument('--epochs',default=2,type = int, help='number of epochs')
    parser.add_argument('--lr',default=5e-4,type = float, help='learning rate')
    parser.add_argument('--i_val',default=10,type = float, help='log image after i_val iterations')
    parser.add_argument('--i_save',default=200,type = float, help='save model after i_save iterations')
    parser.add_argument('--i_mesh',default=1000,type = int, help='save mesh after i_mesh iterations')


    opt = parser.parse_args()

    if opt.render_h is None:
        opt.render_size = None
    else:
        opt.render_size = (opt.render_h, opt.render_w)

    options.print_opt(opt)
    #TODO Oct 01: only do style in this file
    main_reconstruct(opt)
