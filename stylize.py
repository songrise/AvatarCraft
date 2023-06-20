# -*- coding : utf-8 -*-
# @FileName  : stylize_canonical.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Sep 25, 2022
# @Github    : https://github.com/songrise
# @Description: Stylize the canonical view of a human, represented as a Instant-NeuS model.
#                Use grad checkpointing to get gradient over whole image first, then update model by patch-rendering.

import os
import argparse

import imageio
import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from models import instant_nsr, diffusion
from utils import render_utils, utils, debug_utils
from options import options
from utils.constant import CANONICAL_CAMERA_DIST_TRAIN, CANONICAL_CAMERA_DIST_VAL, WHITE_BKG, BLACK_BKG, NOISE_BKG, CHESSBOARD_BKG, CAN_HEAD_OFFSET, CAN_HEAD_CAMERA_DIST, NSR_BOUND
from tqdm import  trange
import einops
import random


####DEBUG Variables######
STYLE_CANONICAL = True
fix_geo = False


class Trainer():
    def __init__(self, opt):
  
        utils.fix_randomness(42)
        utils.print_notification('Generate avatar with text prompt.')
        self.opt = opt

        self.device = torch.device('cuda' if opt.use_cuda else 'cpu')
  
        self.load_scene()
        self.net_gt, self.net_style = self.setup_model()
        self.loss = self.setup_loss()
        self.optimizer, self.scheduler = self.setup_optimizer()



    def train(self):
        global_step = 0
        H, W = self.H, self.W
        opt = self.opt
        n_epoch = opt.coarse_epochs + opt.fine_epochs
        render_args_val = {
            "render_h" : self.H,
            "render_w" : self.W,

        }
        self.log_img(render_args_val, 0)

        for epoch in range(n_epoch):
            # genearte poses for each epoch
            if opt.augment_cam:
                style_head = opt.stylize_head
                head_rate = opt.coarse_head if epoch < opt.coarse_epochs else opt.fine_head
                render_poses, desc = render_utils.style_360_path(self.center, self.up, CANONICAL_CAMERA_DIST_TRAIN, self.n_cap, add_noise = True, noise_scale=2.0, style_head=style_head, head_offset = CAN_HEAD_OFFSET, head_rate=head_rate, head_dist=CAN_HEAD_CAMERA_DIST)
            else:
                #TODO Feb 23: simplify this if else
                if opt.stylize_head:
                    head_rate = opt.coarse_head if epoch < opt.coarse_epochs else opt.fine_head
                    render_poses, desc = render_utils.style_360_path(self.center, self.up, CANONICAL_CAMERA_DIST_TRAIN, self.n_cap, add_noise = False, style_head=opt.stylize_head, head_offset = CAN_HEAD_OFFSET, head_rate=head_rate, head_dist=CAN_HEAD_CAMERA_DIST)
                else:
                    render_poses, desc = render_utils.default_360_path(self.center, self.up, CANONICAL_CAMERA_DIST_TRAIN, self.n_cap, add_noise = False)
            
            #randomly permute the order of the captures

            n_views = len(render_poses)
            perm = torch.randperm(n_views)
            for _ in trange(n_views, desc='Epoch %d' % epoch):
                i = perm[_]
                
                # generate cap            
                can_cap = render_utils.pose2cap([H, W], render_poses[i])
                rays_o, rays_d = render_utils.cap2rays(can_cap) #H*W, 3

                if opt.augment_bkg:
                    bkg_key = random.randint(WHITE_BKG, NOISE_BKG) #AvatarCLIP style background augmentation (w/o chessboard)
                else:
                    bkg_key = WHITE_BKG if opt.white_bkg else BLACK_BKG 

                tgt_text = self.tgt_text
                if opt.augment_text:
                    tgt_text = f"{desc[i]} {tgt_text}"
                    # print(tgt_text)

            
            
                with torch.enable_grad():
                    # generate all rays in current capture
                    if epoch < opt.coarse_epochs:
                        subsample_scale = opt.subsample_scale
                    else: # for fine stage, the resolution is doubled
                        subsample_scale = min(1, opt.subsample_scale // 2)
                    assert subsample_scale in [1, 2, 4, 8, 16], 'subsample scale must be 1, 2, 4, 8, or 16'

                    rays_o, rays_d = rays_o.reshape(H, W, 3), rays_d.reshape(H, W, 3)
                    rays_o, rays_d= render_utils.sparse_ray_sampling(rays_o, rays_d, subsample_scale) #H, W, 3
                    train_h, train_w = H // subsample_scale, W // subsample_scale
                    rays_o, rays_d = einops.rearrange(rays_o, 'h w c -> (h w) c'), einops.rearrange(rays_d, 'h w c -> (h w) c')

                    render_args_global = {
                        "render_h" : train_h,
                        "render_w" : train_w,

                    }

                    rgb_pred_global, _, extra_out =  self.render_val(rays_o, rays_d, bkg_key = bkg_key, render_args = render_args_global)
                    #!########Style loss#########
                    

                    rgb_pred_global.requires_grad = True # to calculate gradient over the whole img

                    
                    # clip loss is deprecated
                    if opt.guidance_type == "clip" and opt.w_clip > 0:
                        raise NotImplementedError


                    elif opt.guidance_type == "diffusion":
                        text_embedding = self.loss["style"].get_text_embeds([tgt_text])
                        # get grad over the whole image
                        self.loss["style"].mannual_backward(text_embedding, rgb_pred_global, opt.guidance_scale)



                    #!########Backward#########
                    # Nerf-Art style gradient checkpointing
                    rgb_global_grad = rgb_pred_global.grad.clone().detach()
                    rgb_global_grad = einops.rearrange(rgb_global_grad, "1 c h w -> (h w) c")
                    del rgb_pred_global

                    n_rays = train_h * train_w
                    batch_size = min(opt.batch_size, n_rays) 
                    avg_eikonal = []           
                    self.optimizer.zero_grad()


                    # patch-based backward to the NeuS model.
                    with torch.enable_grad():
                        for i in range(0, n_rays, batch_size):
                            rays_o_patch, rays_d_patch = rays_o[i:i+batch_size,...], rays_d[i:i+batch_size,...]
                            if opt.implicit_model == "neus":
                                raise NotImplementedError
                            elif opt.implicit_model == "instant_nsr":
                                rgb_pred_patch, eikonal_loss, extra_out = render_utils.render_instantnsr_naive(
                                    self.net_style, rays_o_patch, rays_d_patch, 
                                    requires_grad = True, bkg_key = bkg_key,
                                    return_torch = True, rays_per_batch = batch_size, 
                                    perturb = 1.0, return_raw=True,
                                    render_can=True, bound = NSR_BOUND)

                                opacity_pred_patch = extra_out["weight_sum"]

                            # backward stylization loss
                            rgb_pred_patch.backward(gradient=rgb_global_grad[i:i+batch_size,...], retain_graph=True)

                            # calculate and backward regularization loss
                            if opt.w_eikonal > 0.0:
                                eikonal_loss = eikonal_loss * opt.w_eikonal 
                                avg_eikonal.append(eikonal_loss.detach().cpu().numpy())
                                eikonal_loss.backward(retain_graph=True)
                                del eikonal_loss


                            if opt.implicit_model == "neus": #if do mask-based geo reg
                                raise NotImplementedError

                            if opt.implicit_model == "instant_nsr":
                                rgb_gt_patch, eikonal_loss, extra_out = render_utils.render_instantnsr_naive(
                                    self.net_gt, rays_o_patch, rays_d_patch, 
                                    requires_grad = True, bkg_key = bkg_key,
                                    return_torch = True, rays_per_batch = batch_size,
                                    perturb = True, return_raw=True,
                                    render_can=True)

                                opacity_gt_patch = extra_out["weight_sum"]
                                opacity_pred_patch = torch.clamp(opacity_pred_patch, 0.0, 1.0)
                                opacity_gt_patch = torch.clamp(opacity_gt_patch, 0.0, 1.0).detach()
                                if opt.guidance_type == "clip":
                                    raise NotImplementedError
                                elif opt.guidance_type == "diffusion":
                                    opacity_loss = F.smooth_l1_loss(opacity_pred_patch, opacity_gt_patch) * 1e5
                                # print("opacity loss", opacity_loss.item())
                                if opt.use_opacity:
                                    opacity_loss.backward(retain_graph=False)
                                del opacity_loss, opacity_pred_patch, opacity_gt_patch, rgb_gt_patch                                

                            del rgb_pred_patch

                    # print("avg eikonal loss: ", np.mean(avg_eikonal))
                    self.optimizer.step()
                
                #!####Logging and saving#####
                if (global_step+1) % opt.i_val == 0:
                    self.log_img(render_args_val, global_step)
 
                # log weights
                if (global_step+1) % opt.i_save == 0:
                    self.log_model(global_step)

                # log mesh
                if (global_step+1) % opt.i_mesh == 0:
                    self.log_mesh(global_step)

                global_step += 1
                # scheduler.step()
            print("Current learning rate: {}".format(self.scheduler.get_last_lr()))
        self.log_model(global_step)
        utils.print_notification(f'Finished training {opt.exp_name}.')
 

    def log_img(self, render_args, step):

        val_poses, _ = render_utils.default_360_path(self.center, self.up, CANONICAL_CAMERA_DIST_VAL, self.n_cap)
        #assume val and train use same resolution
        val_caps = {"body":None}
        val_cap = render_utils.pose2cap([self.H, self.W], val_poses[0])
        val_caps["body"] = val_cap

        if opt.stylize_head:
            head_offset = np.array([0.0, 1.0, 0.0]).astype(np.float32)
            head_offset = head_offset * CAN_HEAD_OFFSET
            val_poses, _ = render_utils.default_360_path(self.center+head_offset, self.up, CAN_HEAD_CAMERA_DIST, self.n_cap)
            val_cap = render_utils.pose2cap([self.H, self.W], val_poses[0])
            val_caps["head"] = val_cap

        for key, cap in val_caps.items():
            
            rays_o_val, rays_d_val = render_utils.cap2rays(cap)
            rgb_val,_, __ = self.render_val(rays_o_val, rays_d_val, bkg_key = WHITE_BKG, render_args = render_args)

            rgb_val = rgb_val.detach().cpu().numpy()
            # squeeze if batch size is 1
            if rgb_val.shape[0] == 1:
                rgb_val = rgb_val.squeeze(0)
            
            if rgb_val.shape[0] == 3:
                rgb_val = rgb_val.transpose(1,2,0)

            rgb_val = utils.integerify_img(rgb_val)
            save_path = os.path.join('./style', 'canonical_360/', opt.exp_name, f'{opt.exp_name}_{str(step+1).zfill(4)}_{key}.png')
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            imageio.imsave(save_path, rgb_val)
            print(f'image saved: {save_path}')

    def log_model(self,step):
        weights_path = os.path.join('./style', 'canonical_360', opt.exp_name, f'{self.opt.exp_name}_{str(step+1).zfill(4)}.pth.tar')
        if not os.path.isdir(os.path.dirname(weights_path)):
            os.makedirs(os.path.dirname(weights_path))
        torch.save(self.net_style.state_dict(), weights_path)
        print(f'weights saved: {weights_path}')


    def log_mesh(self, step):
        device = self.device
        mesh_path = os.path.join('./style', 'canonical_360', self.opt.exp_name, f'{opt.exp_name}_{str(step+1).zfill(4)}.ply')
        if self.opt.implicit_model == "instant_nsr":
            vert, face = self.net_style.extract_geometry(NSR_BOUND, 512, device=device)

        utils.save_mesh(vert, face, mesh_path)

    def render_val(self, rays_o:torch.Tensor, rays_d:torch.Tensor, bkg_key:int = WHITE_BKG, render_args:dict = None):
        """
        render all given rays use network_gt, without recording gradients.
        rays_o: [H*W, 3]


        Returns:
            rgb_pred: [1, H, W, 3]
            eikonal_loss: float
            extra_out: dict, may contain depth_pred, weight_pred
        """
        render_h = render_args["render_h"]
        render_w = render_args["render_w"]
        # cap_id = render_args["cap_id"]
        # scene = render_args["scene"]
        # forward_vert = render_args["forward_vert"]
        # dummy_cap = None


        # this branch is deprecated
        if self.opt.guidance_type == "clip" and opt.clip_type == "dir":
            raise NotImplementedError
        if self.opt.implicit_model == "neus":
            raise NotImplementedError

        elif self.opt.implicit_model == "instant_nsr":
            #TODO May 17: remove hardcoded ray num
            rgb_pred, eikonal_loss, extra_out = render_utils.render_instantnsr_naive(
                self.net_style, rays_o, rays_d, 4096, 
                requires_grad = False, bkg_key = bkg_key,
                return_torch = True, perturb = True, 
                return_raw=True,render_can=True)

        rgb_pred = rgb_pred.clone().detach()
        rgb_pred = rgb_pred.squeeze(0)

        rgb_pred = einops.repeat(rgb_pred, '(h w) c ->1 c h w', h=render_h, w=render_w)
        
        return rgb_pred, eikonal_loss, extra_out

    @debug_utils.log_exec
    def load_scene(self):
        
        ###prepare data####
        
        self.center, self.up = np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        #!HARDCODED Dec 13: 
        self.n_cap = 100
        self.H, self.W = 256, 256


        self.cap_id = 0 #the id of pose, only useful for posed training
        self.tgt_text = opt.tgt_text


    
    @debug_utils.log_exec
    def setup_model(self) ->List[torch.nn.Module]:
        net_gt, net_style = instant_nsr.NeRFNetwork(), instant_nsr.NeRFNetwork()

        net_gt.load_state_dict(torch.load(opt.weights_path))
        net_style.load_state_dict(torch.load(opt.weights_path))

        # this is used to get the gt accumulated opacity. Alternatively, you may 
        # try to implement this as surface rendering of the canonical SMPL mesh.
        net_gt.eval()  

        return [net_gt.cuda(), net_style.cuda()]
    
    @debug_utils.log_exec
    def setup_loss(self):
        """
        load the loss model
        """
        loss_dict = {}
        if opt.guidance_type == "clip":
            raise NotImplementedError
        elif opt.guidance_type == "diffusion":
            style_loss = diffusion.StableDiffusion(self.device, version = opt.sd_version)

        loss_dict['style'] = style_loss
        return loss_dict
        

    def setup_optimizer(self):
        optim_list = [
                {"params": self.net_style.parameters(), "lr": self.opt.lr},
            ]

        optimizer = torch.optim.Adam(optim_list)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.epochs//2, gamma=0.5)

        return optimizer, scheduler






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    options.set_general_option(parser)
    opt, _ = parser.parse_known_args()
    options.set_nerf_option(parser)
    options.set_pe_option(parser)
    options.set_render_option(parser)
    options.set_trajectory_option(parser)
    parser.add_argument('--out_dir', default='./out', type=str, help='weights dir')
    parser.add_argument('--offset_scale', default=1.0, type=float, help='scale the predicted offset')
    parser.add_argument('--geo_threshold', default=-1, type=float, help='')
    parser.add_argument('--normalize', default=True, type=options.str2bool, help='')
    parser.add_argument('--bkg_range_scale', default=3, type=float, help='extend near/far range for background')
    parser.add_argument('--human_range_scale', default=1.5, type=float, help='extend near/far range for human')
    parser.add_argument('--num_offset_nets', default=1, type=int, help='how many offset networks')
    parser.add_argument('--offset_scale_type', default='linear', type=str, help='no/linear/tanh')
    parser.add_argument('--data_format', default = "neuman", choices=["neuman","neus"], type=str, help='dataset format, used for loading camera file')

    #########Stylization###########
    #nerf related
    parser.add_argument('--n_sample', default=64, type=int, help='number of pts to sample along a ray')
    parser.add_argument('--guidance_type', default= "diffusion", choices=["clip", "diffusion"], type=str, help='method to guide stylization')
    parser.add_argument('--tgt_text',default="zombie",type = str,help='target description for the desired avatar, more detailed description is better')
    parser.add_argument('--subsample_scale', default=4, type=int, help='subsample scale for training, must be multiple of 2') 
    parser.add_argument('--stylize_head', default=True, type = options.str2bool, help= "whether to stylize head")
    parser.add_argument('--implicit_model', type = str, default = "instant_nsr", choices = ["neus", "instant_nsr"], help= "which implicit model to use")
    parser.add_argument('--batch_size', type = int, help= "maximum number of rays to be rendered together", default=4096)


    # parser.add_argument('--img_hw', default = [160, 160], type=list, help='image height and width')
    #clip loss related
    parser.add_argument('--clip_type',default="abs",type=str,choices=['dir','abs'], help='directional or absolute CLIP')
    parser.add_argument('--w_clip',default=1.0,type = float, help='weight for CLIP loss')
    parser.add_argument('--w_perp',default=1.0,type = float, help='weight for perceptual loss')
    parser.add_argument('--w_contrast',default=0.2,type = float, help='weight for global contrastive loss')

    #diffusion related
    parser.add_argument("--guidance_scale", default=100, type=float, help="magnitude of the diffusion guidance signal")
    parser.add_argument("--sd_version", default="1.5", type=str, choices=["1.5", "2.0"], help="version of the Stable diffusion")
    # parser.add_argument("--diff_steps", default=100, type=int, help="number of diffusion steps")

    #regularization related
    parser.add_argument('--use_opacity', default=True, type=options.str2bool, help='whether to use opacity')
    parser.add_argument('--w_opacity', default=10000.0, type=float, help='weight for opacity loss')
    parser.add_argument('--w_eikonal', default=0.01, type=float, help='weight for eikonal loss')
    parser.add_argument("--exp_name", type=str, default="zombie", help="name of the experiment")
    parser.add_argument('--epochs',default=2,type = int, help='number of epochs')
    parser.add_argument('--coarse_epochs',default=40,type = int, help='number of epochs for coarse training')
    parser.add_argument('--fine_epochs',default=20,type = int, help='number of epochs for fine training')
    parser.add_argument('--lr',default=5e-3,type = float, help='learning rate')
    parser.add_argument('--i_val',default=100,type = int, help='log image after i_val iterations')
    parser.add_argument('--i_save',default=1000,type = int, help='save model after i_save iterations')
    parser.add_argument('--i_mesh',default=1000,type = int, help='save mesh after i_mesh iterations')

    # augmentations
    parser.add_argument('--augment_bkg',default=True, type = options.str2bool, help='whether to perform random background augmentation')
    parser.add_argument('--augment_cam',default=True, type = options.str2bool, help='whether to perform randomized camera augmentation')
    parser.add_argument('--augment_text',default=True, type = options.str2bool, help='whether to perform view-dependent text augmentation')

    #m-bbox specific
    parser.add_argument('--coarse_head', default = 0.2, type=float, help='how much (ratio) for head box rendering in coarse training')
    parser.add_argument('--fine_head', default = 0.5, type=float, help='how much  (ratio) for head box rendering in fine training')
    opt = parser.parse_args()
    if opt.render_h is None:
        opt.render_size = None
    else:
        opt.render_size = (opt.render_h, opt.render_w)

    options.print_opt(opt)

    trainer = Trainer(opt)
    trainer.train()
