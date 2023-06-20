# -*- coding : utf-8 -*-
# @FileName  : diffusion.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Oct 20, 2022
# @Github    : https://github.com/songrise
# @Description: Dream Fusion loss, implementation from https://github.com/ashawkey/stable-dreamfusion



from prompt_toolkit import prompt
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



class StableDiffusion(nn.Module):
    """
    Warper class for the SDS loss based on the stable diffusion model.
    """
    def __init__(self, device, version):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '') # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            print(f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')
        
        self.sd_version = version
        self.device = device
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.use_depth = False
        if self.sd_version == '1.5':
            self.model_key = 'runwayml/stable-diffusion-v1-5'
        elif self.sd_version == '2.0':
            self.use_depth = True
            self.model_key = 'stabilityai/stable-diffusion-2-depth'
        print(f'[INFO] loading stable diffusion {self.sd_version} ...')
                
        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained(self.model_key, subfolder="vae",use_auth_token=self.token).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_key, subfolder = "tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_key, subfolder = "text_encoder").to(self.device)

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(self.model_key, subfolder="unet").to(self.device)

        # 4. Create a scheduler for inference
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)  
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        # self.scheduler = DDIMScheduler()
        # self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')


    def get_text_embeds(self, prompt:list):
        if not isinstance(prompt, list):
            prompt = [prompt]
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer([''] * len(prompt), padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    
    def mannual_backward(self, text_embeddings, pred_rgb:torch.Tensor, guidance_scale=100, pred_depth:torch.Tensor = None) -> None:
        """
        backward the SDS loss the the pred_rgb.
        Input:
            pred_rgb: Tensor, [1, 3, H, W] assume requires grad

        return:
            grad_map: [1, 3, H, W], in the same dimension.
        """ 
        h, w = pred_rgb.shape[-2:]

        # zero pad to 512x512
        pred_rgb_512 = torchvision.transforms.functional.pad(pred_rgb, ((512-w)//2, ), fill=0, padding_mode='constant')
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        # debug_utils.dump_tensor(pred_rgb_512, 'pred_rgb_512.pkl')
        if self.use_depth and pred_depth is not None:
            pred_depth = F.interpolate(pred_depth, size=(64, 64), mode='bicubic',
                                align_corners=False)
            pred_depth = 2.0 * (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min()) - 1.0
            pred_depth = torch.cat([pred_depth] * 2)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()
        latents = self.encode_imgs(pred_rgb_512)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)        
            if self.use_depth and pred_depth is not None:
                latent_model_input = torch.cat([latent_model_input, pred_depth], dim=1)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        grad = grad.clamp(-1, 1)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)


   

    def calc_grad(self, text_embeddings, pred_rgb:torch.Tensor, guidance_scale=100) -> torch.Tensor:
        """
        calculate the gradient of the predicted rgb
        Input:
            pred_rgb: Tensor, [1, 3, H, W] assume requires grad

        return:
            grad_map: [1, 3, H, W], in the same dimension.
        """
           # interp to 512x512 to be fed into vae.

        # _t = time.time()
        h, w = pred_rgb.shape[-2:]
 

        # zero pad to 512x512
        pred_rgb_512 = torchvision.transforms.functional.pad(pred_rgb, ((512-w)//2, ), fill=0, padding_mode='constant')
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()
        latents = self.encode_imgs(pred_rgb_512)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        grad = grad.clamp(-1, 1)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        pred_rgb_grad = pred_rgb.grad.detach().clone()
        return pred_rgb_grad




    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100):
        """
        Use sd to perform one step update of the model.
        """
        
        # interp to 512x512 to be fed into vae.

        # _t = time.time()
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        # t = torch.randint(500,501 [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()
        latents = self.encode_imgs(pred_rgb_512)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        grad = grad.clamp(-1, 1)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return 0 # dummy loss value

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        # with torch.autocast('cuda'):
        with torch.cuda.amp.autocast():
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
            # imgs = self.vae.decode(latents)

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':
    import os
    import numpy as np

    def fix_randomness(seed=42):

        # random.seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # torch.manual_seed(seed)
        # np.random.seed(seed)

        # https: // www.zhihu.com/question/542479848/answer/2567626957
        os.environ['PYTHONHASHSEED'] = str(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        np.random.seed(seed)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    import argparse
    import matplotlib.pyplot as plt
    import PIL.Image as Image

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default = "a photo of a cute corgi")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--steps', type=int, default=100)
    opt = parser.parse_args()

    device = torch.device('cuda')

    sd = StableDiffusion(device,version="1.5")
    # prompt_ = "front view of the face of Captain Marvel, photorealistic style."
    # prompt_ = "front view of the body of the Hulk wearing blue jeans, photorealistic style."
    prompt_ = "front view of the body of st, photorealistic style."
    imgs = []
    # fix_randomness(42)

    for _ in range(4):
        img = sd.prompt_to_img(prompt_, 512, 512, opt.steps)
        img = img / 255.
        imgs.append(torch.from_numpy(img))
        print("done one")
    imgs = torch.cat(imgs, dim=0)
    # save image as a grid
    imgs = imgs.permute(0, 3, 1, 2)
    img_grid = torchvision.utils.make_grid(imgs, nrow = 5, padding = 10)
    torchvision.utils.save_image(img_grid, 'img_grid.png')
    print('Image saved as img_grid.png')
