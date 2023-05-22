# AvatarCraft

## [Website](https://avatar-craft.github.io/)| [Arixv](https://arxiv.org/abs/2303.17606) | [Video](https://www.youtube.com/watch?v=aB4h6_WmW5s) | [Data](https://drive.google.com/drive/folders/1fKosS6JfidXF-XO8ai15Qb18KpKzQQ5q?usp=sharing)

![teaser](asset/teaser.png)

## Abstract
Neural implicit fields are powerful for representing 3D scenes and generating high-quality novel views, but it re mains challenging to use such implicit representations for creating a 3D human avatar with a specific identity and artistic style that can be easily animated. Our proposed method, AvatarCraft, addresses this challenge by using diffusion models to guide the learning of geometry and texture for a neural avatar based on a single text prompt. We care fully design the optimization of neural implicit fields using diffusion models, including a coarse-to-fine multi-bounding box training strategy, shape regularization, and diffusion- based constraints, to produce high-quality geometry and texture. Additionally, we make the human avatar animatable by deforming the neural implicit field with an explicit warping field that maps the target human mesh to a template human mesh, both represented using parametric human models. This simplifies the animation and reshaping of the generated avatar by controlling pose and shape parameters. Extensive experiments on various text descriptions show that AvatarCraft is effective and robust in creating human avatars and rendering novel views, poses, and shapes.

## Setup (WIP)
Use Conda to create a virtual environment and install dependencies:
```
conda create -n avatar python=3.7 -y;
conda activate avatar;
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch;
# For RTX 30 series GPU with CUDA version 11.x, please use:
# conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c fvcore -c iopath -c conda-forge fvcore iopath;
conda install -c bottler nvidiacub;
conda install pytorch3d -c pytorch3d;
conda install -c conda-forge igl;
pip install opencv-python joblib open3d imageio==2.25.0 tensorboardX chumpy lpips scikit-image ipython matplotlib einops trimesh pymcubes;
pip install diffusers transformers;
```
## Data
[**Compulsory**] Register and download the [SMPL](https://smpl.is.tue.mpg.de/) model, put it under `./data` path:
```
data
  |-- smplx
      |-- smpl
            |-- SMPL_NEUTRAL.pkl
```

[**Compulsory**] Download our pretrained [bare SMPL ckpt](https://drive.google.com/file/d/1Exq0EO5WqtxXKJ41o97trqqBDYOOTUh7/view?usp=share_link), put it into `./ckpt` path.


[**Optional**] If you would like to animate the generated avatar, you need sequence of SMPL poses. In our project, we use [AMASS](https://amass.is.tue.mpg.de/) dataset (SMPL+H) to generate the poses. Specifically, we use the SFU subset. We provide a [script](utils/convert_amass.py) to convert the AMASS dataset to our format. You may also use your own pose sequence.

[**Optional**] If you would like to reshape the generated avatar, you need the betas (shape parameter) of SMPL model. We provide some randomly generated betas [here](https://drive.google.com/file/d/1Exq0EO5WqtxXKJ41o97trqqBDYOOTUh7/view?usp=share_link). Put them under `data/smpl_poses`. You may also use your own betas.


## Avatar Creation
use the following command to create an avatar. We test our code on A100-80G, if you encounter OOM error, please reduce the batch size.
```
python stylize.py --weights_path "ckpts/bare_smpl.pth.tar" --tgt_text "Hulk, photorealistic style" --exp_name "hulk" --batch_size 4096
```

After creation, you can render the canonical avatar with following command:
```
python render_canonical.py --weights_path path/to/generated_avatar.pth.tar --exp_name "hulk" --render_h 256 --render_w 256
```

## Avatar Articulation
Once you have genreated the canonical avatar, you can articulate it with the SMPL parameters. 
Use following command to animate the avatar, where `--poseseq_path` is the path to the pose sequence processed by our [script](utils/convert_amass.py).
```
python render_wrap.py --weights_path path/to/generated_avatar.pth.tar --exp_name "hulk" --render_type animate --render_h 256 --render_w 256 --poseseq_path path/to/pose_sequence.pkl
```

Use the following command to reshape the avatar. Specifically, you could interpolate the betas between two avatars. 
```
python render_wrap.py --weights_path path/to/generated_avatar.pth.tar --exp_name "hulk" --render_type interp_pose --render_h 256 --render_w 256 --shape_from_path data/smpl_betas/betas_fat.pkl --shape_to_path data/smpl_betas/betas_skinny.pkl
```


##Citation
If you find our work useful in your research, please consider citing:
```
@article{jiang2023avatarcraft,
  title={AvatarCraft: Transforming Text into Neural Human Avatars with Parameterized Shape and Pose Control},
  author={Jiang, Ruixiang and Wang, Can and Zhang, Jingbo and Chai, Menglei and He, Mingming and Chen, Dongdong and Liao, Jing},
  journal={arXiv preprint arXiv:2303.17606},
  year={2023}
}
```
