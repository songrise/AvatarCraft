import torch
import numpy as np

sample_rate = 10
# path = "/workspace/songrise/CLIP-Actor/datasets/amass/SFU/SFU/0008/0008_Yoga001_poses.npz"
path = "path/to/amass/*.npz"
with open(path, "rb") as f:
    data = np.load(f)
    poses = data["poses"][:, :63]
    poses = poses[::sample_rate]
    betas = data["betas"][:10]
    poses = np.concatenate([poses, torch.zeros(poses.shape[0], 9)], axis=1)
    poses = poses.reshape(-1, 24, 3)
    # poses[:,0,:] = 0.0
    poses = poses.astype(np.float32)
    with open("data/amass_processed/amass_rope.pkl", "wb") as f:
        np.save(f, poses)

    
print(f"Done with processing")