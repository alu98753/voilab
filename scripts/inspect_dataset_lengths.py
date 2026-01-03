
import hydra
import torch
import numpy as np
import os
import dill
from diffusion_policy.dataset.umi_dataset import UmiDataset
from omegaconf import OmegaConf

def inspect_episode(dataset, idx, label):
    episode_len = dataset.replay_buffer.episode_lengths[idx]
    keys_list = list(dataset.replay_buffer.data.keys())
    
    start = dataset.replay_buffer.episode_ends[idx-1] if idx > 0 else 0
    end = dataset.replay_buffer.episode_ends[idx]
    
    print(f"{label} (Idx {idx}): Length = {episode_len}")
    
    if 'robot0_eef_pos' in keys_list:
        pos = dataset.replay_buffer.data['robot0_eef_pos'][start:end]
        grip = dataset.replay_buffer.data['robot0_gripper_width'][start:end]
        
        path_length = np.sum(np.linalg.norm(pos[1:] - pos[:-1], axis=1))
        grip_min, grip_max = np.min(grip), np.max(grip)
        
        print(f"      Path: {path_length:.4f}m, Grip: {grip_min:.4f} -> {grip_max:.4f}")

def main():
    checkpoint = "/mnt/zi/00_course/voilab/data/outputs/2026.01.03/19.34.31_train_diffusion_unet_timm_vit_finetune_umi/checkpoints/latest.ckpt"
    dataset_path = "/mnt/zi/00_course/voilab/AsiaDragon_All_285/simulation_dataset.zarr.zip"
    
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.task.dataset.dataset_path = dataset_path
    
    print(f"Loading dataset from {dataset_path}...")
    dataset_cfg = OmegaConf.to_container(cfg.task.dataset, resolve=True)
    dataset = hydra.utils.instantiate(dataset_cfg)
    
    print("\n--- Validation Episode Lengths ---")
    val_indices = np.where(dataset.val_mask)[0]
    for i, idx in enumerate(val_indices[:5]):
        inspect_episode(dataset, idx, f"Val Ep {i}")
        
    print("\n--- Training Episode Lengths (Sample) ---")
    train_indices = np.where(~dataset.val_mask)[0]
    for i, idx in enumerate(train_indices[:5]):
        inspect_episode(dataset, idx, f"Train Ep {i}")

if __name__ == "__main__":
    main()
