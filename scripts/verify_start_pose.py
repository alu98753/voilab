
import zarr
import numpy as np
import os
import torch

def inspect_dataset_start_pose(dataset_path):
    print(f"Loading dataset from: {dataset_path}")
    store = zarr.ZipStore(dataset_path, mode='r')
    root = zarr.group(store)
    
    data = root['data']
    meta = root['meta']
    episode_ends = meta['episode_ends'][:]
    
    print(f"Total episodes: {len(episode_ends)}")
    
    # Check first few episodes
    for i in range(min(5, len(episode_ends))):
        start_idx = 0 if i == 0 else episode_ends[i-1]
        
        # Check if robot0_demo_start_pose exists
        if 'robot0_demo_start_pose' in data:
            pose = data['robot0_demo_start_pose'][start_idx]
            print(f"Episode {i} Start Pose (Joints): {pose}")
            
            # Also check EE pos if available to compare with hardcoded [4.99...]
            if 'robot0_eef_pos' in data:
                ee_pos = data['robot0_eef_pos'][start_idx]
                print(f"Episode {i} Start EE Pos: {ee_pos}")
                
                hardcoded_target = np.array([4.99, 2.52, 1.09])
                dist = np.linalg.norm(ee_pos - hardcoded_target)
                print(f"  Distance to Hardcoded Target [4.99, 2.52, 1.09]: {dist:.4f}")
        else:
            print("robot0_demo_start_pose not found in dataset!")

if __name__ == "__main__":
    inspect_dataset_start_pose('./AsiaDragon_All_285/simulation_dataset.zarr.zip')
