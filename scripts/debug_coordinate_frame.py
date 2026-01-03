
import numpy as np
import zarr
from zarr.storage import ZipStore

dataset_path = "/mnt/zi/00_course/voilab/AsiaDragon_All_285/simulation_dataset_clean.zarr.zip"
store = ZipStore(dataset_path, mode='r')
root = zarr.group(store)

episode_ends = root['meta/episode_ends'][:]
print(f"Number of episodes: {len(episode_ends)}")

# First episode
end = episode_ends[0]
print(f"First episode length: {end}")

eef_pos = root['data/robot0_eef_pos'][0:end]
print(f"First 5 EEF Poses:\n{eef_pos[:5]}")
print(f"Mean EEF Position: {np.mean(eef_pos, axis=0)}")
print(f"Min  EEF Position: {np.min(eef_pos, axis=0)}")
print(f"Max  EEF Position: {np.max(eef_pos, axis=0)}")

store.close()
