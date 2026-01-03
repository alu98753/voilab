import zarr
import argparse
import json
import numpy as np
from zarr.storage import ZipStore
from tqdm import tqdm

def verify_dataset(original_path, cleaned_path, mapping_path):
    print(f"--- Verifying Cleaned Dataset ---")
    print(f"Original: {original_path}")
    print(f"Cleaned:  {cleaned_path}")
    print(f"Mapping:  {mapping_path}")

    # 1. Load Mapping
    with open(mapping_path, 'r') as f:
        mapping_data = json.load(f)
    
    new_to_old = {int(k): int(v) for k, v in mapping_data["new_to_old_index"].items()}
    removed_indices = set(mapping_data["removed_zarr_indices"])
    
    print(f"Mapping loaded: {len(new_to_old)} episodes kept, {len(removed_indices)} removed.")

    # 2. Open Zarrs
    orig_store = ZipStore(original_path, mode='r')
    clean_store = ZipStore(cleaned_path, mode='r')
    
    orig_root = zarr.group(orig_store)
    clean_root = zarr.group(clean_store)

    orig_ends = orig_root['meta/episode_ends'][:]
    clean_ends = clean_root['meta/episode_ends'][:]

    # 3. Verify Episode Count
    assert len(clean_ends) == len(new_to_old), f"Episode count mismatch! Zarr says {len(clean_ends)}, Mapping says {len(new_to_old)}"
    print("[PASS] Episode count matches mapping.")

    # 4. Verify Data Content (Sampling)
    # Check random 5 episodes + first + last
    indices_to_check = [0, len(clean_ends)-1]
    if len(clean_ends) > 5:
        indices_to_check += np.random.choice(len(clean_ends), 5, replace=False).tolist()
    indices_to_check = sorted(list(set(indices_to_check)))

    arrays_to_check = ['data/camera0_rgb', 'data/robot0_eef_pos']

    print(f"\nChecking sample episodes: {indices_to_check}")

    for new_idx in tqdm(indices_to_check, desc="Verifying Data"):
        old_idx = new_to_old[new_idx]
        
        # Get Clean Range
        c_start = 0 if new_idx == 0 else clean_ends[new_idx-1]
        c_end = clean_ends[new_idx]
        c_len = c_end - c_start

        # Get Old Range
        o_start = 0 if old_idx == 0 else orig_ends[old_idx-1]
        o_end = orig_ends[old_idx]
        o_len = o_end - o_start

        # Length Check
        assert c_len == o_len, f"Length mismatch at new_idx {new_idx} (old {old_idx}): {c_len} vs {o_len}"

        # Data Check
        for arr_name in arrays_to_check:
            clean_data = clean_root[arr_name][c_start:c_end]
            orig_data = orig_root[arr_name][o_start:o_end]
            
            if not np.array_equal(clean_data, orig_data):
                raise ValueError(f"Data Mismatch! Array: {arr_name}, new_idx: {new_idx}, old_idx: {old_idx}")
    
    print("\n[PASS] All data checks passed! The dataset is structurally identical to the selected subset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--cleaned", required=True)
    parser.add_argument("--mapping", required=True)
    args = parser.parse_args()
    
    verify_dataset(args.original, args.cleaned, args.mapping)
