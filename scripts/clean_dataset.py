import zarr
import argparse
import numpy as np
import json
import os
import shutil
from zarr.storage import ZipStore
from numcodecs import Blosc
from tqdm import tqdm

def clean_dataset(input_path, output_path, blacklist_path, mapping_path=None, raw_indices_map_path=None):
    print(f"[Clean] Input: {input_path}")
    print(f"[Clean] Output: {output_path}")
    print(f"[Clean] Blacklist: {blacklist_path}")

    # 1. Load Blacklist
    try:
        with open(blacklist_path, 'r') as f:
            blacklist_data = json.load(f)
            if isinstance(blacklist_data, list):
                raw_blacklist = set(blacklist_data)
            else:
                raw_blacklist = set(blacklist_data.get("episodes", []))
    except Exception as e:
        print(f"[Error] Failed to load blacklist: {e}")
        return

    # 1.5 Handle Raw Index Mapping
    blacklist_indices = set()
    if raw_indices_map_path:
        print(f"[Clean] Loading Raw Mapping from: {raw_indices_map_path}")
        try:
            with open(raw_indices_map_path, 'r') as f:
                progress_data = json.load(f)
                completed_episodes = progress_data.get("completed_episodes", [])
                
                # Create map: Raw ID -> Zarr Index
                # Zarr stores episodes in the order of completed_episodes
                raw_to_zarr = {raw_id: i for i, raw_id in enumerate(completed_episodes)}
                
                print(f"[Clean] Mapping loaded. Converting blacklist raw indices...")
                for raw_id in raw_blacklist:
                    if raw_id in raw_to_zarr:
                        zarr_idx = raw_to_zarr[raw_id]
                        blacklist_indices.add(zarr_idx)
                        print(f"  - Raw {raw_id} -> Zarr Index {zarr_idx}")
                    else:
                        print(f"  - Raw {raw_id} NOT FOUND in dataset (already failed/excluded). Skipping.")
        except Exception as e:
             print(f"[Error] Failed to load raw index map: {e}")
             return
    else:
        # Default: user provided Zarr indices directly
        blacklist_indices = raw_blacklist
        # Ensure indices are valid integers
        blacklist_indices = {int(x) for x in blacklist_indices}

    print(f"[Clean] Final list of Zarr indices to remove ({len(blacklist_indices)}): {sorted(list(blacklist_indices))}")

    # 2. Open Input Zarr
    try:
        input_store = ZipStore(input_path, mode='r')
        input_root = zarr.group(input_store)
        input_data = input_root['data']
        input_meta = input_root['meta']
        episode_ends = input_meta['episode_ends'][:]
        total_episodes = len(episode_ends)
    except Exception as e:
        print(f"[Error] Failed to open input Zarr: {e}")
        return

    print(f"[Clean] Total input episodes: {total_episodes}")

    # 3. Calculate Valid Episodes and New Structure
    valid_indices = []
    episode_mappings = {} # new_idx -> old_idx
    
    current_end = 0
    new_episode_ends = []
    total_frames_new = 0

    # Pre-calculate size
    for old_idx in range(total_episodes):
        if old_idx in blacklist_indices:
            continue
        
        valid_indices.append(old_idx)
        
        # Calculate length of this episode
        start = 0 if old_idx == 0 else episode_ends[old_idx-1]
        end = episode_ends[old_idx]
        length = end - start
        
        total_frames_new += length
        new_episode_ends.append(total_frames_new)
        
        episode_mappings[len(new_episode_ends)-1] = old_idx

    print(f"[Clean] New dataset will have {len(valid_indices)} episodes and {total_frames_new} frames.")
    
    if len(valid_indices) == 0:
        print("[Error] No valid episodes remaining! Aborting.")
        return

    # 4. Create Output Zarr
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    output_store = ZipStore(output_path, mode='w')
    output_root = zarr.group(output_store)
    output_data = output_root.create_group("data")
    output_meta = output_root.create_group("meta")

    # Save metadata (episode_ends)
    output_meta.create_dataset("episode_ends", data=np.array(new_episode_ends, dtype=np.int64))

    # Save mapping if requested
    if mapping_path:
        with open(mapping_path, 'w') as f:
            json.dump({
                "new_to_old_index": episode_mappings,
                "removed_zarr_indices": sorted(list(blacklist_indices)),
                "source_file": input_path
            }, f, indent=2)
        print(f"[Clean] Saved mapping to {mapping_path}")

    # 5. Copy Data Arrays
    # Get all array keys from input data group
    array_keys = list(input_data.keys())
    
    for key in array_keys:
        print(f"[Clean] Processing array: {key}")
        input_arr = input_data[key]
        
        # Prepare output array
        shape = list(input_arr.shape)
        shape[0] = total_frames_new
        chunks = list(input_arr.chunks)
        # Ensure chunks don't exceed shape dims if dynamic
        
        output_arr = output_data.create_dataset(
            key,
            shape=tuple(shape),
            chunks=tuple(chunks),
            dtype=input_arr.dtype,
            compressor=compressor
        )

        # Copy data episode by episode
        current_write_idx = 0
        
        for new_idx, old_idx in tqdm(episode_mappings.items(), desc=f"Copying {key}"):
            # Determine range in old array
            old_start = 0 if old_idx == 0 else episode_ends[old_idx-1]
            old_end = episode_ends[old_idx]
            length = old_end - old_start
            
            # Read data
            chunk_data = input_arr[old_start:old_end]
            
            # Write data
            output_arr[current_write_idx : current_write_idx+length] = chunk_data
            
            current_write_idx += length

    input_store.close()
    output_store.close()
    print(f"[Clean] Successfully created {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Zarr dataset by removing blacklisted episodes.")
    parser.add_argument("--input", required=True, help="Path to input Zarr zip")
    parser.add_argument("--output", required=True, help="Path to output Zarr zip")
    parser.add_argument("--blacklist", required=True, help="Path to blacklist JSON file")
    parser.add_argument("--mapping", help="Path to save index mapping JSON (optional)", default="dataset_cleaning_mapping.json")
    parser.add_argument("--raw_indices_map", help="Path to .previous_progress.json to map Raw index to Zarr index", default=None)
    
    args = parser.parse_args()
    
    clean_dataset(args.input, args.output, args.blacklist, args.mapping, args.raw_indices_map)
