"""
評估 Diffusion Policy 模型在 Isaac Sim 環境中的表現（僅使用 Validation Episodes）

此腳本會：
1. 加載數據集並識別 validation episodes（根據訓練時的 val_ratio 和 seed）
2. 只對這些 validation episodes 進行評估
3. 在 Isaac Sim 環境中運行策略
4. 使用 registry 中的成功判斷邏輯評估每個 episode
5. 計算並輸出成功率等指標

用法:
uv run voilab eval-model \
    --checkpoint data/outputs/2025.12.31/08.01.31_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt \
    --output_dir data/eval_output \
    --task kitchen \
    --dataset_path ./AsiaDragon_1127_100/simulation_dataset.zarr.zip \
    --n_episodes 1 --headless 

或者直接使用:
python scripts/eval_kitchen.py \
    --checkpoint data/outputs/2025.12.31/08.01.31_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt \
    --output_dir data/eval_output \
    --task kitchen \
    --dataset_path ./AsiaDragon_1127_100/simulation_dataset.zarr.zip \
    --n_episodes 1 --headless 
"""

import sys
import os
import pathlib

# Detect project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Force load from source to ensure edits to packages are reflected
sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages/diffusion_policy/src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages/umi/src"))
print(f"[Eval] Project Root: {PROJECT_ROOT}")
print(f"[Eval] sys.path[0]: {sys.path[0]}")

import click
import numpy as np # RESTORED: Isaac Sim might need numpy pre-loaded

# Initialize Isaac Sim early to avoid segfaults
# CRITICAL: This must happen before any torch imports!
# Filter sys.argv to prevent SimulationApp from choking on custom flags
original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
if '--headless' in original_argv:
    sys.argv.append('--headless')

# Using config from generate_data.py to ensure stability
simulation_app_config = {
    "headless": '--headless' in original_argv,
    "width": 1280,
    "height": 720,
    "enable_streaming": False,
    "extensions": ["isaacsim.robot_motion.motion_generation"] 
}
print(f"[Eval] Initializing SimulationApp with config: {simulation_app_config}")
from isaacsim import SimulationApp
simulation_app = SimulationApp(simulation_app_config)
# Run one update to settle the engine
simulation_app.update()
print("[Eval] SimulationApp initialized successfully.")

# Restore sys.argv for Click/Argparse
sys.argv = original_argv




# Already detected above

# Fix HF Cache issue (Disk full)
os.environ['HF_HOME'] = os.path.join(PROJECT_ROOT, 'data/.cache/huggingface')
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
print(f"[Eval] Set HF_HOME to: {os.environ['HF_HOME']}")

import dill
import hydra
from omegaconf import OmegaConf
import inspect
import json


from diffusion_policy.workspace.base_workspace import BaseWorkspace
# Already set above

import diffusion_policy

# import diffusion_policy # MOVED INSIDE MAIN
print("[Eval] All imports completed (delayed diffusion_policy load).", flush=True)

@click.command()
@click.option('-c', '--checkpoint', required=True, help='Path to checkpoint file')
@click.option('-o', '--output_dir', required=True, help='Output directory for evaluation results')
@click.option('-d', '--device', default='cuda:0', help='Device to run on (cuda:0, cpu, etc.)')
@click.option('--task', type=click.Choice(['kitchen', 'dining-room', 'living-room']), 
              required=True, help='Task name')
@click.option('--dataset_path', default=None, help='Path to dataset zarr zip file (optional, for reference)')
@click.option('--n_episodes', default=10, type=int, help='Number of evaluation episodes')
@click.option('--headless', is_flag=True, help='Run in headless mode (no GUI)')
@click.option('--random_poses', is_flag=True, help='Use random poses instead of recorded poses from JSON')
@click.option('--config-name', default='train_diffusion_unet_timm_umi_workspace', 
              help='Config name to use')
@click.option('--config-path', default='packages/diffusion_policy/src/diffusion_policy/config',
              help='Path to config directory')
@click.option('--replay_gt', is_flag=True, help='Replay Ground Truth from dataset instead of running policy')
def main(checkpoint, output_dir, device, task, dataset_path, n_episodes, headless, random_poses, config_name, config_path, replay_gt):
    """
    評估訓練好的 Diffusion Policy 模型（僅使用 Validation Episodes）
    
    此腳本會：
    1. 加載數據集並識別 validation episodes
    2. 加載訓練好的模型 checkpoint
    3. 在 Isaac Sim 環境中運行評估（僅對 validation episodes）
    4. 使用 registry 中的成功判斷邏輯評估每個 episode
    5. 計算並輸出成功率等指標
    """
    import torch
    import tqdm
    # Prevent OpenMP conflict between Torch and Isaac Sim
    torch.set_num_threads(1)
    
    print(f"[Eval] Loaded diffusion_policy from: {diffusion_policy.__file__}")

    # Refine output_dir based on checkpoint path
    # Example checkpoint: data/outputs/2025.12.31/08.01.31_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt
    # We want to extract: 2025.12.31/08.01.31_train_diffusion_unet_timm_umi/latest
    try:
        ckpt_path = pathlib.Path(checkpoint).resolve()
        parts = ckpt_path.parts
        # Search for 'outputs' in the path to find the date and run name
        if 'outputs' in parts:
            outputs_idx = parts.index('outputs')
            if len(parts) > outputs_idx + 2:
                date_str = parts[outputs_idx + 1]
                run_str = parts[outputs_idx + 2]
                ckpt_name = ckpt_path.stem # 'latest'
                
                # Construct refined path: root / date / run / ckpt_name
                output_dir = os.path.join(output_dir, date_str, run_str, ckpt_name)
                print(f"[Eval] Refined output directory: {output_dir}")
    except Exception as e:
        print(f"[Eval] WARNING: Could not refine output directory path: {e}")

    if os.path.exists(output_dir):
        # click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        pass
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"[Eval] ========== Evaluation Configuration ==========")
    print(f"[Eval] Checkpoint: {checkpoint}")
    print(f"[Eval] Task: {task}")
    print(f"[Eval] Output directory: {output_dir}")
    print(f"[Eval] Device: {device}")
    print(f"[Eval] Number of episodes: {n_episodes}")
    print(f"[Eval] Headless mode: {headless}")
    if dataset_path:
        print(f"[Eval] Dataset path: {dataset_path}")
    print(f"[Eval] Evaluation mode: {'Random Poses' if random_poses else 'Recorded Poses'}")
    print(f"[Eval] Replay GT: {replay_gt}")
    print(f"[Eval] ==============================================")
    
    # Load checkpoint
    print(f"\n[Eval] Loading checkpoint from: {checkpoint}")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg_payload = cfg # Explicit alias for policy instantiation safety
    
    # Override dataset path if provided
    if dataset_path:
        cfg.task.dataset_path = dataset_path
        # Also override internal dataset config if it exists
        if hasattr(cfg.task, 'dataset') and hasattr(cfg.task.dataset, 'dataset_path'):
            cfg.task.dataset.dataset_path = dataset_path
        print(f"[Eval] Using dataset: {dataset_path}")
    elif not hasattr(cfg.task, 'dataset_path') or cfg.task.dataset_path is None:
        raise ValueError("Dataset path must be provided either via --dataset_path or in config")
    
    # Override task name
    cfg.task.name = task
    
    # Load dataset to get validation episodes
    print(f"\n[Eval] Loading dataset to identify validation episodes...")
    from diffusion_policy.dataset.umi_dataset import UmiDataset
    train_dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = train_dataset.get_validation_dataset()
    
    # Get validation episode indices from train_dataset (val_mask is True for validation episodes)
    val_mask = train_dataset.val_mask if hasattr(train_dataset, 'val_mask') else None
    if val_mask is not None:
        val_episode_indices = np.where(val_mask)[0].tolist()
        zarr_episode_indices = val_episode_indices # Keep original Zarr indices
        n_val_episodes = len(val_episode_indices)
        total_episodes = len(val_mask)
        print(f"[Eval] Found {n_val_episodes} validation episodes out of {total_episodes} total episodes")
        
        # Override n_episodes to match validation episodes count
        if n_episodes is None or n_episodes > n_val_episodes:
            original_n_episodes = n_episodes
            n_episodes = n_val_episodes
            if original_n_episodes is not None:
                print(f"[Eval] Requested {original_n_episodes} episodes, but only {n_val_episodes} validation episodes available.")
            print(f"[Eval] Limiting evaluation to {n_episodes} validation episodes")
    else:
        print(f"[Eval] WARNING: Could not get validation mask from dataset. Using all episodes.")
        val_episode_indices = None
        zarr_episode_indices = None
    
    # Calculate object_poses.json path and map original indices
    object_poses_path = None
    if not random_poses and dataset_path:
        # Expected: dataset_path is ./AsiaDragon_1127_100/simulation_dataset.zarr.zip
        # JSON: ./AsiaDragon_1127_100/demos/mapping/object_poses.json
        dataset_dir = os.path.dirname(dataset_path)
        
        # 1. Look for .previous_progress.json to map sequential zarr indices to original raw indices
        progress_path = os.path.join(dataset_dir, '.previous_progress.json')
        completed_episodes = None
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r') as f:
                    progress_data = json.load(f)
                    completed_episodes = progress_data.get('completed_episodes', [])
                    print(f"[Eval] Loaded mapping from {progress_path}, {len(completed_episodes)} episodes found.")
            except Exception as e:
                print(f"[Eval] WARNING: Failed to load {progress_path}: {e}")
        
        # 2. Map val_episode_indices to original indices
        if completed_episodes is not None and val_episode_indices is not None:
            # val_episode_indices are sequential indices into the Zarr's 'successful' subset.
            # completed_episodes[idx] gives the original raw index for that Zarr entry.
            mapped_indices = []
            for idx in val_episode_indices:
                if idx < len(completed_episodes):
                    mapped_indices.append(completed_episodes[idx])
                else:
                    print(f"[Eval] WARNING: Index {idx} out of range for completed_episodes (len={len(completed_episodes)})")
                    mapped_indices.append(idx) # Fallback to sequential
            
            print(f"[Eval] Mapping Zarr indices to original raw indices:")
            print(f"       Zarr: {val_episode_indices[:10]}...")
            print(f"       Raw:  {mapped_indices[:10]}...")
            simulation_episode_indices = mapped_indices
        else:
            simulation_episode_indices = val_episode_indices

        # 3. Locate object_poses.json
        potential_path = os.path.join(dataset_dir, 'demos', 'mapping', 'object_poses.json')
        if os.path.exists(potential_path):
            object_poses_path = potential_path
            print(f"[Eval] Found object poses at: {object_poses_path}")
        else:
            print(f"[Eval] WARNING: object_poses.json not found at {potential_path}. Falling back to random mode.")
            random_poses = True
    
    
    # Initialize policy directly using Payload Config (Proven to work)
    print(f"[Debug] Instantiating policy directly from payload config: {cfg_payload.policy._target_}", flush=True)
    try:
        model = hydra.utils.instantiate(cfg_payload.policy)
    except Exception as e:
        print(f"[Fatal] Failed to instantiate policy from payload: {e}", flush=True)
        # Fallback to hydration config if payload fails (unlikely)
        print(f"[Debug] Fallback to hydra config...", flush=True)
        model = hydra.utils.instantiate(cfg.policy)
    
    print(f"[Debug] Loading policy weights...", flush=True)
    if 'model' in payload['state_dicts']:
        model.load_state_dict(payload['state_dicts']['model'])
        print("[Eval] Loaded model weights from checkpoint.", flush=True)
    else:
        raise ValueError("Checkpoint payload missing 'model' state dict!")

    workspace = None # No workspace object
    policy = model
    if cfg.training.use_ema and 'ema_model' in payload['state_dicts']:
        print("[Eval] Loading EMA model for evaluation...", flush=True)
        import copy
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(payload['state_dicts']['ema_model'])
        policy = ema_model
    
    # Mock workspace for compatibility if needed, but we used policy from workspace later
    # We need to ensure we don't try to access workspace.model later
    
    print(f"[Debug] cfg.task.env_runner target after workspace init: {cfg.task.env_runner['_target_']}", flush=True)

    # Override env_runner to use IsaacSimRunner for evaluation
    # The training config might use RealPushTImageRunner, but we need IsaacSimRunner for eval
    if not hasattr(cfg.task, 'env_runner') or \
       cfg.task.env_runner._target_ != 'diffusion_policy.env_runner.isaac_sim_runner.IsaacSimRunner':
        print(f"[Eval] Overriding env_runner to use IsaacSimRunner")
        # Try to use isaac_sim task config if available
        try:
            from hydra import compose, initialize_config_dir
            # Fix config_path to be relative to project root
            if not os.path.isabs(config_path):
                # If relative, make it absolute based on project root
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                config_path_abs = os.path.join(project_root, config_path)
            else:
                config_path_abs = config_path
            
            if os.path.exists(config_path_abs):
                print(f"[Debug] Loading config from: {config_path_abs}", flush=True)
                with initialize_config_dir(config_dir=config_path_abs, version_base=None):
                    isaac_sim_cfg = compose(config_name="task/isaac_sim")
                    print(f"[Debug] isaac_sim_cfg content: {isaac_sim_cfg}", flush=True)
                    if hasattr(isaac_sim_cfg, 'task') and hasattr(isaac_sim_cfg.task, 'env_runner'):
                        print(f"[Debug] isaac_sim_cfg.task.env_runner target: {isaac_sim_cfg.task.env_runner['_target_']}", flush=True)
                        cfg.task.env_runner = isaac_sim_cfg.task.env_runner
                        print(f"[Eval] Loaded env_runner from isaac_sim config")
                        print(f"[Debug] cfg.task.env_runner target after override: {cfg.task.env_runner['_target_']}", flush=True)
            else:
                raise FileNotFoundError(f"Config path not found: {config_path_abs}")
        except Exception as e:
            print(f"[Eval] WARNING: Could not load isaac_sim config: {e}")
            print(f"[Eval] Creating default IsaacSimRunner config")
            # Fallback: create default IsaacSimRunner config
            cfg.task.env_runner = OmegaConf.create({
                '_target_': 'diffusion_policy.env_runner.isaac_sim_runner.IsaacSimRunner',
                'urdf_path': '/workspace/voilab/assets/franka_panda/franka_panda_umi-isaacsim.urdf',
                'usd_path': '/workspace/voilab/assets/ED305_scene/ED305.usd',
                'headless': headless,
                'n_episodes': n_episodes,
                'max_steps_per_episode': 200,
                'save_video': True,
                'save_observation_data': False,
                'n_obs_steps': 2,
                'n_action_steps': 1,
                'fps': 30,
                'crf': 22
            })
    

    # Get task registry for success criteria
    print(f"\n[Eval] Loading task registry for: {task}")
    import registry 
    registry_class = registry.get_task_registry(task)
    if not registry_class.validate_environment():
        print(f"[Eval] WARNING: Registry validation failed for task {task}")
    
    is_episode_completed = registry_class.is_episode_completed
    print(f"[Eval] Success criteria loaded from: {registry_class.__name__}")
    
    # Get env runner config
    env_runner_cfg = cfg.task.env_runner
    
    # Get the actual class to inspect its signature
    runner_class = hydra.utils.get_class(env_runner_cfg['_target_'])
    
    # Get the __init__ signature to see what parameters it accepts
    sig = inspect.signature(runner_class.__init__)
    runner_kwargs = {'output_dir': output_dir}
    
    # Add all parameters from config that the runner accepts
    env_runner_dict = OmegaConf.to_container(env_runner_cfg, resolve=True)
    for key, value in env_runner_dict.items():
        if key != '_target_' and key in sig.parameters:
            runner_kwargs[key] = value
    
    # Override n_episodes if runner accepts it
    if n_episodes is not None and 'n_episodes' in sig.parameters:
        runner_kwargs['n_episodes'] = n_episodes
    
    # Pass object poses and episode indices configuration
    if 'use_recorded_poses' in sig.parameters:
        runner_kwargs['use_recorded_poses'] = not random_poses
    
    if object_poses_path and 'object_poses_path' in sig.parameters:
        runner_kwargs['object_poses_path'] = object_poses_path

    if simulation_episode_indices is not None and 'episode_indices' in sig.parameters:
        # Limit to requested number of episodes
        runner_kwargs['episode_indices'] = simulation_episode_indices[:n_episodes] if n_episodes else simulation_episode_indices
        if zarr_episode_indices is not None and 'dataset_episode_indices' in sig.parameters:
             runner_kwargs['dataset_episode_indices'] = zarr_episode_indices[:n_episodes] if n_episodes else zarr_episode_indices
        
        print(f"[Eval] Using simulation episode indices: {runner_kwargs['episode_indices']}")
        if 'dataset_episode_indices' in runner_kwargs:
            print(f"[Eval] Using dataset episode indices: {runner_kwargs['dataset_episode_indices']}")
    elif simulation_episode_indices is not None and 'val_episode_indices' in sig.parameters:
        runner_kwargs['val_episode_indices'] = simulation_episode_indices[:n_episodes] if n_episodes else simulation_episode_indices
        print(f"[Eval] Using simulation episode indices: {runner_kwargs['val_episode_indices']}")
    
    # Override headless if runner accepts it
    if headless and 'headless' in sig.parameters:
        runner_kwargs['headless'] = True
    
    # Ensure shape_meta is passed
    if 'shape_meta' in sig.parameters:
        runner_kwargs['shape_meta'] = cfg.shape_meta
    
    # Pass validation dataset for MSE calculation
    if val_dataset is not None and 'validation_dataset' in sig.parameters:
        runner_kwargs['validation_dataset'] = val_dataset
        
    if replay_gt and 'replay_gt' in sig.parameters:
        runner_kwargs['replay_gt'] = True
        # Set a very high limit; the runner will break exactly when GT actions end
        runner_kwargs['max_steps_per_episode'] = 10000 
        print(f"[Eval] Enabling GT Replay Mode (max_steps extended to 10000)")
    
    # Instantiate env runner
    print(f"\n[Eval] Creating environment runner: {env_runner_cfg['_target_']}")
    env_runner = hydra.utils.instantiate(
        env_runner_cfg,
        **runner_kwargs
    )
    
    
    # Inject check function into runner
    # The runner must handle calling this function
    env_runner.check_success_fn = is_episode_completed
    
    # Inject Registry Config for Pose Setup
    if hasattr(env_runner, 'set_registry_config'):
        env_runner.set_registry_config(registry_class.get_config())
        
    print(f"[Eval] Injected success check function and config from registry for task: {task}")
    # -------------------------------------

    # HACK: Force Isaac Sim initialization before PyTorch grabs CUDA context
    # Moved AFTER injection so _setup_simulation can use the config
    if hasattr(env_runner, '_setup_simulation'):
        print("[Eval] Forcing early Isaac Sim setup...")
        env_runner._setup_simulation()

    # Policy is already loaded above
    
    device = torch.device(device)
    print(f"[Debug] Moving policy to device: {device}", flush=True)
    policy.to(device)
    print(f"[Debug] Policy moved to device.", flush=True)
    policy.eval()
    print(f"[Debug] Policy set to eval mode.", flush=True)
    
    print(f"[Eval] Policy loaded and set to eval mode")
    print(f"[Eval] Policy device: {device}")
    

    
    # Run evaluation
    print(f"\n[Eval] ========== Starting Evaluation ==========")
    runner_log = env_runner.run(policy)
    
    # Extract success rate from runner log
    success_rate = runner_log.get('success_rate', 0.0)
    avg_episode_length = runner_log.get('avg_episode_length', 0.0)
    episode_stats = runner_log.get('episode_stats', [])
    
    print(f"\n[Eval] ========== Evaluation Results (Validation Episodes) ==========")
    print(f"[Eval] Total episodes evaluated: {len(episode_stats)}")
    if val_episode_indices is not None:
        print(f"[Eval] Validation episode indices used: {val_episode_indices[:len(episode_stats)]}")
    print(f"[Eval] Success rate: {success_rate:.2%}")
    print(f"[Eval] Average episode length: {avg_episode_length:.2f} steps")
    
    # Count successes
    successes = sum(1 for ep in episode_stats if ep.get('success', False))
    print(f"[Eval] Successful episodes: {successes}/{len(episode_stats)}")
    
    # Print per-episode results
    if episode_stats:
        print(f"\n[Eval] Per-episode results:")
        for i, ep in enumerate(episode_stats):
            status = "✓" if ep.get('success', False) else "✗"
            sim_idx = ep.get('episode_idx', i)
            ds_idx = ep.get('dataset_episode_idx', sim_idx)
            print(f"[Eval]   Episode {i+1} (sim idx {sim_idx}, ds idx {ds_idx}): {status} (length: {ep.get('episode_length', 0)} steps)")
    
    # Dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, (np.ndarray, np.generic)):
            json_log[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
        elif isinstance(value, (list, dict, str, int, float, bool)):
            json_log[key] = value
        else:
            json_log[key] = str(value)
    
    # Add evaluation summary
    json_log['evaluation_summary'] = {
        'total_episodes': len(episode_stats),
        'successful_episodes': successes,
        'success_rate': float(success_rate),
        'avg_episode_length': float(avg_episode_length),
        'task': task,
        'checkpoint': checkpoint,
        'device': str(device),
        'evaluation_mode': 'validation_episodes_only',
        'validation_episode_indices': val_episode_indices[:len(episode_stats)] if val_episode_indices else None
    }
    
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True, default=str)
    print(f"\n[Eval] Results saved to: {out_path}")
    import subprocess
    print(f"[Debug] Listing {output_dir} content:")
    subprocess.run(["ls", "-l", output_dir])
    print(f"[Eval] ==========================================")
    
    return json_log


if __name__ == '__main__':
    main()

