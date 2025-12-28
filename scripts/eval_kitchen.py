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
    --checkpoint data/outputs/2025.12.27/02.15.38_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt \
    --output_dir data/eval_output \
    --task kitchen \
    --dataset_path ./AsiaDragon_1127_100/simulation_dataset.zarr.zip \
    --n_episodes 1 --headless 

或者直接使用:
python scripts/eval_kitchen.py \
    --checkpoint data/outputs/2025.12.27/02.15.38_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt \
    --output_dir data/eval_output \
    --task kitchen \
    --dataset_path /path/to/dataset.zarr.zip \
    --n_episodes 10
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import sys
import os
import pathlib
import click
import torch
import numpy as np
import tqdm
import dill
import hydra
from omegaconf import OmegaConf

# Initialize Isaac Sim early to avoid segfaults
if '--headless' in sys.argv:
    print("[Eval] Initializing SimulationApp (headless)...")
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})
elif '--help' not in sys.argv:
    # Do nothing for help
    pass
else:
    print("[Eval] Initializing SimulationApp (windowed)...")
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})

from diffusion_policy.workspace.base_workspace import BaseWorkspace

# 添加 scripts 目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry

# 添加 packages 目錄到路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'packages', 'diffusion_policy', 'src'))

from diffusion_policy.workspace.base_workspace import BaseWorkspace


@click.command()
@click.option('-c', '--checkpoint', required=True, help='Path to checkpoint file')
@click.option('-o', '--output_dir', required=True, help='Output directory for evaluation results')
@click.option('-d', '--device', default='cuda:0', help='Device to run on (cuda:0, cpu, etc.)')
@click.option('--task', type=click.Choice(['kitchen', 'dining-room', 'living-room']), 
              required=True, help='Task name')
@click.option('--dataset_path', default=None, help='Path to dataset zarr zip file (optional, for reference)')
@click.option('--n_episodes', default=10, type=int, help='Number of evaluation episodes')
@click.option('--headless', is_flag=True, help='Run in headless mode (no GUI)')
@click.option('--config-name', default='train_diffusion_unet_timm_umi_workspace', 
              help='Config name to use')
@click.option('--config-path', default='packages/diffusion_policy/src/diffusion_policy/config',
              help='Path to config directory')
def main(checkpoint, output_dir, device, task, dataset_path, n_episodes, headless, config_name, config_path):
    """
    評估訓練好的 Diffusion Policy 模型（僅使用 Validation Episodes）
    
    此腳本會：
    1. 加載數據集並識別 validation episodes
    2. 加載訓練好的模型 checkpoint
    3. 在 Isaac Sim 環境中運行評估（僅對 validation episodes）
    4. 使用 registry 中的成功判斷邏輯評估每個 episode
    5. 計算並輸出成功率等指標
    """
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
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
    print(f"[Eval] Evaluation mode: Validation episodes only")
    print(f"[Eval] ==============================================")
    
    # Load checkpoint
    print(f"\n[Eval] Loading checkpoint from: {checkpoint}")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
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
    print(f"[Debug] Instantiating dataset...")
    train_dataset = hydra.utils.instantiate(cfg.task.dataset)
    print(f"[Debug] Dataset instantiated.")
    val_dataset = train_dataset.get_validation_dataset()
    
    # Get validation episode indices from train_dataset (val_mask is True for validation episodes)
    val_mask = train_dataset.val_mask if hasattr(train_dataset, 'val_mask') else None
    if val_mask is not None:
        val_episode_indices = np.where(val_mask)[0].tolist()
        n_val_episodes = len(val_episode_indices)
        total_episodes = len(val_mask)
        print(f"[Eval] Found {n_val_episodes} validation episodes out of {total_episodes} total episodes")
        print(f"[Eval] Validation episode indices: {val_episode_indices[:10]}{'...' if len(val_episode_indices) > 10 else ''}")
        
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
    
    # Override env_runner to use IsaacSimRunner for evaluation
    # The training config might use RealPushTImageRunner, but we need IsaacSimRunner for eval
    if not hasattr(cfg.task, 'env_runner') or \
       cfg.task.env_runner._target_ != 'diffusion_policy.env_runner.isaac_sim_runner.IsaacSimRunner':
        print(f"[Eval] Overriding env_runner to use IsaacSimRunner")
        # Try to use isaac_sim task config if available
        try:
            from hydra import compose, initialize
            # Fix config_path to be relative to project root
            if not os.path.isabs(config_path):
                # If relative, make it absolute based on project root
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                config_path_abs = os.path.join(project_root, config_path)
            else:
                config_path_abs = config_path
            
            if os.path.exists(config_path_abs):
                with initialize(config_path=config_path_abs, version_base=None):
                    isaac_sim_cfg = compose(config_name="task/isaac_sim")
                    if hasattr(isaac_sim_cfg, 'env_runner'):
                        cfg.task.env_runner = isaac_sim_cfg.env_runner
                        print(f"[Eval] Loaded env_runner from isaac_sim config")
            else:
                raise FileNotFoundError(f"Config path not found: {config_path_abs}")
        except Exception as e:
            print(f"[Eval] WARNING: Could not load isaac_sim config: {e}")
            print(f"[Eval] Creating default IsaacSimRunner config")
            # Fallback: create default IsaacSimRunner config
            cfg.task.env_runner = OmegaConf.create({
                '_target_': 'diffusion_policy.env_runner.isaac_sim_runner.IsaacSimRunner',
                'urdf_path': '/workspace/voilab/assets/franka_panda/franka_panda_umi-isaacsim.urdf',
                'usd_path': None,
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
    
    # Initialize workspace
    print(f"[Debug] Loading workspace class: {cfg._target_}")
    instance = hydra.utils.get_class(cfg._target_)
    print(f"[Debug] Workspace class loaded. Initializing instance...")
    workspace: BaseWorkspace = instance(cfg, output_dir=output_dir)
    print(f"[Debug] Workspace initialized. Loading payload...")
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    print(f"[Debug] Payload loaded.")
    
    # Get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
        print(f"[Eval] Using EMA model")
    else:
        print(f"[Eval] Using regular model")
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    print(f"[Eval] Policy loaded and set to eval mode")
    print(f"[Eval] Policy device: {device}")
    
    # Get task registry for success criteria
    print(f"\n[Eval] Loading task registry for: {task}")
    registry_class = registry.get_task_registry(task)
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
    
    # Pass validation episode indices if runner accepts it
    if val_episode_indices is not None and 'episode_indices' in sig.parameters:
        # Limit to requested number of episodes
        runner_kwargs['episode_indices'] = val_episode_indices[:n_episodes] if n_episodes else val_episode_indices
        print(f"[Eval] Using validation episode indices: {runner_kwargs['episode_indices']}")
    elif val_episode_indices is not None and 'val_episode_indices' in sig.parameters:
        runner_kwargs['val_episode_indices'] = val_episode_indices[:n_episodes] if n_episodes else val_episode_indices
        print(f"[Eval] Using validation episode indices: {runner_kwargs['val_episode_indices']}")
    
    # Override headless if runner accepts it
    if headless and 'headless' in sig.parameters:
        runner_kwargs['headless'] = True
    
    # Ensure shape_meta is passed
    if 'shape_meta' in sig.parameters:
        runner_kwargs['shape_meta'] = cfg.shape_meta
    
    # Instantiate env runner
    print(f"\n[Eval] Creating environment runner: {env_runner_cfg['_target_']}")
    print(f"[Debug] Calling hydra.utils.instantiate for env_runner...")
    env_runner = hydra.utils.instantiate(
        env_runner_cfg,
        **runner_kwargs
    )
    print(f"[Debug] Env runner instantiated.")
    
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
            episode_idx = val_episode_indices[i] if val_episode_indices and i < len(val_episode_indices) else i
            print(f"[Eval]   Episode {i+1} (dataset idx {episode_idx}): {status} (length: {ep.get('episode_length', 0)} steps)")
    
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
    print(f"[Eval] ==========================================")
    
    return json_log


if __name__ == '__main__':
    main()

