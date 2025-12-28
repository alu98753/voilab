import os
import click
import subprocess
import sys
from pathlib import Path

from voila.app import Voila


@click.group()
def cli():
    pass


@cli.command()
def launch_viewer():
    """Launch the replay buffer viewer."""
    argv = ["--no-browser", "nbs/replay_buffer_viewer.ipynb"]
    Voila.launch_instance(argv=argv)


@cli.command()
def launch_dataset_visualizer():
    """Launch the dataset visualizer for reviewing collected demonstrations."""
    argv = ["--no-browser", "nbs/dataset_visualizer.ipynb"]
    Voila.launch_instance(argv=argv)


@cli.command()
@click.option('--task', type=click.Choice(['kitchen', 'dining-room', 'living-room']),
              required=True, help='Task environment to load')
@click.option('--session_dir', type=str, help='Path to UMI session directory for trajectory replay')
@click.option('--episode', default=0, type=int, help='Episode to replay')
@click.option('--width', default=1280, help='Window width')
@click.option('--height', default=720, help='Window height')
def launch_simulator(task, session_dir, episode, width, height):
    """Launch Isaac Sim with ROS2 bridge enabled"""
    try:
        # Prepare environment
        env_vars = os.environ.copy()
        env_vars.update({
            "OMNI_KIT_ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "DISPLAY": os.getenv("DISPLAY", ":1"),
            "NVIDIA_VISIBLE_DEVICES": "all",
            "NVIDIA_DRIVER_CAPABILITIES": "all,graphics,display,utility,compute",
            "ROS_LOCALHOST_ONLY": "0",
            "ROS_DOMAIN_ID": "0",
            "ROS_DISTRO": "humble",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
            "TASK_NAME": task,
            "WINDOW_WIDTH": str(width),
            "WINDOW_HEIGHT": str(height),
        })
        
        click.echo(f"[CLI] Launching Isaac Sim + ROS2: task={task}, resolution={width}x{height}")
        
        # Convert host path to container path
        # Docker maps the project root to /workspace/voilab
        if session_dir:
            # Get the project root (where docker-compose.yaml is located)
            # This script is in src/voilab/cli.py, so project root is 2 levels up
            script_dir = Path(__file__).parent.parent.parent
            project_root = str(script_dir.resolve())
            
            session_dir_abs = os.path.abspath(session_dir) if not os.path.isabs(session_dir) else session_dir
            
            # If session_dir is within project root, convert to container path
            if session_dir_abs.startswith(project_root):
                relative_path = os.path.relpath(session_dir_abs, project_root)
                container_session_dir = f"/workspace/voilab/{relative_path}"
                click.echo(f"[CLI] Converted path: {session_dir} -> {container_session_dir}")
            # If session_dir is already relative, assume it's relative to project root
            elif not os.path.isabs(session_dir):
                container_session_dir = f"/workspace/voilab/{session_dir}"
                click.echo(f"[CLI] Using relative path: {container_session_dir}")
            # If it's an absolute path outside project root, try to use it directly (may not work)
            else:
                container_session_dir = session_dir
                click.echo(f"[CLI] Warning: session_dir is outside project root, using as-is: {session_dir}", err=True)
        else:
            container_session_dir = session_dir
        
        # Build image
        click.echo("[CLI] Building Docker image...")
        build_cmd = ["docker", "compose", "build", "isaac-sim"]
        subprocess.run(build_cmd, env=env_vars, check=True)

        # container_command = f".venv/bin/python scripts/generate_data.py --task {task} --session_dir {session_dir} --episode {episode}"
        container_command = f".venv/bin/python scripts/generate_data.py --task {task} --session_dir {container_session_dir}"
        # Run container with host network
        click.echo("[CLI] Starting Docker container with host network...")
        compose_run_cmd = [
            "docker", "compose", "run", "--rm",
            "isaac-sim",
            "/bin/bash", "-c",
            container_command
        ]
        subprocess.run(compose_run_cmd, env=env_vars, check=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"[ERROR] Docker execution failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"[ERROR] {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--checkpoint', type=str, required=True, help='Path to checkpoint file')
@click.option('--output_dir', type=str, required=True, help='Output directory for evaluation results')
@click.option('--task', type=click.Choice(['kitchen', 'dining-room', 'living-room']),
              required=True, help='Task name')
@click.option('--dataset_path', type=str, default=None, help='Path to dataset zarr zip file (optional)')
@click.option('--n_episodes', default=10, type=int, help='Number of evaluation episodes')
@click.option('--headless', is_flag=True, help='Run in headless mode (no GUI)')
@click.option('--device', default='cuda:0', help='Device to run on (cuda:0, cpu, etc.)')
def eval_model(checkpoint, output_dir, task, dataset_path, n_episodes, headless, device):
    """Evaluate trained Diffusion Policy model in Isaac Sim environment"""
    try:
        # Prepare environment
        env_vars = os.environ.copy()
        env_vars.update({
            "OMNI_KIT_ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "DISPLAY": os.getenv("DISPLAY", ":1"),
            "NVIDIA_VISIBLE_DEVICES": "all",
            "NVIDIA_DRIVER_CAPABILITIES": "all,graphics,display,utility,compute",
            "ROS_LOCALHOST_ONLY": "0",
            "ROS_DOMAIN_ID": "0",
            "ROS_DISTRO": "humble",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
            "TASK_NAME": task,
        })
        
        click.echo(f"[CLI] Evaluating model: checkpoint={checkpoint}, task={task}, n_episodes={n_episodes}")
        
        # Get project root
        script_dir = Path(__file__).parent.parent.parent
        project_root = str(script_dir.resolve())
        
        # Convert paths to container paths
        def to_container_path(host_path):
            if not host_path:
                return None
            host_path_abs = os.path.abspath(host_path) if not os.path.isabs(host_path) else host_path
            if host_path_abs.startswith(project_root):
                relative_path = os.path.relpath(host_path_abs, project_root)
                return f"/workspace/voilab/{relative_path}"
            elif not os.path.isabs(host_path):
                return f"/workspace/voilab/{host_path}"
            else:
                click.echo(f"[CLI] Warning: Path outside project root: {host_path}", err=True)
                return host_path
        
        container_checkpoint = to_container_path(checkpoint)
        container_output_dir = to_container_path(output_dir)
        container_dataset_path = to_container_path(dataset_path) if dataset_path else None
        
        # Build command arguments
        cmd_args = [
            ".venv/bin/python", "scripts/eval_kitchen.py",
            "--checkpoint", container_checkpoint,
            "--output_dir", container_output_dir,
            "--task", task,
            "--n_episodes", str(n_episodes),
            "--device", device,
        ]
        
        if headless:
            cmd_args.append("--headless")
        if container_dataset_path:
            cmd_args.extend(["--dataset_path", container_dataset_path])
        
        container_command = " ".join(cmd_args)
        
        # Build image
        click.echo("[CLI] Building Docker image...")
        build_cmd = ["docker", "compose", "build", "isaac-sim"]
        subprocess.run(build_cmd, env=env_vars, check=True)
        
        # Run container
        click.echo("[CLI] Starting Docker container for evaluation...")
        compose_run_cmd = [
            "docker", "compose", "run", "--rm",
            "isaac-sim",
            "/bin/bash", "-c",
            container_command
        ]
        subprocess.run(compose_run_cmd, env=env_vars, check=True)
        
    except subprocess.CalledProcessError as e:
        click.echo(f"[ERROR] Docker execution failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"[ERROR] {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--session-dir', type=str, required=True, help='Path to the session directory to replay')
def replay_trajectory(session_dir):
    """Replay a trajectory from a recorded session directory."""
    try:
        # Validate session directory
        if not session_dir:
            click.echo("[ERROR] Session directory is required", err=True)
            sys.exit(1)

        # Prepare environment
        env_vars = os.environ.copy()

        click.echo(f"[CLI] Replaying trajectory from session: {session_dir}")

        # Run pose publisher in docker container
        compose_docker_cmd = [
            "docker", "compose", "run", "voilab-workspace",
            "python", "packages/diffusion_policy/examples/run_dataset_pose_publisher.py",
            "--session_dir", session_dir
        ]

        subprocess.run(compose_docker_cmd, env=env_vars, check=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"[ERROR] Docker execution failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"[ERROR] {str(e)}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    cli()
