import os
import time
import numpy as np
import torch
import cv2
import zarr
from loguru import logger
from typing import Dict, List, Optional, Any
from scipy.spatial.transform import Rotation as R

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

# Isaac Sim imports
import omni.usd
from isaacsim import SimulationApp
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.prims import SingleXFormPrim, RigidPrim
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera

class IsaacSimRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir: str,
        n_episodes: int = 1,
        max_steps_per_episode: int = 200,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        urdf_path: str = "/workspace/voilab/assets/franka_panda/franka_panda_umi-isaacsim.urdf",
        usd_path: str = "/workspace/voilab/assets/ED305_scene/ED305.usd",
        headless: bool = True,
        save_video: bool = True,
        save_observation_data: bool = False,
        **kwargs
    ):
        super().__init__(output_dir)
        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.urdf_path = urdf_path
        self.usd_path = usd_path
        self.headless = headless
        self.save_video = save_video
        self.save_observation_data = save_observation_data

        # Robot config
        self.franka_panda_usd = "/workspace/voilab/assets/franka_panda/franka_panda_arm.usd"
        self.franka_prim_path = "/World/Franka"
        self.lula_robot_description_path = "/workspace/voilab/assets/lula/frank_umi_descriptor.yaml"

        # Rotation transformers
        self.rot_transformer = RotationTransformer(from_rep='rotation_6d', to_rep='quaternion')
        self.obs_rot_transformer = RotationTransformer(from_rep='quaternion', to_rep='rotation_6d')

        # World and state
        self.world = None
        self.panda = None
        self.camera = None
        self.lula_solver = None
        self.art_kine_solver = None
        self.start_eef_rot_quat = None

    def _setup_simulation(self):
        if self.world is not None:
            return

        print(f"[Debug] IsaacSimRunner: Setting up simulation with usd_path={self.usd_path}")
        logger.info("[IsaacSimRunner] Setting up Isaac Sim environment")
        
        # Enable necessary extensions
        print("[Debug] Enabling extensions...")
        enable_extension("isaacsim.robot_motion.motion_generation")

        # Open stage
        print(f"[Debug] Opening stage: {self.usd_path}")
        stage_utils.open_stage(self.usd_path)
        print("[Debug] Stage opened. Creating World...")
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # Add robot
        print(f"[Debug] Adding robot from: {self.franka_panda_usd}")
        robot_prim = stage_utils.add_reference_to_stage(usd_path=self.franka_panda_usd, prim_path=self.franka_prim_path)
        
        # Configure gripper
        print("[Debug] Configuring gripper...")
        gripper = ParallelGripper(
            end_effector_prim_path=self.franka_prim_path + "/panda/panda_rightfinger",
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.05, 0.05]),
            joint_closed_positions=np.array([0.02, 0.02]),
            action_deltas=np.array([0.01, 0.01]),
        )

        print("[Debug] Adding SingleManipulator...")
        self.panda = self.world.scene.add(
            SingleManipulator(
                prim_path=self.franka_prim_path,
                name="eval_franka",
                end_effector_prim_path=self.franka_prim_path + "/panda/panda_rightfinger",
                gripper=gripper,
            )
        )
        
        # Initialize solvers
        print("[Debug] Initializing kinematics solvers...")
        self.lula_solver = LulaKinematicsSolver(
            robot_description_path=self.lula_robot_description_path,
            urdf_path=self.urdf_path
        )
        self.art_kine_solver = ArticulationKinematicsSolver(
            robot_articulation=self.panda,
            kinematics_solver=self.lula_solver,
            end_effector_frame_name="umi_tcp"
        )

        # Setup camera
        print("[Debug] Setting up camera...")
        self.camera = Camera(
            prim_path="/World/Franka/panda/panda_link7/gopro_link/camera",
            resolution=(224, 224)
        )
        self.camera.initialize()

        print("[Debug] Resetting world...")
        self.world.reset()
        logger.info("[IsaacSimRunner] Setup complete")
        print("[Debug] IsaacSimRunner: Setup complete")

    def get_obs(self) -> Dict[str, np.ndarray]:
        # RGB
        print("[Debug] Getting RGB...")
        rgb = self.camera.get_rgb()
        print("[Debug] Got RGB.")
        if rgb is None:
            # Fallback for headless or skip frames
            rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Transpose to (C, H, W)
        rgb = rgb.transpose(2, 0, 1)

        # EEF Pose
        base_pos, base_quat = self.panda.get_world_pose()
        self.lula_solver.set_robot_base_pose(robot_position=base_pos, robot_orientation=base_quat)
        ee_pos, ee_rot_matrix = self.art_kine_solver.compute_end_effector_pose()
        
        # EEF position
        eef_pos = ee_pos.astype(np.float32)

        # EEF rotation (axis-angle)
        ee_rot_quat_xyzw = R.from_matrix(ee_rot_matrix[:3, :3]).as_quat()
        ee_rot_quat_wxyz = ee_rot_quat_xyzw[[3, 0, 1, 2]]
        
        # Convert to 6D rotation as expected by policy (despite the key name being axis_angle)
        eef_rot_6d = self.obs_rot_transformer.forward(
            np.array([ee_rot_quat_wxyz])
        )[0].astype(np.float32)

        # Relative rotation
        if self.start_eef_rot_quat is None:
            self.start_eef_rot_quat = ee_rot_quat_xyzw
        
        start_rot = R.from_quat(self.start_eef_rot_quat)
        curr_rot = R.from_quat(ee_rot_quat_xyzw)
        rel_rot = curr_rot * start_rot.inv()
        rel_rot_xyzw = rel_rot.as_quat()
        rel_rot_wxyz = rel_rot_xyzw[[3, 0, 1, 2]]
        rel_rot_6d = self.obs_rot_transformer.forward(
             np.array([rel_rot_wxyz])
        )[0].astype(np.float32)

        # Gripper width
        joint_pos = self.panda.get_joint_positions()
        gripper_width = np.array([joint_pos[-2] + joint_pos[-1]], dtype=np.float32)

        return {
            'camera0_rgb': rgb,
            'robot0_eef_pos': eef_pos,
            'robot0_eef_rot_axis_angle': eef_rot_6d, # Shape (6,)
            'robot0_eef_rot_axis_angle_wrt_start': rel_rot_6d, # Shape (6,)
            'robot0_gripper_width': gripper_width
        }

    def run(self, policy: BaseImagePolicy) -> Dict:
        self._setup_simulation()
        device = policy.device
        
        all_episode_stats = []
        
        for episode_idx in range(self.n_episodes):
            logger.info(f"[IsaacSimRunner] Starting episode {episode_idx+1}/{self.n_episodes}")
            self.world.reset()
            self.start_eef_rot_quat = None
            policy.reset()
            print(f"[Debug] Episode {episode_idx+1}: World reset done.")
            
            obs_buffer = {k: [] for k in ['camera0_rgb', 'robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_eef_rot_axis_angle_wrt_start', 'robot0_gripper_width']}
            
            frames = []
            done = False
            step_idx = 0
            
            while not done and step_idx < self.max_steps_per_episode:
                # Capture current observation
                obs = self.get_obs()
                for k, v in obs.items():
                    obs_buffer[k].append(v)
                    if len(obs_buffer[k]) > self.n_obs_steps:
                        obs_buffer[k].pop(0)

                if len(obs_buffer['camera0_rgb']) < self.n_obs_steps:
                    print(f"[Debug] Warming up obs buffer {len(obs_buffer['camera0_rgb'])}/{self.n_obs_steps}")
                    self.world.step(render=not self.headless)
                    continue

                # Prepare observations for policy
                # Policy expects [1, n_obs_steps, ...]
                policy_obs = {}
                for k, v in obs_buffer.items():
                    val = torch.from_numpy(np.array(v)).unsqueeze(0).to(device)
                    if k == 'camera0_rgb':
                        # Input is already (B, T, C, H, W) = (1, T, 3, 224, 224)
                        # Normalize to [0,1] float32
                        val = val.float() / 255.0
                    policy_obs[k] = val
                
                # Predict action
                print("[Debug] Predicting action...")
                with torch.no_grad():
                    action_dict = policy.predict_action(policy_obs)
                print("[Debug] Action predicted.")
                
                # Execute action (n_action_steps)
                actions = action_dict['action'][0].cpu().numpy() # [horizon, 10]
                
                for i in range(min(self.n_action_steps, len(actions))):
                    action = actions[i]
                    # action: [pos(3), rot6d(6), gripper(1)]
                    target_pos = action[:3]
                    target_gripper_width = action[9]

                    target_rot6d = action[3:9]
                    target_rot_quat_wxyz = self.rot_transformer.forward(target_rot6d[None, :])[0]
                    target_rot_quat_xyzw = target_rot_quat_wxyz[[1, 2, 3, 0]]
                    # Note: Isaac Sim uses [w, x, y, z] for orientation in some places, but ArticulationKinematicsSolver uses [w, x, y, z] too.
                    
                    # Compute IK
                    ik_action, success = self.art_kine_solver.compute_inverse_kinematics(
                        target_position=target_pos,
                        target_orientation=target_rot_quat_wxyz
                    )
                    
                    if success:
                        self.panda.set_joint_positions(ik_action.joint_positions, np.arange(7))
                        # Set gripper
                        g_pos = target_gripper_width / 2.0
                        self.panda.gripper.set_joint_positions(np.array([g_pos, g_pos]))
                    
                    
                    print(f"[Debug] Stepping simulation (Action step {i})...")
                    self.world.step(render=not self.headless)
                    
                    if self.save_video:
                        frame = self.camera.get_rgb()
                        if frame is not None:
                            frames.append(frame)
                    
                    step_idx += 1
                    if step_idx >= self.max_steps_per_episode:
                        break
                
                # In a real eval, we would check success here using registry
                # For now, we just run to max steps or a simple done flag
                # (You can integrate registry.is_episode_completed(info) here)

            # Save video
            if self.save_video and frames:
                video_path = os.path.join(self.output_dir, f"eval_ep_{episode_idx}.mp4")
                height, width, _ = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
                for f in frames:
                    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                out.release()
                logger.info(f"[IsaacSimRunner] Saved video to {video_path}")

            all_episode_stats.append({
                'success': False, # Update with real success check
                'episode_length': step_idx
            })

        return {
            'episode_stats': all_episode_stats, # Required by eval script
            'success_rate': np.mean([s['success'] for s in all_episode_stats]),
            'avg_steps': np.mean([s['episode_length'] for s in all_episode_stats])
        }
