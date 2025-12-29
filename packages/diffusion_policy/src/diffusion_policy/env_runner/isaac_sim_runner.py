import os
import time
import numpy as np
import torch
import cv2
import zarr
from loguru import logger
from typing import Dict, List, Optional, Any
import collections
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
        self.check_success_fn = None
        self.registry_config = None

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
        
        if self.registry_config:
            # 1. Apply Robot Pose
            franka_pose = self.registry_config.get("franka_pose", {})
            trans = franka_pose.get("translation")
            rot = franka_pose.get("rotation_quat")
            if trans is not None and rot is not None:
                print(f"[Debug] Setting Registry Robot Pose: {trans}")
                robot_xform = SingleXFormPrim(prim_path=self.franka_prim_path)
                robot_xform.set_local_pose(
                    translation=np.array(trans) / stage_utils.get_stage_units(),
                    orientation=np.array(rot)
                )

            # 2. Load Preload Objects
            env_vars = self.registry_config.get("environment_vars", {})
            preload_objects = env_vars.get("PRELOAD_OBJECTS", [])
            ASSETS_DIR = "/workspace/voilab/assets"
            self.object_prims = {}
            
            for entry in preload_objects:
                raw_name = entry.get("name")
                asset_filename = entry.get("assets")
                prim_path = entry.get("prim_path")
                
                if not (raw_name and asset_filename and prim_path):
                    continue

                full_asset_path = os.path.join(ASSETS_DIR, asset_filename)
                if not os.path.exists(full_asset_path):
                    print(f"[IsaacSimRunner] WARNING: Asset not found: {full_asset_path}")
                    continue

                try:
                    stage_utils.add_reference_to_stage(
                        usd_path=full_asset_path,
                        prim_path=prim_path
                    )
                    # Create Prim wrapper
                    obj_prim = SingleXFormPrim(prim_path=prim_path, name=raw_name)
                    self.object_prims[raw_name] = obj_prim
                    print(f"[IsaacSimRunner] Loaded object: {raw_name} at {prim_path}")
                except Exception as e:
                    print(f"[IsaacSimRunner] ERROR loading {raw_name}: {e}")
        
        # Configure gripper
        
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
            prim_path="/World/Franka/panda/panda_link7/gopro_link/Camera",
            resolution=(224, 224)
        )
        self.camera.initialize()

        print("[Debug] Resetting world...")
        self.world.reset()
        logger.info("[IsaacSimRunner] Setup complete")
        print("[Debug] IsaacSimRunner: Setup complete")

    def set_registry_config(self, config: Dict):
        self.registry_config = config
        print("[Debug] IsaacSimRunner: Registry config set.")

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
        
        n_obs_steps = 2
        
        for episode_idx in range(self.n_episodes):
            logger.info(f"[IsaacSimRunner] Starting episode {episode_idx+1}/{self.n_episodes}")
            self.world.reset()
            # Render once to get valid camera data
            self.world.step(render=True)
            
            self.start_eef_rot_quat = None
            policy.reset()
            print(f"[Debug] Episode {episode_idx+1}: World reset done.")
            
            # Reset Objects (Cups) to fixed initial positions if loaded
            if hasattr(self, 'object_prims') and self.object_prims:
                # Expected Robot Base: [4.5, 2.7, 0.9]
                # Table Height: ~0.9
                
                # Pink Cup (Left-ish, forward)
                if 'pink cup' in self.object_prims:
                    # [4.85, 2.60, 0.92] (Z slightly above table)
                    pos = np.array([4.85, 2.60, 0.92]) 
                    self.object_prims['pink cup'].set_world_pose(position=pos)
                    print(f"[IsaacSimRunner] Reset pink cup to {pos}")
                
                # Blue Cup (Right-ish, forward)
                if 'blue cup' in self.object_prims:
                     # [4.85, 2.80, 0.92]
                    pos = np.array([4.85, 2.80, 0.92])
                    self.object_prims['blue cup'].set_world_pose(position=pos)
                    print(f"[IsaacSimRunner] Reset blue cup to {pos}")

                # Allow physics to settle
                for _ in range(10): 
                    self.world.step(render=False)
            
            obs_buffer = collections.deque(maxlen=n_obs_steps)
            # Warm up
            for _ in range(n_obs_steps):
                obs_buffer.append(self.get_obs())
            
            video_frames = []
            is_success = False
            done = False
            step_idx = 0
            
            while not done and step_idx < self.max_steps_per_episode:
                # Prepare Observation Batch (B, T, D)
                # Stack
                # obs_buffer contains dicts. We want dict of (B=1, T, D)
                current_obs = self.get_obs()
                obs_buffer.append(current_obs)
                
                batch_obs = {}
                # Assume all keys in buffer are present in all entries
                keys = obs_buffer[0].keys()
                for key in keys:
                    # Stack along Time (creates T, ...)
                    stacked = np.stack([x[key] for x in obs_buffer])
                    # Add Batch Dim (1, T, ...)
                    batch_obs[key] = stacked[None, ...]
                
                # Relativize Poses (robot0_eef_pos, robot0_eef_rot_axis_angle) wrt Current (Last) Frame
                # UMI Expects relative obs
                
                # Position: Seq - Current
                key_pos = 'robot0_eef_pos'
                if key_pos in batch_obs:
                     current_pos = batch_obs[key_pos][:, -1:, :] # (1, 1, 3)
                     batch_obs[key_pos] = batch_obs[key_pos] - current_pos
                
                # Rotation: Relative to Current (R_current.inv * R_seq) or similar
                # Using our transformer (Forward: Quat->6D, Inverse: 6D->Quat)
                # Current obs eef_rot_axis_angle is 6D.
                # Compute Relative Rotation Sequence
                key_rot = 'robot0_eef_rot_axis_angle'
                if key_rot in batch_obs:
                    # Convert to Quat/Matrix to compute delta
                    # (1, T, 6)
                    B, T, D = batch_obs[key_rot].shape
                    flat_rot6d = batch_obs[key_rot].reshape(B*T, D)
                    # 6D -> Quat (wxyz) -> Scipy (xyzw)
                    rot_quat_wxyz = self.rot_transformer.forward(flat_rot6d)
                    rot_quat_xyzw = rot_quat_wxyz[:, [1, 2, 3, 0]]
                    rot_objs = R.from_quat(rot_quat_xyzw)
                    
                    # Current Rot (Last in sequence)
                    # Reshape back to identify last
                    # Last rot is at index (B*T - 1) if B=1
                    current_rot_obj = rot_objs[-1] 
                    
                    # Relativize: target_rel = current.inv * target (Local) or target * current.inv?
                    # UMI logic (umi_dataset.py): 
                    # convert_pose_mat_rep(..., base=current, rep='relative')
                    # -> t_rel = t_world * base_world.inv()  (if matrix multiplication order T_rel = T_world * T_base^-1 ?) -- No
                    # Usually T_target_in_base = T_base_world.inv() * T_target_world
                    # So Rot_rel = Rot_base.inv() * Rot_target
                    
                    # Apply to all
                    rot_rel_objs = current_rot_obj.inv() * rot_objs
                    rot_rel_quat_xyzw = rot_rel_objs.as_quat()
                    rot_rel_quat_wxyz = rot_rel_quat_xyzw[:, [3, 0, 1, 2]]
                    
                    # Back to 6D
                    rot_rel_6d = self.rot_transformer.inverse(rot_rel_quat_wxyz)
                    batch_obs[key_rot] = rot_rel_6d.reshape(B, T, D)

                
                # Predict action
                # print("[Debug] Predicting action...")
                with torch.no_grad():
                    action_dict = policy.predict_action(batch_obs)
                # print("[Debug] Action predicted.")
                
                # Execute action (n_action_steps)
                # Action is (B=1, Horizon, D) -> (Horizon, D)
                actions = action_dict['action'][0].cpu().numpy() # [horizon, 10]
                
                # Get start pose for relative actions
                current_ee_pos, current_ee_mat = self.art_kine_solver.compute_end_effector_pose()
                current_ee_quat_xyzw = R.from_matrix(current_ee_mat[:3, :3]).as_quat()
                current_ee_rot = R.from_quat(current_ee_quat_xyzw)

                for i in range(min(self.n_action_steps, len(actions))):
                    action = actions[i]
                    # action: [pos(3), rot6d(6), gripper(1)]
                    
                    # Relative Position: Add to current
                    delta_pos = action[:3]
                    # Debug values
                    if i == 0:
                        print(f"[Debug] Current EE: {current_ee_pos}")
                        print(f"[Debug] Action Delta: {delta_pos}")
                    
                    target_pos = current_ee_pos + delta_pos

                    # Relative Rotation: Compose with current
                    delta_rot6d = action[3:9]
                    delta_rot_quat_wxyz = self.rot_transformer.forward(delta_rot6d[None, :])[0] # shape (4,)
                    delta_rot_quat_xyzw = delta_rot_quat_wxyz[[1, 2, 3, 0]]
                    delta_rot = R.from_quat(delta_rot_quat_xyzw)
                    
                    # Target = Delta * Current (Global delta, consistent with global position delta)
                    target_rot = delta_rot * current_ee_rot
                    target_rot_quat_xyzw = target_rot.as_quat()
                    
                    # Target Orientation for IK (wxyz check: ArticulationKinematicsSolver expects wxyz?)
                    # In Step 251 (original code), it used: target_rot_quat_wxyz
                    # And: target_orientation=target_rot_quat_wxyz
                    target_rot_quat_wxyz = target_rot_quat_xyzw[[3, 0, 1, 2]]
                    
                    # Gripper
                    target_gripper_width = action[9]
                    
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
                    else:
                        print(f"[Debug] IK Failed for step {i}! Target Pos: {target_pos}")
                    
                    
                    # print(f"[Debug] Stepping simulation (Action step {i})...")
                    self.world.step(render=True)
                    
                    if self.save_video:
                        frame = self.camera.get_rgb()
                        if frame is not None:
                            video_frames.append(frame)
                    
                    step_idx += 1
                    if step_idx >= self.max_steps_per_episode:
                        break
                
                if self.check_success_fn:
                    try:
                        # Pass context if needed, currently unused by registry
                        is_success = self.check_success_fn({})
                        if is_success:
                            done = True
                            print(f"[IsaacSimRunner] Episode {episode_idx+1} Success!")
                    except Exception as e:
                        print(f"[IsaacSimRunner] Warning: Success check failed: {e}")
                        
            # Save video
            if self.save_video and video_frames:
                video_path = os.path.join(self.output_dir, f"eval_ep_{episode_idx}.mp4")
                height, width, _ = video_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
                for f in video_frames:
                    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                out.release()
                logger.info(f"[IsaacSimRunner] Saved video to {video_path}")

            all_episode_stats.append({
                'episode_idx': episode_idx,
                'length': step_idx,
                'success': is_success
            })

        print("[IsaacSimRunner] Evaluation complete.")
        success_rate = np.mean([s['success'] for s in all_episode_stats])
        avg_length = np.mean([s['length'] for s in all_episode_stats])
        
        return {
            'episode_stats': all_episode_stats,
            'success_rate': success_rate,
            'avg_episode_length': avg_length
        }
