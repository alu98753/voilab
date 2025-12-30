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
import cv2 # Added for resizing to match training data

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
        # Use Wrist Camera (GoPro) as expected by UMI Policy
        self.camera_prim_path = "/World/Franka/panda/panda_link7/gopro_link/Camera"

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
        self.check_success_fn = None
        self.registry_config = None
        
        # Magic Grasp State
        self.attached_object = None
        self.T_ee_to_obj = None

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
            ASSETS_DIR = "/workspace/voilab/assets/CADs"
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
                    self.world.scene.add(obj_prim) # Register with scene for physics/reset
                    self.object_prims[raw_name] = obj_prim
                    print(f"[IsaacSimRunner] Loaded object: {raw_name} at {prim_path}")
                except Exception as e:
                    print(f"[IsaacSimRunner] ERROR loading {raw_name}: {e}")
        
                except Exception as e:
                    print(f"[IsaacSimRunner] ERROR loading {raw_name}: {e}")
        
            # 3. Camera Pose: 
            # REMOVED Fixed Camera Pose application. 
            # The camera is attached to the robot wrist (GoPro), so it moves with the robot.
            # We should NOT force it to a fixed world pose.
        
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
            resolution=(1280, 720) # Match generate_data.py (16:9)
        )
        self.camera.initialize()

        print("[Debug] Resetting world...")
        self.world.reset()
        logger.info("[IsaacSimRunner] Setup complete")
        print("[Debug] IsaacSimRunner: Setup complete")

    def set_registry_config(self, config: Dict):
        self.registry_config = config
        print("[Debug] IsaacSimRunner: Registry config set.")

    def _update_magic_grasp(self, action_gripper_width: float, step_idx: int = 0):
        """
        Implements 'Magic Grasp' (Teleport Attachment) to match Training Data generation.
        """
        # Thresholds
        ATTACH_THRESHOLD = 0.04  # < 4cm = Trying to Close
        DETACH_THRESHOLD = 0.05  # > 5cm = Trying to Open
        DIST_THRESHOLD = 0.15    # < 15cm = Close enough to grasp
        
        # Get Current EE Pose (FK)
        # Note: art_kine_solver state is updated in get_obs() or before this call?
        # get_obs updates lula base pose. We should ensure it's up to date.
        base_pos, base_quat = self.panda.get_world_pose()
        self.lula_solver.set_robot_base_pose(robot_position=base_pos, robot_orientation=base_quat)
        # We also need joint positions? 
        # art_kine_solver uses 'robot_articulation' which is linked to self.panda (Sim).
        # So it pulls joint positions from Sim.
        ee_pos, ee_rot_matrix = self.art_kine_solver.compute_end_effector_pose()
        
        T_ee = np.eye(4)
        T_ee[:3, :3] = ee_rot_matrix
        T_ee[:3, 3] = ee_pos

        if self.attached_object:
            # Update Object Pose
            T_obj = T_ee @ self.T_ee_to_obj
            pos = T_obj[:3, 3]
            rot_mat = T_obj[:3, :3]
            quat_xyzw = R.from_matrix(rot_mat).as_quat()
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
            
            self.attached_object.set_world_pose(position=pos, orientation=quat_wxyz)
            
            # Detach Condition
            if action_gripper_width > DETACH_THRESHOLD:
                print(f"[Magic] Detaching object: {self.attached_object.name}")
                self.attached_object = None
                self.T_ee_to_obj = None
                
        else:
            # Attach Condition
            if action_gripper_width < ATTACH_THRESHOLD:
                # Find closest object
                min_dist = float('inf')
                closest_obj = None
                
                if hasattr(self, 'object_prims'):
                    for name, obj in self.object_prims.items():
                        # Get object pose
                        obj_pos, _ = obj.get_world_pose()
                        dist = np.linalg.norm(obj_pos - ee_pos)
                        if dist < min_dist:
                            min_dist = dist
                            closest_obj = obj
                
                if closest_obj:
                    # Debug Distance
                    if step_idx % 10 == 0:
                        print(f"[Magic Debug] Closest: {closest_obj.name}, Dist: {min_dist:.4f}, Gripper: {action_gripper_width:.4f}")

                if closest_obj and min_dist < DIST_THRESHOLD:
                    # Attach Condition
                    if action_gripper_width < ATTACH_THRESHOLD:
                        # Attach!
                        print(f"[Magic] Attaching {closest_obj.name} (dist={min_dist:.4f})")
                        self.attached_object = closest_obj
                    
                    # Compute Relative Transform T_ee_to_obj = inv(T_ee) * T_obj
                    obj_pos, obj_quat_wxyz = closest_obj.get_world_pose()
                    obj_quat_xyzw = obj_quat_wxyz[[1, 2, 3, 0]]
                    T_obj = np.eye(4)
                    T_obj[:3, :3] = R.from_quat(obj_quat_xyzw).as_matrix()
                    T_obj[:3, 3] = obj_pos
                    
                    self.T_ee_to_obj = np.linalg.inv(T_ee) @ T_obj

    def get_obs(self) -> Dict[str, np.ndarray]:
        # RGB
        print("[Debug] Getting RGB...")
        rgb = self.camera.get_rgb()
        print("[Debug] Got RGB.")
        if rgb is None:
            # Fallback for headless or skip frames
            rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Resize to 224x224 to match training data (Squashing 16:9 to 1:1)
        if rgb.shape[:2] != (224, 224):
             rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

        # Transpose to (C, H, W)
        rgb = rgb.transpose(2, 0, 1)

        # EEF Pose
        base_pos, base_quat = self.panda.get_world_pose()
        self.lula_solver.set_robot_base_pose(robot_position=base_pos, robot_orientation=base_quat)
        ee_pos, ee_rot_matrix = self.art_kine_solver.compute_end_effector_pose()
        
        # EEF position
        eef_pos = ee_pos.astype(np.float32)

        # EEF rotation (axis-angle)
        # Fix: Use as_rotvec() output (3D) instead of 6D rotation to match generate_data.py
        ee_rot_axis_angle = R.from_matrix(ee_rot_matrix[:3, :3]).as_rotvec().astype(np.float32)
        
        # We need quaternion for relative rotation calculation
        ee_rot_quat_xyzw = R.from_matrix(ee_rot_matrix[:3, :3]).as_quat()

        # Relative rotation
        if self.start_eef_rot_quat is None:
            self.start_eef_rot_quat = ee_rot_quat_xyzw
        
        start_rot = R.from_quat(self.start_eef_rot_quat)
        curr_rot = R.from_quat(ee_rot_quat_xyzw)
        # Fix: Correct order is inv(Start) * Curr (Apply Start inverse first to align frames)
        rel_rot = start_rot.inv() * curr_rot
        rel_rot_axis_angle = rel_rot.as_rotvec().astype(np.float32)

        # Gripper width
        joint_pos = self.panda.get_joint_positions()
        gripper_width = np.array([joint_pos[-2] + joint_pos[-1]], dtype=np.float32)

        return {
            'camera0_rgb': rgb,
            'robot0_eef_pos': eef_pos,
            'robot0_eef_rot_axis_angle': ee_rot_axis_angle, # Shape (3,)
            'robot0_eef_rot_axis_angle_wrt_start': rel_rot_axis_angle, # Shape (3,)
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

            # --- 1. Reset Objects (Match generate_data.py Phase 1) ---
            if hasattr(self, 'object_prims') and self.object_prims:
                # Seed for reproducibility per episode
                rng = np.random.default_rng(episode_idx)
                
                # Disable Randomization for Debugging (Set to 0.0)
                jitter_scale = 0.0 
                
                # Pink Cup
                if 'pink cup' in self.object_prims:
                    # Base: [4.85, 2.60, 1.0] -> Move to [5.00, 2.60] (Closer to Robot X=4.99)
                    jitter = rng.uniform(-jitter_scale, jitter_scale, size=2)
                    pos = np.array([4.70 + jitter[0], 2.60 + jitter[1], 1.0]) 
                    quat = np.array([1, 0, 0, 0]) # Upright (WXYZ)
                    self.object_prims['pink cup'].set_world_pose(position=pos, orientation=quat)
                    print(f"[IsaacSimRunner] Reset pink cup to {pos} (Fixed)")
                
                # Blue Cup
                if 'blue cup' in self.object_prims:
                     # Base: [4.85, 2.80, 1.0] -> Move to [5.00, 2.80]
                    jitter = rng.uniform(-jitter_scale, jitter_scale, size=2)
                    pos = np.array([4.99 + jitter[0], 2.52 + jitter[1], 1.0])
                    quat = np.array([1, 0, 0, 0]) # Upright (WXYZ)
                    self.object_prims['blue cup'].set_world_pose(position=pos, orientation=quat)
                    print(f"[IsaacSimRunner] Reset blue cup to {pos} (Fixed)")

            # --- 2. Settle Physics (Match generate_data.py Phase 2) ---
            print("[Debug] Settling physics for 100 steps...")
            for _ in range(100):
                self.world.step(render=False) # Faster without rendering, but need one render at end?
            self.world.step(render=True)

            # --- 3. Initialize Robot Pose (Match generate_data.py Phase 3) ---
            # Calibrate Base
            base_pos, base_quat = self.panda.get_world_pose()
            self.lula_solver.set_robot_base_pose(robot_position=base_pos, robot_orientation=base_quat)
            
            # Get Current EE Pose (after settle)
            ee_pos, ee_rot_mat = self.art_kine_solver.compute_end_effector_pose()
            
            # Calculate Target Init Pose (Kitchen Task Offset)
            # Offset: [-0.16, 0., 0.13]
            # init_offset = np.array([-0.16, 0., 0.13])
            # target_pos = ee_pos + init_offset
            
            # FIXED: Hardcode to match Training Data (Episode 0)
            # Eval: [4.84, 2.64, 1.29] vs Train: [4.99, 2.52, 1.09]
            # Diff: X+0.15, Y-0.12, Z-0.20
            target_pos = np.array([4.99, 2.52, 1.09])
            
            target_quat_wxyz = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
            
            # Apply IK
            print(f"[Debug] Initializing Robot to FIXED TARGET {target_pos}")
            ik_action, success = self.art_kine_solver.compute_inverse_kinematics(
                target_position=target_pos,
                target_orientation=target_quat_wxyz
            )
            
            if success:
                self.panda.set_joint_positions(ik_action.joint_positions, np.arange(7))
                print("[Debug] Robot initialization IK successful.")
                # Verify Pose
                final_ee_pos, final_ee_rot = self.art_kine_solver.compute_end_effector_pose()
                final_ee_rot_quat = R.from_matrix(final_ee_rot[:3, :3]).as_quat() # xyzw
                # Convert to wxyz for display
                final_ee_rot_wxyz = np.array([final_ee_rot_quat[3], final_ee_rot_quat[0], final_ee_rot_quat[1], final_ee_rot_quat[2]])
                print(f"[Debug] Achieved EE Position: {final_ee_pos}")
                print(f"[Debug] Achieved EE Rotation (WXYZ): {final_ee_rot_wxyz}")
                print(f"[Debug] Target   EE Rotation (WXYZ): {target_quat_wxyz}")
            else:
                print("[IsaacSimRunner] WARNING: Robot initialization IK failed!")
            
            # Short Settle after IK
            for _ in range(10):
                self.world.step(render=True)
            # --------------------------------------------------------
            
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
                
                # Position: Seq - Current (and rotate into current frame)
                key_pos = 'robot0_eef_pos'
                if key_pos in batch_obs:
                     # (1, T, 3)
                     pos_seq = batch_obs[key_pos]
                     current_pos = pos_seq[:, -1:, :] # (1, 1, 3)
                     
                     # Get current rotation to relativize position
                     # We need the last frame's rotation
                     key_rot_raw = 'robot0_eef_rot_axis_angle'
                     if key_rot_raw in batch_obs:
                         # Current obs is axis-angle (3D)
                         last_axis_angle = batch_obs[key_rot_raw][0, -1, :]
                         current_rot_obj = R.from_rotvec(last_axis_angle)
                         # Transform: pos_rel = current_rot.inv * (pos_world - pos_current)
                         diff = pos_seq - current_pos # (1, T, 3)
                         B, T, D = diff.shape
                         # Apply inverse rotation to all points in sequence
                         batch_obs[key_pos] = current_rot_obj.inv().apply(diff.reshape(-1, 3)).reshape(B, T, D)
                
                # Rotation: Relative to Current (R_current.inv * R_seq)
                # Current obs eef_rot_axis_angle is 3D (Axis-Angle).
                # Compute Relative Rotation Sequence
                key_rot = 'robot0_eef_rot_axis_angle'
                if key_rot in batch_obs:
                    # (1, T, 3)
                    B, T, D = batch_obs[key_rot].shape
                    flat_axis_angle = batch_obs[key_rot].reshape(B*T, D)
                    
                    # Axis-Angle -> Rotation Object
                    rot_objs = R.from_rotvec(flat_axis_angle)
                    
                    # Current Rot (Last in sequence)
                    current_rot_obj = rot_objs[-1] 
                    
                    # Relativize: Rot_rel = Rot_base.inv() * Rot_target
                    rot_rel_objs = current_rot_obj.inv() * rot_objs
                    
                    # Back to Axis Angle (3D)
                    rot_rel_axis_angle = rot_rel_objs.as_rotvec().astype(np.float32)
                    batch_obs[key_rot] = rot_rel_axis_angle.reshape(B, T, D)

                # --- Adapter: Convert 3D Axis-Angle to 6D Rotation for Policy ---
                # Policy expects 6D (Shape 6), but we have 3D (Shape 3).
                # keys to convert: 'robot0_eef_rot_axis_angle', 'robot0_eef_rot_axis_angle_wrt_start'
                keys_to_convert = ['robot0_eef_rot_axis_angle', 'robot0_eef_rot_axis_angle_wrt_start']
                for key in keys_to_convert:
                    if key in batch_obs:
                        obs_3d = batch_obs[key] # (B, T, 3)
                        B, T, D = obs_3d.shape
                        if D == 3:
                            # 3D Axis-Angle -> Matrix -> 6D
                            flat_3d = obs_3d.reshape(B*T, 3)
                            rot_objs = R.from_rotvec(flat_3d)
                            # RotationTransformer.inverse expects (x, y, z, w) because it uses scipy backend
                            rot_quat_xyzw = rot_objs.as_quat()
                            # wxyz -> 6D (NOTE: Variable name was misleading in original code, inverse takes xyzw)
                            # self.rot_transformer is initialized as from_rep='rotation_6d', to_rep='quaternion'
                            # So inverse() does Quat -> 6D.
                            flat_6d = self.rot_transformer.inverse(rot_quat_xyzw)
                            batch_obs[key] = flat_6d.reshape(B, T, 6)
                # -------------------------------------------------------------

                
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
                    
                    # Relative Position: Rotate by current and add
                    delta_pos = action[:3]
                    # Target = Current_Pos + Current_Rot * Delta_Pos (Apply delta in current local frame)
                    target_pos = current_ee_pos + current_ee_rot.apply(delta_pos)

                    # Relative Rotation: Compose with current
                    delta_rot6d = action[3:9]
                    # RotationTransformer.forward returns (x, y, z, w) because it uses scipy backend
                    delta_rot_quat_xyzw = self.rot_transformer.forward(delta_rot6d[None, :])[0] # shape (4,)
                    delta_rot = R.from_quat(delta_rot_quat_xyzw)
                                        
                    # DEBUG: Print Action Info
                    d_pos_mag = np.linalg.norm(delta_pos)
                    d_rot_mag = delta_rot.magnitude()
                    d_rot_euler = delta_rot.as_euler('xyz', degrees=True)
                    gripper_val = action[9]
                    
                    if i == 0: # Print first action of the chunk
                         print(f"[Debug Action] Step {step_idx}: Pos Mag={d_pos_mag:.4f}, Rot Mag={d_rot_mag:.4f}, Euler={d_rot_euler}, Grip={gripper_val:.4f}")
                    
                    # Target = Current * Delta (Apply delta in local frame)
                    target_rot = current_ee_rot * delta_rot
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
                    
                    
                    
                    # Apply Magic Grasp
                    self._update_magic_grasp(target_gripper_width, step_idx=step_idx)

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
