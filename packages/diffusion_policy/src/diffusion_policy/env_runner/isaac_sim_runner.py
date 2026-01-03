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
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
import omni.kit.app

class IsaacSimRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir: str,
        n_episodes: int = 1,
        max_steps_per_episode: int = 200,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        urdf_path: str = None, # Set dynamically
        usd_path: str = None,  # Set dynamically
        headless: bool = True,
        save_video: bool = True,
        save_observation_data: bool = False,
        use_recorded_poses: bool = True,
        object_poses_path: Optional[str] = None,
        episode_indices: Optional[List[int]] = None,
        dataset_episode_indices: Optional[List[int]] = None,
        validation_dataset: Optional[Any] = None,
        replay_gt: bool = False,
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
        self.replay_gt = replay_gt
        # Force video recording if GT Replay is enabled, as visual verification is critical
        if self.replay_gt:
            self.save_video = True
        else:
            self.save_video = save_video
        self.save_observation_data = save_observation_data
        self.use_recorded_poses = use_recorded_poses
        self.object_poses_path = object_poses_path
        self.episode_indices = episode_indices
        self.dataset_episode_indices = dataset_episode_indices
        self.validation_dataset = validation_dataset
        self.replay_gt = replay_gt

        # Detect project root
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
        print(f"[IsaacSimRunner] Project Root detected: {self.project_root}")

        if self.urdf_path is None:
            self.urdf_path = os.path.join(self.project_root, "assets/franka_panda/franka_panda_umi-isaacsim.urdf")
        if self.usd_path is None:
            self.usd_path = os.path.join(self.project_root, "assets/ED305_scene/ED305.usd")

        # Robot config
        self.franka_panda_usd = os.path.join(self.project_root, "assets/franka_panda/franka_panda_arm.usd")
        self.franka_prim_path = "/World/Franka"
        self.lula_robot_description_path = os.path.join(self.project_root, "assets/lula/frank_umi_descriptor.yaml")
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
        
        # FIX: Update App to ensure extensions are loaded before Stage access
        omni.kit.app.get_app().update()

        # Open stage
        print(f"[Debug] Opening stage: {self.usd_path}")
        stage_utils.open_stage(self.usd_path)
        
        # FIX: Update App to ensure Stage is loaded before World creation
        omni.kit.app.get_app().update()
        
        print("[Debug] Stage opened. Creating World...")
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # Add robot
        print(f"[Debug] Adding robot from: {self.franka_panda_usd}")
        robot_prim = stage_utils.add_reference_to_stage(usd_path=self.franka_panda_usd, prim_path=self.franka_prim_path)
        
        # FIX: Select AlternateFinger and Quality variants to match generate_data.py
        # This ensures the physical structure and TCP offsets match the UMI dataset.
        robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
        robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")
        
        if self.registry_config:
            # Poses are now applied in _reset_robot_pose() after World.reset()
            # This ensures they persist across episode restarts
            # 2. Load Preload Objects
            env_vars = self.registry_config.get("environment_vars", {})
            preload_objects = env_vars.get("PRELOAD_OBJECTS", [])
            ASSETS_DIR = os.path.join(self.project_root, "assets/CADs")
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
        
            # 3. Camera Pose: 
            # REMOVED Fixed Camera Pose application. 
            # The camera is attached to the robot wrist (GoPro), so it moves with the robot.
            # We should NOT force it to a fixed world pose.
        
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
            prim_path=self.camera_prim_path,
            name="eval_camera",
            resolution=(224, 224) # Match updated generate_data.py (Square)
        )
        self.camera.initialize()
        
        # Verify Resolution and Aperture
        res = self.camera.get_resolution()
        h_ap = self.camera.get_horizontal_aperture()
        v_ap = self.camera.get_vertical_aperture()
        print(f"[IsaacSimRunner] Camera Configured: Resolution={res}, HorizAperture={h_ap:.4f}, VertAperture={v_ap:.4f}")
        
        # Setup Front Monitoring Camera (Match generate_data.py)
        print("[Debug] Setting up front monitoring camera...")
        stage_units = stage_utils.get_stage_units()
        # Default franka_translation from registry or hardcoded
        franka_pose = self.registry_config.get("franka_pose", {}) if self.registry_config else {}
        franka_trans = franka_pose.get("translation", [4.5, 2.7, 0.9])
        robot_pos = np.array(franka_trans) / stage_units
        front_camera_pos = robot_pos + (np.array([6.5, 0.0, 1.75]) / stage_units)
        
        # Hardcoded Euler Angles to avoid Matrix Roll issues
        euler_angles = [180, 165, 0] # [x, y, z]
        rot = R.from_euler('xyz', euler_angles, degrees=True)
        rot_quat_xyzw = rot.as_quat()
        rot_quat_wxyz = np.array([rot_quat_xyzw[3], rot_quat_xyzw[0], rot_quat_xyzw[1], rot_quat_xyzw[2]])
        
        self.fixed_camera_front = Camera(
            prim_path="/World/FixedCameraFront",
            name="fixed_camera_front",
            position=front_camera_pos,
            orientation=rot_quat_wxyz,
            resolution=(1280, 720)
        )
        self.fixed_camera_front.initialize()
        print(f"[IsaacSimRunner] Front Camera Configured at {front_camera_pos}")

        print("[Debug] Resetting world...")
        self.world.reset()
        logger.info("[IsaacSimRunner] Setup complete")
        print("[Debug] IsaacSimRunner: Setup complete")

    def set_registry_config(self, config: Dict):
        self.registry_config = config
        print("[Debug] IsaacSimRunner: Registry config set.")

    def _reset_robot_pose(self):
        """Applies registry-defined robot pose. Must be called after world.reset()."""
        if self.registry_config and self.panda:
            franka_pose = self.registry_config.get("franka_pose", {})
            trans = franka_pose.get("translation")
            rot = franka_pose.get("rotation_quat")
            if trans is not None and rot is not None:
                stage_units = stage_utils.get_stage_units()
                target_trans = np.array(trans) / stage_units
                print(f"[IsaacSimRunner] Resetting Robot Base Pose to: {target_trans}")
                
                # Use the XFormPrim to move the base
                robot_xform = SingleXFormPrim(prim_path=self.franka_prim_path)
                robot_xform.set_local_pose(
                    translation=target_trans,
                    orientation=np.array(rot)
                )
                
                # FORCE update of prim to ensure solver sees it
                omni.kit.app.get_app().update()

                # Also update Lula solver to match new base!
                self.lula_solver.set_robot_base_pose(
                    robot_position=target_trans,
                    robot_orientation=np.array(rot)
                )

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

    def _normalize_object_name(self, name: str) -> str:
        return name.strip().lower().replace(" ", "_")

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
        
        # Pre-calculate episode to sampler indices mapping for MSE calculation
        episode_to_sampler_indices = {}
        if self.validation_dataset is not None:
            import bisect
            dataset = self.validation_dataset
            episode_ends = dataset.replay_buffer.episode_ends[:]
            for i in range(len(dataset)):
                start_ptr, end_ptr, _, _ = dataset.sampler.indices[i]
                # Find which episode this segment belongs to
                ep_idx = bisect.bisect_right(episode_ends, start_ptr)
                if ep_idx not in episode_to_sampler_indices:
                    ep_start = episode_ends[ep_idx-1] if ep_idx > 0 else 0
                    ep_end = episode_ends[ep_idx]
                    episode_to_sampler_indices[ep_idx] = {
                        'sampler_indices': [],
                        'rb_range': (ep_start, ep_end)
                    }
                episode_to_sampler_indices[ep_idx]['sampler_indices'].append(i)
            print(f"[IsaacSimRunner] Pre-indexed {len(episode_to_sampler_indices)} episodes from validation dataset for MSE calculation")
        
        # Pre-load object poses JSON to match indices correctly
        object_poses_data = []
        if self.use_recorded_poses and self.object_poses_path and os.path.exists(self.object_poses_path):
            try:
                import json
                with open(self.object_poses_path, 'r') as f:
                    object_poses_data = json.load(f)
                    if not isinstance(object_poses_data, list):
                        object_poses_data = [object_poses_data]
            except Exception as e:
                print(f"[IsaacSimRunner] WARNING: Failed to load object poses JSON: {e}")
        if self.replay_gt and self.validation_dataset is None:
            raise ValueError("replay_gt=True requires validation_dataset to be passed to IsaacSimRunner.")

        all_episode_stats = []
        
        
        # Determine episodes to run
        if self.episode_indices is not None:
            ep_indices = self.episode_indices
            if self.dataset_episode_indices is not None:
                dataset_ep_indices = self.dataset_episode_indices
            else:
                dataset_ep_indices = ep_indices # Fallback
        else:
            ep_indices = range(self.n_episodes)
            dataset_ep_indices = ep_indices
        
        for i, (episode_idx, dataset_ep_idx) in enumerate(zip(ep_indices, dataset_ep_indices)):
            logger.info(f"[IsaacSimRunner] Starting episode {i+1}/{len(ep_indices)} (sim_idx: {episode_idx}, dataset_idx: {dataset_ep_idx})")
            self.world.reset()
            # FIX: Re-apply Robot Base Pose after world reset!
            self._reset_robot_pose()
            
            # Render once to get valid camera data
            self.world.step(render=True)
            self.start_eef_rot_quat = None
            policy.reset()
            print(f"[Debug] Episode {episode_idx+1}: World reset & Robot pose applied.")


            # --- 1. Reset Objects (Match generate_data.py Phase 1) ---
            if hasattr(self, 'object_prims') and self.object_prims:
                # OPTION A: Load recorded poses from JSON (Default)
                if self.use_recorded_poses and self.object_poses_path and os.path.exists(self.object_poses_path):
                    try:
                        from object_loader import load_object_transforms_from_json
                        
                        # CORRECT INDEX MAPPING: Find the JSON entry that matches our RB frame range
                        json_idx = episode_idx # Fallback
                        if dataset_ep_idx in episode_to_sampler_indices:
                            rb_start = episode_to_sampler_indices[dataset_ep_idx]['rb_range'][0]
                            for j, entry in enumerate(object_poses_data):
                                ep_range = entry.get('episode_range', [0, 0])
                                if ep_range[0] <= rb_start < ep_range[1]:
                                    json_idx = j
                                    print(f"[IsaacSimRunner] Map dataset_idx {dataset_ep_idx} (start frame {rb_start}) -> JSON index {json_idx}")
                                    break

                        object_transforms = load_object_transforms_from_json(
                            self.object_poses_path,
                            episode_index=json_idx,
                            aruco_tag_pose=self.registry_config.get("aruco_tag_pose") if self.registry_config else None,
                            cfg=self.registry_config,
                        )
                        
                        if len(object_transforms) > 0:
                            print(f"[IsaacSimRunner] Loaded {len(object_transforms)} object transforms from JSON index {json_idx} for sim_episode {episode_idx}")
                            print(f"[IsaacSimRunner] Loading recorded poses for sim_episode {episode_idx}")
                            for obj in object_transforms:
                                obj_name = self._normalize_object_name(obj["object_name"])
                                if obj_name in self.object_prims:
                                    obj_pos = np.array(obj["position"], dtype=np.float64)
                                    # FIX: Only apply position to match generate_data.py behavior (keeps cups upright)
                                    self.object_prims[obj_name].set_world_pose(position=obj_pos)
                                    print(f"[IsaacSimRunner] Positioned {obj_name} at {obj_pos} from recorded data (orientation maintained)")
                                else:
                                    # Try a more fuzzy match if needed (e.g. "cup" vs "pink_cup")
                                    matched = False
                                    for prim_name in self.object_prims.keys():
                                        if self._normalize_object_name(prim_name) == obj_name or \
                                           obj_name in self._normalize_object_name(prim_name) or \
                                           self._normalize_object_name(prim_name) in obj_name:
                                            obj_pos = np.array(obj["position"], dtype=np.float64)
                                            # FIX: Only apply position to match generate_data.py behavior (keeps cups upright)
                                            self.object_prims[prim_name].set_world_pose(position=obj_pos)
                                            print(f"[IsaacSimRunner] Positioned {prim_name} at {obj_pos} (fuzzy match with {obj_name}, orientation maintained)")
                                            matched = True
                                            break
                                    if not matched:
                                        print(f"[IsaacSimRunner] WARNING: Object {obj_name} from JSON not found in scene")
                        else:
                            print(f"[IsaacSimRunner] WARNING: No transforms found for episode {episode_idx} in JSON")
                    except Exception as e:
                        print(f"[IsaacSimRunner] ERROR loading recorded poses: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to random mode if JSON loading fails? Or just continue?
                
                # OPTION B: Random Jitter (Fallback or Explicitly requested)
                else:
                    print(f"[IsaacSimRunner] Using random jitter mode for episode {episode_idx}")
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
                # IMPROVEMENT: Use render=True and sleep to match generate_data.py for higher stability
                self.world.step(render=True)
                time.sleep(1 / 60)
            print("[Debug] Physics settled.")

            # --- 3. Initialize Robot Pose (Match generate_data.py Phase 3) ---
            # Calibrate Base
            base_pos, base_quat = self.panda.get_world_pose()
            self.lula_solver.set_robot_base_pose(robot_position=base_pos, robot_orientation=base_quat)
            
            # Initialize Robot Joint Positions
            if self.replay_gt and self.validation_dataset is not None:
                # GT Replay Mode: Initialize to exact start pose from dataset
                if dataset_ep_idx in episode_to_sampler_indices:
                    info = episode_to_sampler_indices[dataset_ep_idx]
                    sampler_indices = info['sampler_indices']
                    rb_start, rb_end = info['rb_range']
                    
                    if len(sampler_indices) > 0:
                        start_sampler_idx = sampler_indices[0]
                        batch = self.validation_dataset[start_sampler_idx]
                        
                        # Get robot0_demo_start_pose
                        # Shape: (1, 6) or (6,) depending on batching (Cartesian: X, Y, Z, Ax, Ay, Az)
                        start_pose_6d = batch.get('robot0_demo_start_pose')
                        
                        # Fallback: UmiDataset.__getitem__ DELETES _demo_start_pose from obs_dict!
                        # We must fetch it directly from the replay_buffer if missing in batch.
                        if start_pose_6d is None and hasattr(self.validation_dataset, 'replay_buffer'):
                            print(f"[IsaacSimRunner] robot0_demo_start_pose missing in batch. Fetching from ReplayBuffer...")
                            try:
                                rb = self.validation_dataset.replay_buffer
                                if 'robot0_demo_start_pose' in rb:
                                    # Fetch at specific frame index `rb_start`
                                    start_pose_6d = rb['robot0_demo_start_pose'][rb_start]
                                    if isinstance(start_pose_6d, np.ndarray):
                                        pass # already numpy
                                    elif isinstance(start_pose_6d, torch.Tensor):
                                        start_pose_6d = start_pose_6d.cpu().numpy()
                            except Exception as e:
                                print(f"[IsaacSimRunner] FAILED to fetch from ReplayBuffer: {e}")

                        
                        if start_pose_6d is not None:
                            if isinstance(start_pose_6d, torch.Tensor):
                                start_pose_6d = start_pose_6d.cpu().numpy()

                            
                            # Handle batch dim if present
                            if start_pose_6d.ndim > 1:
                                start_pose_6d = start_pose_6d[0]
                                
                            print(f"[IsaacSimRunner] GT Replay: Dataset Start Pose (6D): {start_pose_6d}")
                            
                            # Parse 6D Pose
                            start_pos = start_pose_6d[:3]
                            start_rot_axis_angle = start_pose_6d[3:]
                            
                            # Convert Axis-Angle to Quaternion (WXYZ for Solver)
                            start_rot_quat_xyzw = R.from_rotvec(start_rot_axis_angle).as_quat()
                            start_rot_quat_wxyz = start_rot_quat_xyzw[[3, 0, 1, 2]]
                            
                            print(f"[IsaacSimRunner] IK Target - Pos: {start_pos}, Rot(WXYZ): {start_rot_quat_wxyz}")
                            
                            # Compute IK
                            ik_action, success = self.art_kine_solver.compute_inverse_kinematics(
                                target_position=start_pos,
                                target_orientation=start_rot_quat_wxyz
                            )
                            
                            if success:
                                print(f"[IsaacSimRunner] IK Success. Setting joint positions...")
                                self.panda.set_joint_positions(ik_action.joint_positions, np.arange(7))
                                # Debug: check if set properly
                                jp_now = self.panda.get_joint_positions()
                                print(f"[IsaacSimRunner] Joints after Set: {jp_now}")
                            else:
                                print(f"[IsaacSimRunner] CRITICAL WARNING: IK Failed for Dataset Start Pose!")
                            
                            # Update solvers with new state
                            # Removing explicit step(render=True) to avoid potential crash
                            # self.world.step(render=True) 
                            
                        else:
                            print(f"[IsaacSimRunner] WARNING: robot0_demo_start_pose not found in dataset for episode {dataset_ep_idx}")
                else:
                    print(f"[IsaacSimRunner] WARNING: No sampler indices found for episode {dataset_ep_idx}")
            
            else:
                # Inference Mode: Use Registry Config or Default
                pass
            
            # Short Settle
            print("[Debug] Warming up renderer (20 steps)...")
            for i in range(20):
                self.world.step(render=True)
            print("[Debug] Renderer warmed up.")
            
            # Check Post-Settle Pose
            ee_pos_final, _ = self.art_kine_solver.compute_end_effector_pose()
            print(f"[IsaacSimRunner] Debug: EE Pose AFTER Settle: {ee_pos_final}")
            # --------------------------------------------------------
            
            obs_buffer = collections.deque(maxlen=self.n_obs_steps)
            # Warm up
            for _ in range(self.n_obs_steps):
                obs_buffer.append(self.get_obs())
            
            video_frames = []
            video_frames_front = []
            is_success = False
            done = False
            # --- 6. GT Replay Pre-fetch (Optimization) ---
            gt_abs_poses = None
            gt_gripper_widths = None
            if self.replay_gt and hasattr(self.validation_dataset, 'replay_buffer'):
                if dataset_ep_idx in episode_to_sampler_indices:
                    info = episode_to_sampler_indices[dataset_ep_idx]
                    start_idx, end_idx = info['rb_range']
                    
                    print(f"[IsaacSimRunner] GT Replay: Replaying full episode for dataset_idx {dataset_ep_idx}")
                    rb = self.validation_dataset.replay_buffer
                    print(f"[IsaacSimRunner] GT Replay: Pre-fetching {end_idx - start_idx} absolute poses (RB indices {start_idx} to {end_idx})...")
                        
                    try:
                        # Note: SequenceSampler indices for an episode are usually contiguous in ReplayBuffer
                        # We can slice them or iterate. Slicing is faster.
                        gt_abs_pos = rb['robot0_eef_pos'][start_idx:end_idx]
                        gt_abs_rot = rb['robot0_eef_rot_axis_angle'][start_idx:end_idx]
                        gt_abs_gripper = rb['robot0_gripper_width'][start_idx:end_idx]
                        
                        gt_abs_poses = (gt_abs_pos, gt_abs_rot)
                        gt_gripper_widths = gt_abs_gripper
                    except Exception as e:
                        print(f"[IsaacSimRunner] WARNING: Failed to pre-fetch absolute poses: {e}")

            # GT Replay Limit: Ignore max_steps_per_episode, use actual length
            step_limit = len(gt_abs_poses[0]) if (self.replay_gt and gt_abs_poses is not None) else self.max_steps_per_episode
            
            step_idx = 0
            while not done and step_idx < step_limit:
                # --- 7. GT Replay Optimized Execution ---
                if self.replay_gt and gt_abs_poses is not None:
                    if step_idx < len(gt_abs_poses[0]):
                        target_pos = gt_abs_poses[0][step_idx]
                        target_rot_axis_angle = gt_abs_poses[1][step_idx]
                        target_gripper_width = gt_gripper_widths[step_idx][0]
                        
                        target_rot_quat_xyzw = R.from_rotvec(target_rot_axis_angle).as_quat()
                        target_rot_quat_wxyz = target_rot_quat_xyzw[[3, 0, 1, 2]]
                        
                        ik_action, success = self.art_kine_solver.compute_inverse_kinematics(
                            target_position=target_pos,
                            target_orientation=target_rot_quat_wxyz
                        )
                        
                        if success:
                            self.panda.set_joint_positions(ik_action.joint_positions, np.arange(7))
                            g_pos = target_gripper_width / 2.0
                            self.panda.gripper.set_joint_positions(np.array([g_pos, g_pos]))
                            
                            # Debug: Verify reachability every 10 steps
                            if step_idx % 10 == 0:
                                curr_ee_pos, _ = self.art_kine_solver.compute_end_effector_pose()
                                dist = np.linalg.norm(curr_ee_pos - target_pos)
                                if dist > 0.02:
                                    print(f"[Warning] GT Replay Step {step_idx}: Actual EE pos {curr_ee_pos} is {dist:.4f}m away from target {target_pos}")
                                else:
                                    print(f"[Debug] GT Replay Step {step_idx}: Reached {curr_ee_pos} (Target {target_pos})")
                            
                            self._update_magic_grasp(target_gripper_width, step_idx=step_idx)
                        else:
                            if step_idx % 50 == 0:
                                print(f"[IsaacSimRunner] GT Replay: IK Failed at step {step_idx}. Target: {target_pos}")

                        self.world.step(render=True)
                        if self.save_video:
                            frame = self.camera.get_rgb()
                            if frame is not None: video_frames.append(frame)
                            frame_front = self.fixed_camera_front.get_rgb()
                            if frame_front is not None: video_frames_front.append(frame_front)
                        
                        step_idx += 1
                        continue # Skip standard policy processing
                    else:
                        break

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

                
                # policy execution
                with torch.no_grad():
                    action_dict = policy.predict_action(batch_obs)
                actions = action_dict['action'][0].cpu().numpy() # [horizon, 10]
                exec_steps = self.n_action_steps
                
                # Execute action (n_action_steps or 1 for legacy GT)
                # ... [Rest of the standard action execution loop below] ...
                
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
                
                # Save front video
                if video_frames_front:
                    video_path_front = os.path.join(self.output_dir, f"eval_ep_{episode_idx}_front.mp4")
                    height, width, _ = video_frames_front[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_front = cv2.VideoWriter(video_path_front, fourcc, 30.0, (width, height))
                    for f in video_frames_front:
                        out_front.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                    out_front.release()
                    logger.info(f"[IsaacSimRunner] Saved front video to {video_path_front}")

            # Calculate Dataset MSE for this episode
            episode_mse = {}
            if self.validation_dataset is not None and dataset_ep_idx in episode_to_sampler_indices:
                info = episode_to_sampler_indices[dataset_ep_idx]
                sampler_indices = info['sampler_indices']
                from torch.utils.data import Subset, DataLoader
                import torch.nn.functional as F
                
                ep_subset = Subset(self.validation_dataset, sampler_indices)
                # Use 0 workers for stability within Isaac Sim environment
                ep_loader = DataLoader(ep_subset, batch_size=32, num_workers=0)
                
                total_mse_stats = collections.defaultdict(float)
                count = 0
                
                for batch in ep_loader:
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    with torch.no_grad():
                        # Predict action (open-loop)
                        pred_action = policy.predict_action(batch['obs'])['action']
                        gt_action = batch['action']
                        
                        # Calculate MSE details
                        B, T, D = pred_action.shape
                        pred_action = pred_action.view(B, T, -1, 10)
                        gt_action = gt_action.view(B, T, -1, 10)
                        
                        mse = F.mse_loss(pred_action, gt_action, reduction='none')
                        # Sum over all dims except batch
                        total_mse_stats['mse'] += mse.mean().item() * B
                        total_mse_stats['mse_pos'] += mse[..., :3].mean().item() * B
                        total_mse_stats['mse_rot'] += mse[..., 3:9].mean().item() * B
                        total_mse_stats['mse_width'] += mse[..., 9].mean().item() * B
                        count += B
                
                if count > 0:
                    for k in total_mse_stats:
                        episode_mse[k] = total_mse_stats[k] / count
                    
                    print(f"[IsaacSimRunner] Episode {i+1} (idx {episode_idx}) Dataset MSE: {episode_mse['mse']:.6f} "
                          f"(pos: {episode_mse['mse_pos']:.6f}, rot: {episode_mse['mse_rot']:.6f}, width: {episode_mse['mse_width']:.6f})")

            all_episode_stats.append({
                'episode_idx': episode_idx,
                'dataset_episode_idx': dataset_ep_idx,
                'episode_length': step_idx,
                'success': is_success,
                'dataset_mse': episode_mse
            })

        print("[IsaacSimRunner] Evaluation complete.")
        success_rate = np.mean([s['success'] for s in all_episode_stats])
        avg_length = np.mean([s['episode_length'] for s in all_episode_stats])
        
        return {
            'episode_stats': all_episode_stats,
            'success_rate': success_rate,
            'avg_episode_length': avg_length
        }
