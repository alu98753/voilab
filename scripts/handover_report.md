# Isaac Sim Evaluation 除錯階段性報告 (Handover Report)

本報告記錄了針對 `isaac_sim_runner.py` 進行評估測試時的除錯發現、解決方案以及尚未解決的核心問題，供後續開發參考。

---

## 1. 已解決的問題與發現 (Solved Issues & Findings)

### A. 機器人初始化 (Robot Initialization)
*   **問題**: 機器人開始時處於預設姿勢，而非任務準備姿勢，導致視角不對且 IK 容易失敗。
*   **發現**: 參考 `scripts/generate_data.py` (Line 875-900)，發現資料生成時有顯式的初始化位姿設定。
*   **解決**: 在 `IsaacSimRunner.run()` 中加入了初始化邏輯：
    -   實施 `calibrate_robot_base` 校準。
    -   設定初始 EE (End-Effector) 為 `INIT_EE_POS` 加上偏移 `[-0.16, 0., 0.13]`。
    -   設定初始 Orientation (WXYZ): `[0.0081739, -0.9366365, 0.350194, 0.0030561]`。

### B. 物體設定與穩定性 (Object Setup & Stability)
*   **發現**: 
    -   `simulation_dataset.zarr.zip` 中未記錄物體初始位置 (經 `inspect_zarr.py` 核實)。
    -   杯子生成高度過低會與桌面碰撞彈飛；若無指定旋轉則會保持倒地狀態。
*   **解決**:
    -   **立起機制**: 強制設定杯子 Orientation 為 Identity Quaternion `[1, 0, 0, 0]`，確保正立。
    -   **生成優化**: 將生成高度提至 `Z=1.0` (略高於桌面)，使其自然落下。
    -   **隨機化**: 實作了以 `episode_idx` 為種子的領域隨機化 (Domain Randomization)，為杯子 XY 座標加入 `+/- 5cm` 的隨機偏差，增加評估多樣性。

### C. 觀察空間格式適配 (Observation Format Adapter)
*   **問題**: Simulator 輸出之旋轉為 **3D Axis-Angle**，但 Policy 預期為 **6D Rotation**。
*   **發現**: `umi_dataset.py` 在訓練讀取環節會將資料從 3D 轉為 6D，但模擬器原生輸出為 3D。
*   **解決**: 實作了 **Observation Adapter**，在餵給 Policy 前將 Axis-Angle 轉為 6D 格式。

---

### D. 核心問題解決：相機/手臂 180 度抖動 (Solved: 180-deg Flip & Jitter)
*   **問題**: 機器人動作與相機畫面出現劇烈的 180 度來回翻轉與抖動，導致無法順利接近目標物。
*   **根本原因 (Root Cause)**: 四元數分量順序混淆 (Quaternion Component Mismatch)。
    -   `isac_sim_runner.py` 錯誤地假設 `RotationTransformer` 使用 `(w, x, y, z)` 格式。
    -   實際上 `RotationTransformer` 基於 `scipy.spatial.transform.Rotation`，其預設格式為 `(x, y, z, w)`。
    -   程式碼中手動進行了 `[3, 0, 1, 2]` 與 `[1, 2, 3, 0]` 的交換操作，導致 **Identity Rotation (單位旋轉)** `[0, 0, 0, 1]` (xyzw) 被錯誤轉換為 `[0, 0, 1, 0]`，即繞 Z 軸旋轉 180 度。
*   **解決方案**:
    -   修正 `isaac_sim_runner.py` 中的 Observation Encoding 與 Action Decoding 邏輯。
    -   移除所有手動的四元數分量交換操作，直接使用 `RotationTransformer` 的輸出 (預設即為正確的 `xyzw`)。
    -   驗證後，機器人動作平滑，相機視角正常。


---


---

## 3. 資料與 IO 規格分析 (Data & IO Specification Analysis)

經查閱 `scripts/generate_data.py` (資料生成) 與 `umi_dataset.py` (訓練資料處理)，整理規格如下：

### A. 資料生成 (Data Generation) - `generate_data.py` / Zarr
資料以 **Global World Frame** (或 Robot Base Frame) 儲存，未經相對化處理。
*   `robot0_eef_pos`: shape `[T, 3]`. Absolute Position (Meters).
*   `robot0_eef_rot_axis_angle`: shape `[T, 3]`. Absolute Rotation (Axis-Angle).
*   `robot0_demo_start_pose`: shape `[T, 6]`. Episode Start Pose (Pos + Axis-Angle).

### B. 模型訓練與推論 (Training & Inference) - `UmiDataset` / `IsaacSimRunner`
訓練時 (`UmiDataset`) 會根據 Config (`umi.yaml`) 將資料轉換為 **Relative Frame (相對於當前時間步)**。

*   **Observation**:
    *   **Frame**: 相對於 **Current End-Effector Frame** (最後一個觀測步)。
    *   **Position**: $P_{rel} = R_{current}^{-1} (P_{world} - P_{current})$。
    *   **Rotation**: $R_{rel} = R_{current}^{-1} R_{world}$ (6D Representation)。
    *   **Wrt Start**: $R_{wrt\_start} = R_{start}^{-1} R_{current}$ (Runner 之前錯誤實作為 $R_{current} R_{start}^{-1}$)。
    
*   **Action**:
    *   **Frame**: 相對於 **Current End-Effector Frame**。
    *   **Output**: Target Pose in Local Frame.
    *   **Runner Decoding**: $T_{target} = T_{current} \times T_{action}$ (矩陣乘法).

---

## 2. 剩餘核心問題：手臂持續旋轉無法接近物體 (Unresolved: Continuous Arm Rotation without Approaching)
**(與 IO 分析結果一致，確認為計算順序錯誤)**

**當前狀態**: 修正 180 度翻轉後，機器人不再劇烈翻轉，但出現「手臂持續旋轉」且「無法接近杯子」的現象。

### 核心診斷 (Root Cause Verification)
**wrt_start 計算順序錯誤 (Matrix Multiplication Order)**:
*   **訓練邏輯 (`pose_repr_util.py`)**: `pose_rep='relative'` 執行 `inv(base) @ pose`。
    *   對應旋轉: $R_{relative} = R_{base}^{-1} R_{target}$。
*   **Runner 錯誤邏輯 (`isaac_sim_runner.py`)**:
    *   原程式碼: `rel_rot = curr_rot * start_rot.inv()`。
    *   Scipy 意義: $R_{current}(R_{start}^{-1})$，即先反轉 Start 再應用 Current (順序完全錯誤)。
    *   應修正為: `start_rot.inv() * curr_rot`。

### 驗證結論
Runner 計算的 `wrt_start` 特徵值在幾何意義上是錯誤的，導致 Policy 接收到無意義的 "Start-Relative" 訊號，這是造成手臂異常旋轉的直接原因。



## 3. 建議工具與參考檔案
*   **Zarr 檢查**: `inspect_zarr.py`
*   **位姿定義**: `scripts/generate_data.py`, `umi/common/pose_util.py`
*   **資料載入邏輯**: `diffusion_policy/dataset/umi_dataset.py`
*   **座標轉換定義**: `diffusion_policy/common/pose_repr_util.py`

## 4. 最新進展：相機解析度修復與後續挑戰 (Camera Resolution Fix & Remaining Challenges)

### A. 相機解析度不匹配修復 (Camera Resolution Mismatch Fixed)
*   **問題**: 機器人動作不再翻轉，但出現「抓空氣」或位置偏差，導致抓取失敗。
*   **發現**:
    *   **訓練資料 (Training Data)**: 原始影像解析度為 **1280x720 (16:9)**，在訓練前被 **非等比縮放 (Squash)** 到 **224x224 (1:1)**。這導致影像內容被「水平壓縮」，物體看起來較為細長。
    *   **評估環境 (Runner)**: 原本 `isaac_sim_runner.py` 直接渲染 **224x224**。這導致 Policy 看到的是正常比例 (較胖) 的物體，與它學到的特徵空間 (Feature Space) 不匹配，造成空間感知誤差。
*   **解決**: 修改 `isaac_sim_runner.py`：
    1.  將相機解析度改回 **1280x720**。
    2.  在取圖後使用 `cv2.resize` 強制縮放至 **224x224**，模擬訓練資料的視覺特徵。
*   **成效**: 修正後，Policy 應能正確感知物體位置。

### B. "抓空氣" 問題診斷：TA 的資料生成作弊 (Analysis of "Grasping Air")
修復相機後，機器人能正確移動到杯子附近並嘗試閉合夾爪，但經常「夾空」或無法提起杯子。經深入分析助教的程式碼，發現這是由於資料生成方式與評估方式的根本差異所致。

*   **TA 的資料生成方法 (`generate_data.py` + `motion_plan.py`)**:
    *   **非物理抓取 (Magic Grasp)**: 助教在生成資料時 **並未使用物理抓取**。
    *   **瞬間移動機制**: 當夾爪閉合時，程式會計算並鎖定杯子相對於夾爪的相對位置。隨後每一幀，程式都將杯子 **直接 Teleport (瞬間移動)** 到夾爪目標位置。
    *   **影響**: 訓練資料中的杯子是「完美跟隨」夾爪的，完全無視物理摩擦力或滑動。模型學到的是「只要閉合夾爪，杯子就會跟著動」。

*   **當前評估問題**:
    *   **物理抓取失敗**: 目前 `isaac_sim_runner.py` 嘗試使用真實物理引擎 (Friction) 來抓取。
    *   **控制模式衝突**: 我們使用 `set_joint_positions` (Teleport) 來控制機器人手臂 (為了重現軌跡)。當一個 Teleport 的手臂試圖物理抓取一個動態剛體時，物理引擎極易發生穿透 (Interpenetration) 或彈出，導致抓取失敗。

* English version:
TA Method Analysis & Evaluation Strategy
1. Analysis of TA's Generation Method
The TA's data generation (
generate_data.py
 + 
motion_plan.py
) uses a "Magic Grasp" (Teleportation Attachment) technique, not physics-based grasping.

Mechanism Details
In 
scripts/motion_plan.py
:

Detection: The script checks if the gripper is in the "close" phase (line 187).
Locking: At a specific moment (counter == 20), it calculates the relative transform T_ee_to_obj between the End-Effector and the Object (line 202) and sets self.attached = True.
Synchronization: In every subsequent simulation step (
step
 function), it calls 
_sync_attached_object
 (line 142).
This function calculates where the object should be based on the robot's current position: New_Obj_Pos = Current_EE_Pos * T_ee_to_obj.
It then Teleports the object to that position using set_prim_world_pose.
Crucially: It bypasses physics collision/friction entirely for the grasped object.
Why This Matters
Training Data: The model (Diffusion Policy) sees images where the cup perfectly follows the gripper, even if the grasp would be physically unstable.
Current Evaluation: Your 
isaac_sim_runner.py
 is trying to use Physics Grasping (Friction) while the robot arm is being Teleported (set_joint_positions).
Teleporting the arm through a physics object often causes the object to pop out or be ignored (no persistent contact force).
This explains why the robot "grasps air" or fails to lift the cup—the physics engine cannot handle the "Teleport Grasp".
2. Recommended Evaluation Strategy
To fairly evaluate the model, we should replicate the TA's "Magic Grasp" in the evaluation environment. The model is trained to control the arm spatially; it assumes that "Close Gripper near Object" = "Object Attached". It did not learn to handle physics slippage.

Proposal: "Magic Grasp" for Evaluation
We will modify 
isaac_sim_runner.py
 to implement a dynamic attachment logic:

Monitor State: Check the Model's output action_gripper_width and the distance between the End-Effector and the Cup.
Attach Condition:
If action_gripper_width < 0.04 (Trying to close)
AND distance(EE, Cup) < 0.1 (Close enough)
AND Not Attached
ACTION: Save T_ee_to_obj and set attached_object = Cup. Disable Physics for Cup (optional, or just overwrite pose).
Update Loop:
If Attached: Teleport Cup to Current_EE_Pos * T_ee_to_obj.
Detach Condition:
If action_gripper_width > 0.04 (Trying to open)
ACTION: Set attached_object = None. Re-enable Physics (gravity takes over).
3. Implementation Plan
We will modify 
isaac_sim_runner.py
 to add this logic. This is cleaner and more reliable than trying to fix the physics of a teleporting robot.

Changes to 
isaac_sim_runner.py
:
Track Objects: Ensure self.object_prims contains the cups (Blue/Pink).
Add magical_grasp() function: Call this inside the simulation loop.
Use set_gripper_width helper: Use the gradual movement logic you provided (optional, but smoother visually).

### C. 建議解決方案：在評估中重現 "Magic Grasp" (Recommendation)
為了公平且有效地評估 Policy (該 Policy 並未學習物理抓取的細微操作)，我們應在 `isaac_sim_runner.py` 中重現助教的作弊邏輯。

**計畫**:
1.  **監控狀態**: 在每個 Simulation Step 檢查夾爪寬度 (Action) 與杯子距離。
2.  **觸發吸附**: 當 `Action < Threshold` (嘗試閉合) 且 `Distance < Threshold` (夠近) 時，觸發 **Attach**。
3.  **執行吸附**: 記錄相對位姿，並在每一幀強制將杯子 Teleport 到夾爪相對位置。
4.  **釋放**: 當 `Action > Threshold` (嘗試張開) 時，解除吸附，恢復物理模擬。

### 分析報告：為何不能直接使用 motion_plan.py

雖然 motion_plan.py 包含了助教的吸附邏輯，但我們 不能直接引用 (Import) 它來用於 isaac_sim_runner.py，原因如下：

控制權與架構不相容 (Control Flow Mismatch):
motion_plan.py (Rule-based): 它是一個「狀態機 (State Machine)」，負責 主動規劃 機器人的動作 (移動->下降->抓取->上升)。它知道自己何時在 "Close Phase"。
isaac_sim_runner.py (Policy-based): 它的動作是由神經網路 (Policy) 決定的。Runner 不知道現在是 "Close Phase" 還是只是路過。我們只能從 Policy 輸出的 Action (夾爪寬度) 來 推測 它的意圖。
結論: motion_plan.py 的邏輯是「我要抓了，所以我吸附」。Runner 的邏輯必須是「我看他像是要抓了，所以我幫他吸附」。兩者觸發機制本質不同。
邏輯移植可行性 (Logic Logic Portability):
其核心數學邏輯 (T_obj = T_ee @ T_ee_to_obj) 非常標準。
我剛剛為您實作的 _update_magic_grasp 已經完全複製 了助教的核心數學邏輯 (Matrix Multiplication 與 Teleportation)。
唯一的差異在於 觸發條件 (Trigger)：助教用固定的計時器 (counter == 20)，我們改用更通用的「距離與動作閾值」，這在 Policy 評估中是必須的改動。
建議: 目前的實作已經提取了 motion_plan.py 的精髓並適配了 Policy 評估的需求。直接引用反而會因為架構衝突導致程式碼極難維護。我們可以放心地使用目前的實作。

觸發條件的詳細差異分析

兩者的根本差異在於「全知視角」與「觀察者視角」的不同：

1. 助教的觸發 (TA's Trigger in motion_plan.py)
機制: 硬編碼計時器 (Hardcoded Timer)。
程式碼: if counter == 20: attached = True (在 Close 階段的第 20 幀)。
原理: 因為助教的腳本是用寫死的軌跡規劃（先移動到 A，再下降到 B，再閉合），程式完全知道 在第 20 幀的時候，夾爪一定剛好包住杯子。
特點: 依賴於「預設軌跡的完美執行」。
2. 我們的觸發 (Our Trigger in isaac_sim_runner.py)
機制: 狀態依賴觸發 (State-Dependent Trigger)。
程式碼: if action < 0.04 and distance < 0.15: attached = True。
原理: Runner 不知道 Policy 打算什麼時候抓。Policy 可能在第 50 幀抓，也可能在第 100 幀抓，甚至可能抓偏了。
我們必須 觀察 (Observe) Policy 的行為：「哦，你的夾爪正在變窄 (想要閉合)，而且這附近有個杯子 (距離夠近)」。
滿足這兩個條件，我們就判定為「抓取意圖成立」，並執行吸附。
特點: 適應性強，能應對 Policy 的各種決策時間點。
潛在影響 (Trade-off):

我們的判定可能比助教的更「寬容」。如果 Policy 在距離杯子 10cm 的地方就提早閉合夾爪 (原本應該抓空)，我們的邏輯 (dist < 15cm) 會主動把杯子「吸過來」。
優點: 能有效解決 Sim 物理引擎的接觸不穩定問題。
缺點: 可能會讓一些「稍微抓歪」的 Policy 也能判定成功 (False Positive)。
調整建議: 如果您希望評估更嚴格，我們可以將 DIST_THRESHOLD = 0.15 (15cm) 改小，例如 0.05 (5cm)，強迫 Policy 必須非常精準地靠近物體才能觸發吸附。目前我也建議先用 0.15 測試，確認能抓到後再收緊。

5. 目前看起來視角是由上往下 並且 高度差不多  但是我的手臂看到杯子卻在旋轉 , 而training data即使杯子在遠方 也會平移過去 再往下抓取他 為何有這麼大的差異