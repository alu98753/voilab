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

## 3. 建議工具與參考檔案
*   **Zarr 檢查**: `inspect_zarr.py`
*   **位姿定義**: `scripts/generate_data.py`, `umi/common/pose_util.py`
*   **資料載入邏輯**: `diffusion_policy/dataset/umi_dataset.py`
*   **座標轉換定義**: `diffusion_policy/common/pose_repr_util.py`
