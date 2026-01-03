# 環境對比分析報告：助教 (TA) vs. 我的平台 (User)

本報告詳細分析了 `launch-simulator` (助教資料生成環境) 與 `eval-model` (使用者評估環境) 之間的差異，旨在找出導致 GT Replay 失敗與機器人動作偏差的根源。

## 1. 核心技術指標對照表

| 特性 | 助教環境 (`launch-simulator`) | 使用者環境 (`eval-model`) | 影響描述 |
| :--- | :--- | :--- | :--- |
| **機器人變體** | `AlternateFinger` + `Quality` | 預設 (Default) | **重大差異**。變體決定了夾爪長度與 TCP 偏移，不一致會導致 IK 解算位置偏差約 5~10cm。 |
| **TCP 定義** | 使用 `umi_tcp` 座標系 | 使用 `umi_tcp` (但基於錯誤變體) | 即使名稱相同，若機器人變體不對，`umi_tcp` 的絕對座標會不同。 |
| **解算器校正** | 頻繁呼叫 `set_robot_base_pose` | 僅在初始化或週期性呼叫 | **重大差異**。若機器人基座有位移，解算器若未同步會導致輸入的 Target 位置被視為相對於 (0,0,0)。 |
| **動作空間 (Action)** | 儲存絕對 Pose (World Frame) | 使用相對 Pose (Relative Frame) | 評估時需將 Policy 預測的相對量疊加回正確的參考座標系 (Reference Frame)。 |
| **渲染穩定性** | 較為穩定 (存檔頻率較低) | 高頻崩潰 (Exit 139) | 使用者環境在 Headless 模式下呼叫 `get_rgb()` 極易觸發 `omni.syntheticdata` 崩潰。 |
| **物理沈降** | 100 steps (含 render) | 無或較短 | 決定了場景物件如杯子是否會「彈跳」或位置是否穩定。 |

---

## 2. 詳細差異分析

### 2.1 機器人變體 (Robot Variant)
助教在 `generate_data.py:569` 明確執行了：
```python
robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
robot.GetVariantSet("Mesh").SetVariantSelection("Quality")
```
這不僅是視覺上的差異。`AlternateFinger` 對應的是掛載了 GoPro 的夾爪模型，其運動學參數（如關節極限與連桿長度）與預設 Panda 夾爪不同。如果不切換，解算器（Lula）計算出的關節角度會讓夾爪去到錯誤的位置。

### 2.2 解算器基座校正 (Solver Base Calibration)
在 Isaac Sim 中，`LulaKinematicsSolver` 預設基座在原點。助教在每次計算前都會執行：
```python
lula_solver.set_robot_base_pose(robot_position, robot_orientation)
```
而評估端的 `IsaacSimRunner.py` 原本漏掉了這個步驟，或者在執行 Action 塊時未即時更新。這導致機器人雖然在場景中位於 `(4.5, 2.7, 0.9)`，但 solver 卻認為它在 `(0,0,0)`。

### 2.3 動作參考座標系 (Action Reference Frame)
*   **助教錄製端**: 將機器人的 World Pose 直接寫入 Zarr 的 `robot0_eef_pos`。
*   **使用者讀取端**: `UmiDataset` 會讀取這些絕對 Pose，並計算每一格與上一格（或觀察格）的 **相對位移 (Delta)**。
*   **評估執行端**: 預測出 Delta 後，必須精確地將其加回「目前執行塊的第一個觀測格」的絕對座標上。如果疊加的基礎座標 (Reference Base) 錯位，或是相對轉動的方向計算錯誤，機器人就會偏離軌跡。

### 2.4 Headless 模式與驅動問題
助教的 `launch-simulator` 在執行時通常較為穩定，而使用者在 Docker 中執行 `eval-model` 時，每當執行到 `camera.get_rgb()`，常會因為 `libomni.syntheticdata.plugin.so` 嘗試存取 GPU 緩衝區失敗而導致 **Segment Fault (139)**。這通常與 Docker 內部的 Vulkan 驅動映射或 Isaac Sim 對多相機渲染的處理有關。

---

## 3. 修復建議與結論

目前我已經針對 **2.1 (變體)** 與 **2.2 (校正)** 實施了修復，並優化了 **2.3 (相對邏輯)**。

這解釋了為什麼之前：
- 機器人「幾乎不動」：因為 Solver 以為基座在原點，想去的位置超出了運動範圍（IK Fail）。
- 軌跡「歪歪的」：因為夾爪型號不對，TCP 計算有偏差。
- 影片「沒錄到」：因為 Docker Volume 映射路徑使用了絕對路徑而非相對路徑。

**建議**: 後續應優先解決 139 崩潰問題（例如：檢查 `nvidia-container-runtime` 設置，或在 GUI 模式下執行），以獲取完整的成功率 (Success Rate) 指標。
