# Isaac Sim 模擬邏輯分析：抓取判定與成功條件

這份文件詳細記錄了在 `voilab` 模擬環境中，夾爪如何判定與物體「連結 (Attach)」以及任務「成功 (Success)」的底層邏輯。

---

## 1. 夾爪連結判定 (Magic Grasp 邏輯)

在助教提供的 `motion_plan.py` 中，採用的是一種**基於時序 (Temporal-based)** 的「Magic Grasp」機制，而非動態物理判定。

### 觸發機制
- **狀態階段**：當 `phase` 進入 `"close"`（夾爪關閉）。
- **觸發點**：計數器達到 **20 步** (`self.counter == 20`)。
- **邏輯假設**：系統假設在移動指令執行完畢且夾爪關閉 20 步後，夾爪一定位於物體的抓取範圍內。

### 實作方式 (如何 Attach)
當觸發點達成時，程式會執行以下步驟：
1. **取樣位姿**：獲取當前手掌 (EE) 的世界坐標 `T_ee` 與物體的世界坐標 `T_obj`。
2. **計算相對矩陣**：`T_ee_to_obj = inv(T_ee) @ T_obj`。
3. **強制座標同步**：在隨後的每一幀，程式會計算 `T_new_obj = T_curr_ee @ T_ee_to_obj`，並強制更新物體位置。這使得物體看起來像是被「黏」在夾爪上。

---

## 2. 如何計算「距離多靠近」？

雖然原始腳本使用計數器觸發，但我們可以根據 **UMI (Universal Manipulation Interface)** 的數據結構計算精確的物理距離：

### 計算公式 (Euclidean Distance)
可以使用 `ee_pos` (End-Effector Position) 與 `obj_pos` (Object Center) 的 L2 Norm：
```python
dist = np.linalg.norm(ee_pos - obj_pos)
```

### 參考指標
- **助教預設閾值**：在 `PickPlace` 的初始化中設有 `attach_dist_thresh = 0.1` (10cm)。
- **分析建議**：若要進行精細分析，可觀察在 `counter == 20` 的那一瞬間，`dist` 是否確實小於 10cm。

---

## 3. 任務成功判定 (Success Criteria)

成功判定邏輯位於 `kitchen_registry.py` 的 `is_episode_completed` 方法中。針對 `kitchen` 任務（杯子堆疊），判定標準如下：

### 核心判定條件
1. **垂直堆疊 (Vertical Ordering)**：
   - 藍色杯子 (`blue_cup`) 的 Z 軸坐標必須高於粉色杯子 (`pink_cup`)。
   - `blue_cup_pos[2] > pink_cup_pos[2]`
2. **水平對齊 (XY Alignment)**：
   - 兩個杯子中心在 XY 平面的距離必須小於 **0.03 公尺 (3cm)**。
   - `np.linalg.norm(blue_cup_pos[:2] - pink_cup_pos[:2]) < 0.03`

### 判定代碼參考
```python
xy_dist = np.linalg.norm(blue_cup_pos[:2] - pink_cup_pos[:2])
success = (blue_cup_pos[2] > pink_cup_pos[2]) and (xy_dist < 0.03)
```

---

## 4. `motion_plan.py` 關鍵參數解析

在 `scripts/motion_plan.py` 中，核心類別是 `PickPlace`（底層動作控制）與 `KitchenMotionPlanner`（任務邏輯）。以下是詳細參數解析：

### A. `PickPlace` 類別 (基底控制)
負責定義機器人抓取與放置的通用行為。

*   **基礎控制函數**：
    *   `get_end_effector_pose_fn`: 獲取機械臂末端 (EE) 當前坐標。
    *   `get_object_world_pose_fn`: 獲取目標物體當前世界坐標。
    *   `apply_ik_solution_fn`: 執行逆向運動學 (IK) 移動機械臂。
    *   `plan_line_cartesian_fn`: 生成笛卡爾空間直線路徑。

*   **動作控制參數**：
    *   **`grasp_quat_wxyz`**: 抓取姿態，預設為垂直向下。
    *   **`grasp_mode`**: 
        *   `"regular"`: 固定下向角度。
        *   `"object_based"`: 根據物體長軸自動計算角度（餐具適用）。
    *   **`open_width / close_width`**: 夾爪張開與閉合寬度 (預設 0.08m / 0.03m)。
    *   **`close_steps / hold_steps`**: 閉合持續 30 步，隨後停留 10 步以穩定抓取。
    *   **`step_move / step_descend`**: 移動步長。平移為 0.01m，接近物體時降速至 0.005m。

*   **連結判定參數**：
    *   `attach_dist_thresh`: 連結距離閾值 (10cm)。
    *   `gripper_close_thresh / gripper_open_thresh`: 夾爪物理開合狀態判定值。

### B. `KitchenMotionPlanner` (廚房任務邏輯)
定義針對疊杯子任務的具體動作偏移量 (Offsets)：

1.  **`pick_above_offset`**: `[-0.05, -0.075, 0.10]` (準備點，上方 10cm)
2.  **`pick_offset`**: `[-0.05, -0.075, -0.12]` (抓取深度，深入杯內 12cm)
3.  **`lift_offset`**: `[0, 0, 0.25]` (提起高度，抬升 25cm)
4.  **`place_above_offset`**: `[-0.05, -0.07, 0.15]` (目標物上方 15cm)
5.  **`place_offset`**: `[-0.05, -0.07, 0.03]` (最終放置高度，保留 3cm 餘裕)

### C. `PickPlace.start()` 關鍵參數
*   **`attached_object_path`**: 當前抓取對象（如：藍色杯子）。
*   **`target_object_path`**: 當前放置目標（如：粉色杯子）。
*   **`fix_target_pose`**: 若設定，則機器人前往固定座標而非動態追蹤目標物。
*   **`retreat_after_place`**: 設為 `True` 時，放掉物體後機械臂會自動抬升以防碰撞。

---

## 5. 數據多樣性分析 (Data Diversity Analysis)

針對使用 `KitchenMotionPlanner` 生成的模擬數據，其多樣性與侷限性分析如下：

### 為什麼數據具有多樣性？
雖然路徑邏輯（Offsets）是固定的，但**初始環境狀態**是動態的：
*   **物體初始位姿**：每個 Episode 會讀取不同的 `object_poses.json`。因為機械臂是移動到 `物體坐標 + Offset`，當杯子位置不同時，機械臂在世界坐標系下的運動軌跡（Trajectory）也會隨之改變。
*   **感知數據差異**：由於物體位置改變，相機觀測到的影像（RGB）以及機械臂末端坐標也會呈現數值的變化。

### 為什麼數據會呈現高度相似？
「相似性」主要體現在**動作風格 (Motion Style)**：
*   **速度一致**：所有回合均採用相同的平移與下降步長。
*   **節奏一致**：永遠遵循「準備點 -> 垂直下降 -> 抓取 -> 抬升」的固定節奏。
*   **解決方式單一**：缺乏避障、不同抓取角度或應對隨機干擾的探索性動作。

### 對 Diffusion Policy 訓練的影響
*   **優勢**：提供了大量高品質且覆蓋了不同空間分佈的成功範例，有助於模型學習基礎的物件關聯與空間對準。
*   **挑戰**：由於缺乏噪聲與失敗案例，模型學出來的動作可能過於僵硬。在後續優化中，建議可引入 **Action Noise** 或隨機化的 **Pick Offsets** 來增加魯棒性。

## 6. 參數間的關係與意義 (Variable Relationships)

### A. 「距離」vs 「寬度」：邏輯與物理的解耦
*   **參數對比**： generate_data的: `attach_dist_thresh` (10cm) vs motion_plan的 `close_width` (3cm)。
*   **邏輯關係**：`attach_dist_thresh` 決定了「何時可以開始黏合」，而 `close_width` 決定了「手指夾到多緊」。
*   **設計意義**：由於杯子有厚度與體積，夾爪中心不須與杯子中心重合。只要進入 10cm 範圍並執行閉合動作（至 3cm 寬），物理上就能完成穩定包覆。

### B. 「步數」與「時間」的轉換 (Tempo)
*   **基準頻率**：Isaac Sim 預設運行於 60 FPS (1 Step ≈ 0.016s)。
*   **閉合時間**：`close_steps = 30` (約 **0.5 秒**)。
*   **穩定時間**：`hold_steps = 10` (約 **0.16 秒**)。
*   **為什麼要 Hold？** 物理引擎（PhysX）計算摩擦力與接觸點需要時間。這 0.16 秒的停留能防止夾爪在物體還沒「坐穩」前就提起，避免產生穿透或物體彈飛。

### C. 「移動步長」與「動態精度」
*   **參數對比**：`step_move` (0.01m) vs `step_descend` (0.005m)。
*   **速度轉換**：
    *   `step_move` ≈ **0.6 m/s** (高效跨越)
    *   `step_descend` ≈ **0.3 m/s** (精準對準)
*   **設計意義**：在接近物體的最後 10cm，速度減半可減少機械臂因 IK 解算抖動產生的慣性，確保抓取前的姿態絕對穩定。

### D. 預抓取機制 (Pre-attachment)
*   **邏輯現象**：`close_steps` 設為 30，但 Attach 觸發在第 20 步。
*   **設計意義**：這是一種「視覺補償」。在夾爪尚未完全合擾（合到 2/3）時就啟動座標同步，會讓物體看起來像是因為夾爪收縮而自然地被吸往中心，而不是等夾爪夾死後物體才瞬間移動，整體視覺效果更為流暢。

## 8. 解釋 `motion_plan.py` 中的機器人從抓取到放置的完整流程

為了讓大家更直觀地理解 `PickPlace` 控制器是如何運作的，我們將整個過程拆解為機器人的「心路歷程」。想像機器人正在執行「把杯子疊起來」的任務：

### 階段 1：靠近與準備 (Approach)
1.  **`move_above` (就位)**：機器人先飛到杯子上方 10 公分`pick_above_offset`的地方停好。這是為了防止機械臂在橫向移動時直接撞倒杯子。
2.  **`descend` (深蹲)**：機械臂垂直往下伸，直接進入杯子內部中心（約 12 公分深`pick_offset`）。這時候夾爪是張開的，準備包覆杯壁。

### 關鍵階段 2：關鍵抓取 (The Magic Moment)
3.  **`close` (握緊)**：夾爪開始慢慢收縮到 `close_width`。
    -   **關鍵**：收縮過程需要 30 步。`counter` 數到 20 時，模擬器會認為「應該已經抓穩了」，於是啟動 **Magic Grasp (Attach)**，鎖定相對位姿。
    -   **瞬移判定**：就在這第 20 步，程式會算出夾爪與杯子的「相對距離」，並從這一刻起，讓杯子跟著夾爪一起動。
4.  **`hold` (穩定)**：夾爪合攏後，機器人會故意停一下（約 0.16 秒）等待 `counter` 到達 `close_steps` (預設 30)。這是為了讓物理引擎算清楚摩擦力，確保杯子不會在提起來的瞬間噴掉。

### 階段 3：搬運與對準 (Transport)
5.  **`lift` (起跳)**：抓穩後，機器人垂直向上抬升 25 公分`lift_offset`。
6.  **`move_place` (過場)**：提著杯子，移動到下一個目標（粉色杯子）的正上方 15 公分`place_above_offset`處。
7.  **`descend_place` (瞄準)**：慢慢往下降低到距離目標杯子僅剩 3 公分`place_offset`的高度。這時候杯子已經快要碰到下面的杯子了。

### 階段 4：釋放與撤離 (Release & Reset)
8.  **`release` (放手)**：夾爪張開至 8 公分寬 `open_width`。
    -   **解除魔咒**：就在放開的瞬間，解除杯子與夾爪的`attached` 狀態，清空相對位姿。杯子這時會因為重力，自然地落在下方的杯子上。
9.  **`post_place_lift` (收招)**：機械臂再次往上提，避免在回復位置的時候撞到剛剛疊好的成品。
10. **`done` (收工)**：所有動作結束，等待下一個指令。

### 關於距離的科學小實驗
我們在大日誌中看到的 `dist=0.1547m` (15公分)，其實就是在上述 **階段 2** 的「第 20 步」抓到的瞬間。這代表夾爪目前的中心點距離杯子的物理中心大約 15 公分——考量到杯子高度與夾爪伸入深度，這是非常合理的物理距離！

---

## 9. 進階觀察：為什麼錄影結尾「杯子沒疊好」就斷了？

如果在觀察 Front 或 Robot 視角的影片時，覺得杯子「還沒完全貼合」影片就結束了，這正是助教邏輯中的幾個特性交織而成的結果：

### 1. 提早放手的「3 公分餘裕」
在 `descend_place` 階段，機器人並不會把杯子「壓實」到底。
- **證據**：`place_offset` 的 Z 軸偏移設為 `0.03` (3cm)。
- **邏輯**：機器人移動到距離目標物上方 3 公分時，就認為執行完畢並開始執行 `release` (鬆開夾爪)。剩下的 3 公分是靠**「重力自由落體」**讓杯子自己疊上去。

### 2. 「結束即斷錄」的機制
在數據生成腳本 `generate_data.py` 中，錄影迴圈與控制器的 `done()` 狀態同步：
- **現象**：只要機械臂做完最後的抬升 (`post_place_lift`)，控制器就會立刻回報 `done` 並跳出錄影迴圈。
- **結果**：這時杯子可能還在空中掉落，或者剛碰到杯緣還在晃動（物理引擎還在算 Settling），但因為錄影程式已經關閉，**最後杯子靜止貼合的完美瞬間往往沒被收錄進去。**

### 3. 成功判定標準較寬
在 `kitchen_registry.py` 中，只要「藍杯比粉杯高」且「水平距離 < 3cm」，系統就判定成功。這意味著即使杯子在空中旋轉，只要位置對了，系統就會提前慶祝並結束當前 Episode。

### 總結
這確實是原始邏輯的一部分。它是為了**資料生成效率**而做的權衡。如果要錄製更漂亮的影片，通常需要在 `release` 後多留 0.5~1 秒的「靜置錄製時間 (Settling Time)」。

---

## 7. 總結與評估 (Evaluation Applicability)

*   **資料採集面**：這組參數建立了高成功率的「標準化流程」，適合產出大量的成功示範 (Demonstrations)。
*   **評估 (Eval) 面**：在模型評估時，模型輸出的是連續控制。建議 Eval 應專注於監控 **`dist < attach_dist_thresh`** 這一實體物理指標，而非依賴步數。

---
*Created by Team 1 (AsiaDragon) - Simulation Logic Investigation*
