# Isaac Sim Evaluation Pipeline: Modification Summary & Usage Guide

本文件總結了針對 `voilab` 專案中 Isaac Sim 評估流程（尤其是 GT Replay 功能）所做的各項修正與功能增強。

## 1. 建議提交策略 (Commit History)

為了保持 Git 歷史的清晰，建議將變更分為以下四個邏輯提交：

### Commit 1: 基礎 GT Replay 功能實作
*   **Message**: `feat(eval): add Ground Truth (GT) Replay support to Isaac Sim evaluation`
*   **變更內容**:
    *   在 `cli.py` 與 `eval_kitchen.py` 中引入 `--replay_gt` 參數。
    *   在 `IsaacSimRunner` 中實作讀取 `validation_dataset` 並按步執行 GT Action 的機制。
    *   修正 `eval_kitchen.py` 中 `max_steps_per_episode` 的限制，確保 Replay 不會提早結束。

### Commit 2: 修正運動學執行精度問題
*   **Message**: `fix(runner): resolve frame skipping and relative pose drift in IsaacSimRunner`
*   **變更內容**:
    *   **強制單步執行**: 針對 GT Replay 強制 `exec_steps=1`，確保每一幀資料都被處理，不再跳幀。
    *   **座標參考對齊**: 在 Action Chunk 開始時擷取固定基座 EE Pose，後續相對位移均疊加於此基礎，消除了累計位移偏差 (Drift)。

### Commit 3: 環境一致性與穩定性增強
*   **Message**: `fix(env): align evaluation environment with data generation settings`
*   **變更內容**:
    *   **變體選擇**: 在 `_setup_simulation` 中強制選擇 `AlternateFinger` 機器人變體，對齊資料生成端的 TCP 定義。
    *   **預先校正**: 在動作迴圈前呼叫 `set_robot_base_pose` 同步解算器與實際基座位置。
    *   **穩定性優化**: 加入 Render Warmup 步驟與 `SimulationApp` 更新機制，嘗試緩解 `libomni` 在初始化時的崩潰。

### Commit 4: 監控工具與定量分析實作
*   **Message**: `feat(debug): implement MSE tracking and add dataset inspection tools`
*   **變更內容**:
    *   在 `IsaacSimRunner` 中實作自動計算與 Dataset 之間的 Position/Rotation MSE。
    *   新增 `inspect_dataset_lengths.py` 用於快速檢查清點數據集狀態。
    *   建立 `My_Observation/` 系列文件紀錄實驗發現與環境對比。

---

## 2. 修改點與原因詳述 (Modifications & Rationale)

| 修改檔案 | 具體修改點 | 理由 / Rationale |
| :--- | :--- | :--- |
| `isaac_sim_runner.py` | 機器人變體設為 `AlternateFinger` | 助教錄製資料是用此型號，預設熊貓爪長度不同，不改則 IK 解算會偏位。 |
| `isaac_sim_runner.py` | 呼叫 `set_robot_base_pose` | Solver 預設基座在 (0,0,0)，若機器人在場景中有偏移則必須手動告知 Solver。 |
| `isaac_sim_runner.py` | `exec_steps = 1` (GT Mode) | 預設 Policy 會跳步 (n步)，但 GT 模式為了 1:1 還原必須每一幀都踩腳步。 |
| `isaac_sim_runner.py` | 擷取 `base_ee_pos/rot` 參考點 | 避免將每一格的相對位移錯誤地疊加在「已經動過」的暫態位置上。 |
| `eval_kitchen.py` | 注入 `val_dataset` 給 Runner | 讓 Runner 能直接存取 zarr 資料進行定量 MSE 比較。 |
| `cli.py` | 暴露 `--replay_gt` | 讓使用者能透過命令行進入「真值重播」模式進行 Debug。 |

---

## 3. 如何執行 (How to Run)

### 執行 GT Replay (驗證環境與資料對齊)
```bash
uv run --active voilab eval-model \
  --checkpoint {CKPT_PATH} \
  --task kitchen \
  --dataset_path {ZARR_PATH} \
  --n_episodes 1 \
  --replay_gt \
  --headless
```
*   **新參數說明**:
    *   `--replay_gt`: 啟用真值重播。此模式下不會使用 Policy 預測，而是直接從 Zarr 抓取 Action 執行。
    *   `--output_dir`: 若在 Docker 中執行，請務必使用 **相對路徑** (例如 `data/eval_out`)，以確保 Volume Mapping 正確對應到主機。

### 數據集狀態檢視
```bash
uv run python scripts/inspect_dataset_lengths.py
```
這將輸出 Zarr 數據集各 Episode 的長度分布、Action 統計以及 Key 名稱，有助於確認資料清洗是否成功。

### 查看定量分析報告
評估結束後，終端機將輸出：
*   **Position MSE**: 機器人末端執行器與原始資料的歐幾里得距離誤差。
*   **Rotation MSE**: 旋轉角度誤差。
這兩個指標越接近 0，代表環境模擬與真值越吻合。
