# 驗證報告：Ground Truth Replay 與環境一致性分析

**日期**：2026-01-03
**主題**：透過 GT Replay 驗證 Simulation Environment 與 Training Dataset 的一致性
**狀態**：已確認環境不一致 (Environment Mismatch)

---

## 1. 實驗背景與動機

在之前的評估中，我們觀察到一個矛盾現象：
*   **開迴路誤差 (Open-loop MSE)** 極低 (`0.0003`)，顯示模型預測極準。
*   **閉迴路成功率 (Closed-loop Success)** 為 `0%`，顯示模型在模擬器中完全無法完成任務。

為了釐清這是「模型執行問題」還是「環境設定問題」，我們設計了 **Ground Truth Replay (GT Replay)** 實驗。
**邏輯**：如果不使用模型預測，而是直接讓機器人在模擬器中執行數據集裡的「正確動作」，它能成功嗎？

---

## 2. 實驗方法 (Methodology)

我們修改了 `IsaacSimRunner` 與評估腳本，新增 `--replay_gt` 模式：
1.  **略過模型**：完全繞過 Policy Network 的預測。
2.  **讀取真值**：直接從 `validation_dataset` 中讀取當前時間步 $t$ 的 Action $a_t^{GT}$。
3.  **執行動作**：將 $a_t^{GT}$ 送入模擬器執行。

### 復現指令 (Reproduction)

```bash
uv run --active voilab eval-model \
   --checkpoint /mnt/zi/00_course/voilab/data/outputs/2026.01.03/19.34.31_train_diffusion_unet_timm_vit_finetune_umi/checkpoints/latest.ckpt \
   --output_dir data/eval_output_gt_replay \
   --task kitchen \
   --dataset_path ./AsiaDragon_All_285/simulation_dataset_clean.zarr.zip \
   --n_episodes 1 \
   --headless \
   --replay_gt
```

---

## 3. 實驗結果 (Results)

*   **動作執行**：機器人動作平滑，無異常抖動（排除單位 Scale 錯誤）。
*   **視覺觀察**：機器人手臂移動路徑與預期不相符 移動緩慢 跟資料產生時的影片不同。
*   **最終結果**：**失敗 (Failure)**。

### 單位問題 (Unit Scale) 的排除
針對「是否為單位換算錯誤（m vs mm）」的疑慮，我們認為機率極低：
1.  **動作幅度合理**：`Pos Mag` 落在 0.04~0.08 (4-8cm)，符合機械臂操作範圍。如果是單位錯誤，會變成 40-80m，機器人會飛出畫面。
2.  **數據集自洽**：我們是直接重播數據集數值。如果數據集單位有錯，當初就無法採集成功。

---

## 4. 結論與診斷 (Conclusion & Diagnosis)

**結論：環境不一致 (Environment Mismatch/Domain Bias)。**

目前的模擬環境 (Simulation Environment) 與採集數據集時的環境存在 **幾何偏差 (Geometric Offset)**。
可能的偏差來源：
1.  **機器人底座位置 (Robot Base Position)**：模擬器中的底座可能與數據集設置相差幾公分。
2.  **物體初始位姿 (Object Initial Poses)**：雖然使用了 `object_poses.json`，但可能存在座標系轉換或父子節點層級的差異。

## 5. 後續行動 (Action Items)

請優先集中資源修正模擬環境，而非調整模型訓練參數。

1.  **檢查 `generate_data.py`**：確認數據採集時的 Robot Base 設定。
2.  **檢查 `isaac_sim_runner.py`**：確認評估時的 Robot Load Position。
3.  **視覺對齊**：對比數據集第一幀影像與模擬器初始畫面的視角與相對位置。
