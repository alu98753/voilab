# 07 - JSON Episode Index Mapping Bug 修復報告

## 問題概述

在評估 GT Replay 時，杯子位置與機器人軌跡不對齊，差距約 0.2-0.3m。

## 問題分析過程

### 1. 初始假設：座標旋轉問題

最初懷疑是 ArUco 座標轉換缺少 90° 旋轉。經過追蹤代碼發現：

- `umi_replay.py` 在轉換機器人軌跡時有 90° Z 旋轉
- `object_loader.py` 在轉換物體位置時沒有此旋轉

但這個假設後來被排除，因為 **Zarr 中的 `robot0_eef_pos` 已經是世界座標**（從 `art_kine_solver.compute_end_effector_pose()` 直接獲取）。

### 2. 真正的根因：JSON Episode Index 對應錯誤

#### 數據結構不一致

| 數據來源 | 總幀數 | 總 Episodes | 平均幀/集 |
| :--- | :--- | :--- | :--- |
| **Zarr (simulation)** | 44,062 | 245 | 180 幀 |
| **JSON (object_poses)** | 160,789 | 311 | 517 幀 |

這是**完全不同的計數系統**！

#### 舊代碼的錯誤邏輯

```python
# isaac_sim_runner.py (舊代碼)
rb_start = episode_to_sampler_indices[dataset_ep_idx]['rb_range'][0]
for j, entry in enumerate(object_poses_data):
    ep_range = entry.get('episode_range', [0, 0])
    if ep_range[0] <= rb_start < ep_range[1]:  # ← 錯誤！比較不同計數系統
        json_idx = j
        break
```

問題：
- `rb_range` 是**模擬幀數**
- `episode_range` 是**原始視頻幀數**
- 兩者數值範圍完全不同，匹配結果純屬巧合

#### 隱藏的關鍵因素：status 過濾

`generate_data.py` 在生成數據時會**跳過 `status != 'full'` 的 JSON 條目**：

```
JSON[0]: status=full → Zarr ep 0 ✓
JSON[1]: status=full → Zarr ep 1 ✓
...
JSON[4]: status=none → 跳過 ✗
JSON[5]: status=full → Zarr ep 4 ✓
...
```

這導致 Zarr episode N **不等於** JSON index N。

#### 具體錯誤案例

```
dataset_idx = 20
舊映射 (錯誤): JSON[6]  ← 使用 episode_range 匹配
新映射 (正確): JSON[25] ← 只計算 status=full 的條目
```

## 解決方案

### 修改 `isaac_sim_runner.py`

1. **建立正確的對映表**（lines 459-470）：

```python
# Build zarr_to_json mapping: only count 'full' status entries
zarr_idx = 0
for json_idx, entry in enumerate(object_poses_data):
    if entry.get('status') == 'full':
        zarr_to_json[zarr_idx] = json_idx
        zarr_idx += 1
```

2. **使用對映表查詢**：

```python
# 簡化為一行查詢
json_idx = zarr_to_json.get(dataset_ep_idx, dataset_ep_idx)
```

## 驗證結果

```
[IsaacSimRunner] Built zarr_to_json mapping: 282 Zarr episodes -> 311 JSON entries
[IsaacSimRunner] Map dataset_idx 20 -> JSON index 25  ✓
[IsaacSimRunner] Positioned pink cup at [4.95, 2.68, 1.1]
```

對比 GT 抓取位置 `[4.99, 2.48, 0.91]`，仍有約 0.2m Y 軸誤差，可能需要進一步調查。

## 修改的文件

| 文件 | 修改位置 | 說明 |
| :--- | :--- | :--- |
| `isaac_sim_runner.py` | lines 451-470 | 新增 `zarr_to_json` 對映表建立 |
| `isaac_sim_runner.py` | lines 507-511 | 簡化 JSON index 查詢邏輯 |

## 關鍵教訓

1. **不同數據源的幀計數可能完全不同**：視頻幀 vs 模擬幀
2. **數據生成時的過濾邏輯必須在評估時重現**：`status='full'` 過濾
3. **直接使用索引對應比複雜的範圍匹配更可靠**

## 後續待辦

- [ ] 調查剩餘的 ~0.2m Y 軸誤差
- [ ] 確認是否與 blacklist 過濾有關
- [ ] 驗證 ArUco 轉換參數是否完全一致
