# 2026-05-06 样本量 / 标记量补充实验方案

## 1. 当前决定

当前补充实验采用以下固定口径：

- 沿用 `5.4-duli-liudang` 主线
- 继续使用解偶复用逻辑
- 默认排除 `pig3534`
- `repeat = 3`
- 主结果口径仍然围绕：
  - `single_bayesb_two_step_ls`
  - `single_gblup_two_step_ls`
  - `single_rkhs_two_step_ls`
  - `triple_two_step_ls`

## 2. 样本量实验

### 2.1 目标

回答两个问题：

- 当训练样本量下降时，`single-prior-TabICL` 和 `triple-prior-TabICL` 是否仍能稳定优于对应对照？
- `TabICL + prior` 的增益是否在小样本区间更明显？

### 2.2 当前推荐设计

采用轻量版样本量实验：

- 只新增两个比例：
  - `20%`
  - `60%`
- `100%` 直接复用当前 `5.4-duli-liudang` 的 full-data 正式结果

说明：

- 这样可以最小化新增计算量
- 同时保留一个“小样本点”和一个“中等样本点”
- 最终形成 `20% / 60% / 100%` 三点曲线

### 2.3 trait 范围

默认排除 `pig3534`，只做以下 4 个数据集：

- `cotton1245`
- `rice529`
- `soybean951`
- `wheat406`

每个数据集选 `2` 个 trait：

- `gain trait`：当前 `triple_two_step_ls` 提升较明显
- `flat trait`：当前 `triple_two_step_ls` 接近 0 或较平

当前推荐 trait：

- `cotton1245`
  - gain: `cotton_fiblen_17_18_cotton_fiblen_17_18`
  - flat: `cotton_fibelo_17_18_cotton_fibelo_17_18`
- `rice529`
  - gain: `grain_weight`
  - flat: `grain_width`
- `soybean951`
  - gain: `lw_beijing_2013_lw_beijing_2013`
  - flat: `bbd_beijing_2013_bbd_beijing_2013`
- `wheat406`
  - gain: `sl_e1`
  - flat: `sl_e2`

总计：

- `4 dataset × 2 trait = 8 trait`

### 2.4 数据切分与复用原则

- 外层 `5-fold` 切分保持一致
- `outer-test` 固定不变
- 只在每个 fold 的 `outer-train` 内部做子采样
- 每个比例：
  - `repeat = 3`
- 所有模型共享同一套子样本索引

### 2.5 是否重新搜索 block

当前默认结论：

- **主实验中不重新搜索 block**

原因：

- 样本量实验的目标是隔离“训练样本量变化”本身的影响
- 如果在 `20%` 或 `60%` 下重新搜索 block，会把：
  - 样本量变化
  - 结构重搜噪声
  混在一起
- 小样本下 block search 本身不稳定，容易把实验解释搞乱
- 也不符合当前“解偶复用 + 固定结构”主线

因此当前建议：

- 直接复用 full-data 主线中对应 trait 的 `best_block`

一句话：

- 样本量实验应是 `fixed structure, reduced train size, retrain models`
- 而不是 `re-search structure under each sample ratio`

### 2.6 样本量实验中的解偶复用

对于每个 `trait × fold × proportion × repeat`：

底层重新训练并留档：

- `no_prior-TabICL`
- `BayesB`
- `GBLUP`
- `RKHS`

然后再从这些留档中直接构建：

- `single_bayesb_two_step_ls`
- `single_gblup_two_step_ls`
- `single_rkhs_two_step_ls`
- `triple_two_step_ls`

也就是说：

- 仍然遵循解偶复用
- 不为 single / triple 重复训练底层模型

## 3. 标记量实验

### 3.1 目标

回答：

- 当 marker 数量减少时，`TabICL + prior` 是否仍然保持优势？

### 3.2 当前推荐设计

先做和样本量实验同一批 `8 trait`

marker 横轴建议：

- `1000`
- `2000`
- `5000`
- `10000`

说明：

- `10000` 直接对应当前正式主线
- 其余点为下采样 marker 版本

### 3.3 block 处理建议

当前默认也建议：

- **主实验中不重新搜索 block**

原因与样本量实验相同：

- 先固定结构
- 只观察 marker 数量减少本身带来的变化

如需做“是否需要重搜 block”的补充，可在附录中只挑 `1-2` 个 trait 做一个小型敏感性检查，不进入主表。

## 4. 当前推荐输出目录

样本量实验建议：

- `outputs/5.4-sample_size-decoupled`

标记量实验建议：

- `outputs/5.4-marker_size-decoupled`

## 5. 当前建议优先级

优先级排序：

1. 样本量实验
2. 标记量实验

原因：

- 样本量实验更贴近当前主线
- 更容易支撑 `sample efficiency` / `small-n utility`
- 标记量实验更偏鲁棒性补充
