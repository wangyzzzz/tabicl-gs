# 样本量影响实验设计稿

日期: 2026-04-24

## 1. 实验目标

本实验不再回答“哪个模型在 full-data 下平均精度最高”，而是回答:

- 当前固定版 `dual-prior TabICLv2` 对训练样本量缩减是否更稳健？
- `FM` 引入带来的增益，是否会在 `small-n` 区间放大？
- 这种现象是否在不同数据集、不同 trait 上具有相似趋势？

本实验的主叙事是:

- `sample efficiency`
- `small-n advantage`
- `relative performance retention`

而不是单纯比较不同数据集之间的绝对 Pearson 大小。

## 2. 数据集与 trait 选择

本轮固定使用 5 个数据集，每个数据集选 2 个 trait:

- `rice529`
- `cotton1245`
- `soybean951`
- `pig3534`
- `wheat406`

每个数据集的 2 个 trait 选择规则如下:

1. `gain trait`
   - 选择当前 `dual_prior - prior_only` 提升较明显的 trait
2. `flat trait`
   - 选择当前 `dual_prior - prior_only` 提升较弱、接近 0，或略为负的 trait

目的:

- 让样本量实验既覆盖“FM 确实有增益”的情形
- 也覆盖“FM 增益有限”的情形

因此，本实验的 trait 选择不是随机，而是有意构造成一组:

- `strong-gain case`
- `weak/flat-gain case`

## 3. 样本量缩减方式

### 3.1 主横轴

使用相对训练比例，而不是绝对样本数。

固定比例为:

- `10%`
- `20%`
- `40%`
- `60%`
- `80%`
- `100%`

理由:

- 不同数据集样本量不同，绝对样本数不适合直接横向比较
- 当前实验主要关心“同一个 trait 内，相对样本量下降后性能怎么变”
- 比例设计更适合看 retention curve

### 3.2 抽样位置

仍然保留当前外层 `5-fold CV`。

对于每个 outer fold:

- `outer-test` 固定不变
- 只在该 fold 的 `outer-train` 内部按比例抽子集

因此，每个比例下的比较都共享同一个测试集，保证可比性。

### 3.3 重复抽样

为了避免单次子抽样带来较大方差，每个比例在每个 outer fold 中重复抽样:

- `3` 次

记作:

- `repeat = 1, 2, 3`

所有模型共享同一套子样本索引。

## 4. 模型结构固定策略

本实验采用:

- `固定结构、样本缩减、重新训练`

而不是在每个样本比例下重新搜索全部超参数。

### 4.1 固定的内容

对于每个 trait，以下内容固定:

- `window` block 策略
- `max_snps = 10000`
- `PCA = 99% explained variance`
- `group = 1`
- 该 trait 在 full-data fold1 中搜索得到的 `best block size`

### 4.2 每个比例下重新训练的内容

在每个 `outer fold × proportion × repeat` 组合下，重新训练:

- 第一层 TabICLv2
- 第二层 TabICLv2
- `BayesB`
- `GBLUP`
- `alpha / w`

其中:

- `alpha / w` 仍然遵循当前固定版逻辑
- 在该比例对应的 reduced-train 上，通过 inner-OOF 重新估计

### 4.3 不采用“全量重搜版”的原因

如果在每个样本比例下都重新搜索 `block size`，会引入额外的结构噪声:

- 样本量影响
- 搜参不稳定性

会混在一起，不利于回答“仅仅减少样本量时，模型衰减速度如何”。

因此本实验更适合作为:

- `sample efficiency study`

而不是:

- `joint structure re-optimization study`

## 5. 参与比较的模型

当前建议固定比较以下 5 条主线:

- `dual_prior`
- `prior_only`
- `no_prior`
- `BayesB`
- `GBLUP`

说明:

- `BayesA / BayesLasso` 可作为补充记录，但不必进入主图
- 主图保持紧凑，避免视觉复杂度过高

## 6. 指标体系

## 6.1 基础测试指标

每个 `fold × proportion × repeat × model` 记录:

- `test_pearson`
- `test_r2`

其中主分析以 `test_pearson` 为主。

## 6.2 同一 trait 内的相对指标

### 1. Performance Retention

对于某个 trait 和某个模型:

- `Retention(p) = Pearson(p) / Pearson(100%)`

用途:

- 测量在样本量减少时，性能保留了多少

### 2. Small-n Gap

对于某个 trait:

- `Gap(p) = Pearson_dual(p) - Pearson_prior_only(p)`

用途:

- 测量 FM 分支在该比例下相对统计先验是否仍有额外增益

### 3. Small-n Amplification

定义:

- `Amplification = mean(Gap(10%,20%,40%)) - mean(Gap(80%,100%))`

解释:

- 若该值 > 0，则说明 dual-prior 相对 prior-only 的优势在小样本区更大

### 4. Retention AUC

对每个模型在一个 trait 上计算 `Retention curve` 的面积:

- `AUC_retention`

用途:

- 用单个数概括整个样本量区间的稳健性

## 7. 核心图表设计

### 图 1: 单 trait retention curve

每个 trait 画一张曲线图:

- x 轴: `10% / 20% / 40% / 60% / 80% / 100%`
- y 轴: `Retention`
- 曲线:
  - `dual_prior`
  - `prior_only`
  - `no_prior`
  - `BayesB`
  - `GBLUP`

### 图 2: 每个数据集的双 trait 对照图

每个数据集将两个 trait 并排展示:

- 左: `gain trait`
- 右: `flat trait`

用途:

- 看同一数据集内，小样本效应是否依赖 trait 类型

### 图 3: Small-n amplification barplot

- x 轴: trait
- y 轴: `Small-n Amplification`

用途:

- 看哪些 trait 在小样本区明显更依赖 FM

### 图 4: Dataset-level summary

对每个数据集汇总:

- `mean AUC_retention`
- `mean Small-n Amplification`

作为辅助总览图。

## 8. 落盘规范

新增输出目录建议:

- `outputs/sample_size_impact/<dataset>/<trait>/`

建议文件结构:

- `selection.json`
  - 记录该 trait 是否为 `gain trait` 或 `flat trait`
- `sample_subsets/`
  - 保存每个 fold、每个比例、每个 repeat 的样本索引
- `fold_metrics.csv`
  - 列:
    - `fold`
    - `repeat`
    - `proportion`
    - `model`
    - `n_train`
    - `test_pearson`
    - `test_r2`
- `trait_summary.csv`
  - 汇总 `Retention`, `Gap`, `AUC_retention`, `Amplification`
- `timing_summary.csv`

## 9. 结果解释框架

本实验最终不强调:

- 不同数据集之间谁的绝对 Pearson 更高

而强调:

- 在同一 trait 内，样本量下降时性能怎么波动
- dual-prior 相比 prior-only 是否更抗样本量缩减
- FM 增益是否集中出现在小 `n` 区

因此，本实验允许:

- 不同数据集之间绝对精度不可直接比较
- 只要趋势具有一致性，即可支持结论

## 10. 预期结论形式

如果结果支持预期，结论可写成:

1. `dual_prior` 的 retention curve 高于 `prior_only / no_prior`
2. `no_prior` 在 small-n 区域衰减更快
3. `dual_prior - prior_only` 的 gap 在 small-n 区域被放大

这将支撑一个更高层的叙事:

- `FM-derived block representation acts as a sample-efficient contextual prior, whose contribution becomes more apparent when training data are scarce.`

## 11. 当前设计的一句话总结

本实验采用:

- `固定结构`
- `相对比例缩样`
- `重复重训`
- `同一 trait 内看相对保留率与 small-n 增益`

用于评估当前固定版 `dual-prior TabICLv2` 是否具有真正的 `sample efficiency` 优势。

