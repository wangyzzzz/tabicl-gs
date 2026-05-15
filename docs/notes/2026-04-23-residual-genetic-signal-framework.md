# Residual Genetic Signal Framework 设计稿

日期: 2026-04-23

## 1. 目标重定义

当前项目在 `dual-prior TabICLv2` 上已经取得稳定结果。下一阶段不再以“继续卷平均 Pearson”为主目标，而是回答一个更高层的问题:

- `BayesB + GBLUP` 这类统计遗传先验解释之外，是否存在可被局部上下文模型（TabICLv2 block 表征）恢复的残余遗传信号？

该问题对应的工作名称为:

- `Residual Genetic Signal Framework (RGSF)`

## 2. 核心变量定义

以下定义全部在外层 `5-fold` 框架下，以外层测试集预测为准。

记:

- `y`：真实表型
- `y_bayesb`：BayesB 预测
- `y_gblup`：GBLUP 预测
- `y_tabicl`：第二层 TabICLv2 分支预测
- `alpha, w`：当前固定版 dual-prior 全局权重

定义:

- `y_prior = alpha * y_bayesb + (1 - alpha) * y_gblup`
- `y_dual = w * y_tabicl + (1 - w) * y_prior`
- `r_prior = y - y_prior` (prior 未解释残差)
- `c_tabicl = y_dual - y_prior = w * (y_tabicl - y_prior)` (TabICLv2 净增量贡献)

解释:

- `y_prior` 代表经典统计先验可解释部分
- `r_prior` 代表经典先验未解释部分
- `c_tabicl` 代表 dual 模型中由 TabICLv2 带来的额外修正项

## 3. 框架核心假设

RGSF 的核心假设不是“TabICLv2 全面替代 baseline”，而是:

- `c_tabicl` 与 `r_prior` 在统计上显著对齐
- 即 TabICLv2 主要在补 prior 残差信号，而不是简单重复 prior 已有信息

## 4. 一级指标体系

### 4.1 Trait 级主指标

每个 trait 在外层 5 折上计算以下指标并取均值。为避免指标“同义重复”，主指标分为方向、能量和唯一增益三类:

- `Directional Alignment`: `corr(c_tabicl, r_prior)`
- `Residual Coverage R2`:
  - `R2_residual = 1 - Var(r_prior - c_tabicl) / Var(r_prior)`
- `Unique Incremental R2`:
  - `DeltaR2_unique = R2(y ~ y_prior + y_tabicl) - R2(y ~ y_prior)`
- `Prior Explainability`: `corr(y_prior, y)`
- `Dual Gain` (仅保留为辅助结果): `corr(y_dual, y) - corr(y_prior, y)`

说明:

- `Directional Alignment` 与 `Dual Gain` 在经验上高度耦合，不能作为两条独立证据
- `Residual Coverage R2` 与 `DeltaR2_unique` 是独立证据，必须同时报告
- 取消 `Residual Gain Ratio`，避免在 prior 已很强的 trait 上分母过小导致不稳

### 4.2 稳定性指标

- `fold-wise std` of `Directional Alignment`
- `fold-wise std` of `Residual Coverage R2`
- `sign consistency`: 5 折内 `Directional Alignment > 0` 的比例

### 4.3 Trait 分层与异常情形

新增 trait 类型分层，防止叙事混淆:

- `prior-degenerate`:
  - `corr(y_prior, y) < 0.20`
  - 该类 trait 上 `r_prior` 近似整体信号，需单独汇报，不并入主结论均值
- `prior-effective`:
  - `corr(y_prior, y) >= 0.20`
  - 主结论以该类 trait 为主

## 5. 二级解析模块（修正版）

### 模块 A: 信号分解总览（方向 + 能量 + 唯一增益）

目标:

- 在每个 trait 上把信号拆成 `prior` 与 `TabICLv2 residual` 两部分
- 明确区分“方向对齐”和“残差能量覆盖”

输出:

- trait 表: `prior_only / dual_prior / no_prior / baselines`
- trait 表: `Directional Alignment / Residual Coverage R2 / DeltaR2_unique / Prior Explainability`
- 汇总图: trait 二维相图

相图坐标建议:

- x 轴: `Prior Explainability`
- y 轴: `Residual Coverage R2`

### 模块 B: 残差信号 block 定位（主证据模块）

目标:

- 定位哪些 block 最能承载 `r_prior` 对齐信息

原则:

- 以 block 为遗传单元，不先讨论单 SNP 归因
- 先做可复现实验量，不先做复杂解释学
- block 结果作为“新增信息位置证据”，优先级高于相关系数描述

建议计算:

- `delta_align_b = corr(c_tabicl_without_block_b, r_prior) - corr(c_tabicl_full, r_prior)`
- `delta_cov_b = R2_residual_without_block_b - R2_residual_full`
- 两者联合排序，避免只看方向不看能量

输出:

- 每 trait 的 block 排名
- 跨 fold block 排名一致性
- top blocks 的基因组区间

### 模块 C: 非可加信号对照（新增）

目标:

- 验证 TabICLv2 增量是否与“非可加遗传信号”一致

建议对照:

- `prior_only`
- `prior + RKHS`（或 EGBLUP）
- `dual_prior`

核心输出:

- trait 层 `dual_gain` 与 `RKHS_gain` 的相关性
- `Residual Coverage R2` 在三者之间的变化

说明:

- 该模块提供机制层独立证据，避免仅靠相关系数叙事

### 模块 D: GWAS 对照（非 accuracy 导向）

目标:

- 比较“GWAS 显著性信号”与“RGSF 残差信号”在位点层面的关系

建议对照组:

- `GWAS(y)` top hits
- `GWAS(r_prior)` top hits
- `RGSF top residual blocks -> SNP 集合`

核心输出:

- overlap 与 LD-aware overlap
- unique loci 比例
- trait-specific shared / orthogonal 模式
- 对 `prior-degenerate` 与 `prior-effective` 分层汇报

重点:

- 主叙事是 `shared vs orthogonal residual signal`
- 不是“谁 p-value 更小”

## 6. 最小可执行实验包（Smoke）

先做 2 个 trait:

- 一个 `dual - prior_only` 提升明显
- 一个提升较弱或接近 0

流程:

1. 读取现有 outer-fold 结果（不改训练流程）
2. 计算 `y_prior, y_dual, r_prior, c_tabicl`
3. 计算 `Directional Alignment / Residual Coverage R2 / DeltaR2_unique`
4. 输出 block 删除影响量 (`delta_align_b`, `delta_cov_b`) 与一致性
5. 跑基础 GWAS 对照（先用统一简化配置）
6. 输出 1 张相图 + 2 张 trait case 表

验收标准:

- 至少一个 trait 显示 `Residual Coverage R2` 明显为正且稳定
- 能明确给出 shared 与 orthogonal 位点的定量比例
- 能在 block 层给出稳定 top residual blocks

## 7. 工程落盘规范

新增输出目录建议:

- `outputs/residual_signal_framework/<dataset>/<trait>/`

建议最小文件集:

- `fold_level_signals.csv`
  - 列: `fold, y_true, y_prior, y_tabicl, y_dual, r_prior, c_tabicl`
- `trait_metrics.json`
  - 包含 `Directional Alignment / ResidualCoverageR2 / DeltaR2_unique` 与稳定性指标
- `block_residual_importance.csv`
  - 列: `block_id, delta_align, delta_cov, fold, rank`
- `gwas_compare_summary.json`

统一要求:

- 所有统计量必须能从外层测试集还原
- 禁止使用测试集标签参与任何权重估计与模型拟合

## 8. 与当前主模型的关系

RGSF 不替代当前固定版模型，而是作为其“第二部分分析框架”。

它回答的是:

- 当前模型新增预测贡献的遗传来源是什么
- 这些来源与 GWAS 之间是重叠关系还是互补关系

因此，RGSF 是“机制分析层”，不是“新预测器层”。

## 9. 论文层面的叙事提升点

从“方法精度比较”升级为“遗传信号分解与机制假设”:

- 不是只报告 dual-prior 比 baseline 高多少
- 而是报告 dual-prior 增益对应的残差信号组成
- 并给出其与 GWAS 的 shared / orthogonal 证据

建议主结论句式:

- `TabICLv2 does not merely improve prediction marginally; it recovers trait-specific residual genetic signals not fully captured by classical genomic priors.`
