# 5.4 主线论文中文版大纲

## 1. 一句话定位

本研究要回答的不是“TabICL 能否单独打败所有经典 GS 模型”，而是：**当 foundation model 单独迁移到 genomic selection（GS）场景时往往难以稳定发挥作用，但如果把它与统计遗传中的已有先验知识进行结构化融合，它就能够在 GS 中释放出稳定的预测价值。**

进一步说，本文想提供的不只是一个关于 TabICL 的案例，而是一条更一般的思路：**很多并非为 GS 原生设计的模型，未必适合作为单独预测器直接进入 GS；但如果能与经典统计遗传 prior 有效结合，它们依然可能成为有价值的 GS 组件。**

## 2. 文章主线

全文主线可以固定为四句话：

1. `no_prior-TabICL` 说明 foundation model 在 GS 中并非完全无效，但也不是稳定最优。
2. `BayesB / GBLUP / RKHS` 代表了 GS 中已经被长期验证的统计遗传先验。
3. 当 TabICL 与这些 prior 进行结构化融合后，预测性能可以在 trait 层面获得更稳定的提升。
4. 因而，foundation model 在 GS 中的价值，不一定体现为“替代经典模型”，也可以体现为“与经典 prior 结合后提升经典模型”。

## 3. Introduction

### 3.1 背景切入

- GS 长期依赖 `GBLUP`、Bayes 家族和核方法等统计模型。
- 这些方法分别从 polygenic relationship、sparse marker effect、nonlinear/kernel similarity 等角度建模遗传信号。
- 近年来 foundation model 在多类表格任务中表现出潜力，但其在 GS 场景中的直接迁移价值仍不明确。

### 3.2 当前空白

- GS 数据与常规 tabular benchmark 在样本规模、特征维度、遗传结构和噪声来源上差异很大。
- 因此，foundation model 即使在通用表格任务上有效，也未必能在 GS 中单独稳定超过经典统计模型。
- 现有工作更少讨论一个关键问题：**如果 foundation model 不能单独占优，它是否仍然可以作为一个可利用的信息源，为 GS 带来增益？**

### 3.3 核心问题

本文围绕三个递进问题展开：

1. `TabICL alone` 在 GS 中到底处于什么水平？
2. 如果它不是稳定最优，那么它与经典统计 prior 是否存在互补性？
3. 如果存在互补性，能否通过统一、低额外计算开销的融合框架把这种互补性稳定转化为预测增益？

### 3.4 本文贡献

建议固定为四点：

1. 证明 foundation model 单独迁移到 GS 时具有竞争力，但通常不是稳定最优。
2. 提出一个与统计遗传 prior 结构化融合的框架，使 TabICL 在 GS 中获得稳定实际价值。
3. 证明单 prior 与多 prior 融合都可产生增益，其中 `triple-prior-TabICL` 取得最稳定的主结果。
4. 提供一个可复用的思路：让非 GS 原生模型通过与经典 prior 结合进入 GS，而不要求它们先单独超越所有传统模型。

## 4. Results

结果部分固定为 4 章，不单独再拆出额外的“观点章”。

### Result 1. TabICL 单独用于 GS 时具有竞争力，但并非稳定最优

#### 要回答的问题

- `no_prior-TabICL` 单独作为预测器时，是否已经具备可用的 GS 预测能力？
- 它与 `BayesB / GBLUP / RKHS` 的相对位置如何？

#### 核心叙事

- 这一章不能写成“TabICL 打败经典模型”。
- 应写成：TabICL 单独迁移到 GS 后，在部分 trait 上具有竞争力，说明其确实学到了与预测有关的信息；但其表现并不稳定超过 strongest baseline，因此单独使用并不足以成为文章终点。
- 这正构成后续引入 prior 融合的必要性。

#### 建议展示

- `no_prior-TabICL` vs `BayesB / GBLUP / RKHS`
- 按 trait 的 Pearson 或 R2 对比图
- strongest baseline 分布情况

#### 本章要落下的结论

**Foundation model 在 GS 中不是“无用”，但也不是“单独即最优”；它更像是一个有潜在互补性的预测来源。**

### Result 2. 与统计 prior 融合后，TabICL 可以稳定提升 GS 预测性能

#### 要回答的问题

- 单 prior 融合是否稳定优于对应 prior-only？
- 多 prior 融合是否进一步提升整体稳定性？

#### 主结果方法

- `single_bayesb_two_step_ls`
- `single_gblup_two_step_ls`
- `single_rkhs_two_step_ls`
- `triple_two_step_ls`

#### 核心叙事

- 先展示 single-prior-TabICL：说明 TabICL 与任一经典 prior 结合后，通常都能把该 prior 再抬高。
- 再展示 triple-prior-TabICL：说明多 prior 融合后，整体表现进一步稳定，并成为全文主结果。
- 这里要明确强调：真正有意义的不是 “prior-only 的线性拼接”，而是 “prior + TabICL” 的融合增益。

#### 建议展示

- 三条 single 方法相对各自 prior 的提升百分比
- `triple_two_step_ls` 相对 strongest baseline 的提升百分比
- `triple_two_step_ls` vs `only-triple-prior`
- `single / triple` 的 trait-level compare 主表

#### 本章要落下的结论

**TabICL 的价值主要体现在“把已有统计 prior 进一步提升”，而不是“替代这些 prior”。**

### Result 3. 权重分布揭示了 TabICL 与统计 prior 之间的互补性

#### 要回答的问题

- 融合为什么有效？
- TabICL 在融合中扮演的到底是“主导者”还是“补充者”？
- 单 prior、dual prior、triple prior 的权重变化能说明什么？

#### 核心叙事

- 展示 `single / dual / triple` 下的权重分布，以及 `only-prior` 与 `prior + TabICL` 的差异。
- 重点不在于把 TabICL 写成绝对主导，而在于说明：即使 TabICL 只承担部分权重，它依然可以持续提供可转化为精度增益的互补信息。
- 这一章也是解释全文主线最关键的一章：foundation model 在 GS 中不需要“一个人赢”，它可以作为融合系统中的重要组成部分赢。

#### 建议展示

- `w_tabicl` 与 prior 权重分布图
- `only-prior` vs `prior + TabICL` 的对比
- `single / dual / triple` 权重变化图
- 不同权重求解方案的补充比较可放补充材料

#### 本章要落下的结论

**TabICL 在融合中的作用不是简单重复已有 prior，而是在已有统计遗传先验之外补充了新的可预测信息。**

### Result 4. 该融合框架在样本量和标记量变化下保持鲁棒性

#### 要回答的问题

- 当训练样本量下降时，`TabICL + prior` 的优势是否仍然存在？
- 当 marker 数量减少时，这种融合优势是否仍然存在？

#### 当前固定实验口径

- 默认 `exclude pig3534`
- 样本量：`20% / 60% / 100%`
- 标记量：`1000 / 2000 / 5000 / 10000`
- `repeat = 3`
- 固定既有 `best_block`，不重新搜索
- 继续采用解偶复用逻辑

#### 核心叙事

- 样本量实验主要支撑：在小样本区间，foundation model 与 prior 融合仍然能产生实际增益。
- 标记量实验主要支撑：这种增益并不依赖于某一个特定 marker 规模，具有一定鲁棒性。
- 这章不是单纯做 robustness appendix，而是为全文立意服务：如果一种融合思路只在 full-data、full-marker 条件下有效，它的普适价值会有限；反之，若在不同资源条件下仍然成立，就更能说明这是一个可推广的方法学方向。

#### 本章要落下的结论

**TabICL 与统计 prior 的融合，不只是某些 trait 上的偶然 gain，而是在样本量和标记量变化下仍具稳定趋势的策略。**

#### Discussion 连接句

- 多 prior 融合的价值不是无条件压倒所有 baseline，而是：
  - 在 trait 结构与 prior 匹配时提供可观补偿
  - 在小样本下对困难 trait 尤其有帮助
  - 在某些 trait 上会随着样本量增加而进一步显现互补收益
- 因此，更准确的表述应是：
  - 融合收益具有 trait-dependent 和 sample-size-dependent 特征
  - prior 数量本身不是决定性因素，prior 匹配度更关键

## 5. Discussion

### 5.1 主结论

- TabICL 单独用于 GS 时并不稳定最优。
- 但这并不意味着 foundation model 对 GS 没有价值。
- 当它与统计遗传 prior 进行结构化结合后，可以在 GS 中产生持续、可量化的预测增益。

### 5.2 立意拔高

这里建议用接近定稿的语言：

> Foundation model 直接迁移到 GS 领域时，往往难以单独稳定发挥作用；但这并不意味着这类模型对 GS 没有价值。本研究表明，当 foundation model 与统计遗传中的已有先验知识进行结构化结合时，它可以在 GS 中释放出实际预测价值。因而，我们的工作提供的，不只是一个关于 TabICL 的结果，也是一条让更多非 GS 原生模型进入 GS 领域的潜在路径。

### 5.3 为什么这个结论重要

- 它缓解了“新模型必须先单独打败经典模型，才值得进入 GS”的思维惯性。
- 它把研究问题从“替代”转向“融合”。
- 它为未来更多 tabular model、foundation model、representation model 进入 GS 提供了方法论启发。

### 5.4 局限性

- 当前验证对象以 TabICL 为主，尚未扩展到更多 foundation model。
- 当前 prior 主要固定为 `BayesB / GBLUP / RKHS`。
- 权重学习方式虽已相对稳定，但仍可继续拓展到更一般的融合器形式。

### 5.5 未来工作

- 扩展到更多非 GS 原生模型
- 扩展到更多统计 prior 组合
- 检验该思路在更大样本、更高密度 marker、更多物种中的可迁移性

## 6. Methods

### 6.1 数据与评估范围

- 主结果目录：`outputs/5.4-duli-liudang`
- 主结果默认排除：`pig3534`
- 正式 baseline 固定为：
  - `BayesB`
  - `GBLUP`
  - `RKHS`

### 6.2 正式主结果

- `single_bayesb_two_step_ls`
- `single_gblup_two_step_ls`
- `single_rkhs_two_step_ls`
- `triple_two_step_ls`

### 6.3 解偶复用逻辑

- 底层正式留档只训练：
  - `no_prior-TabICL`
  - `BayesB`
  - `GBLUP`
  - `RKHS`
- 融合结果全部基于留档直接构建：
  - `single-prior-TabICL`
  - `dual-prior-TabICL`
  - `triple-prior-TabICL`
  - `only-prior`

### 6.4 权重学习

- 主结果权重方法固定为 `two_step_ls`
- 先拟合 prior 组合，再拟合 `prior` 与 `TabICL` 的融合
- 主文以该方法为准，其他权重方案放入补充材料

### 6.5 Baseline 几何与 trait 类型分析

- 对每个 trait，记三条 baseline 准确率为 `acc_B`、`acc_G`、`acc_R`
- 定义三 baseline 均值：
  - `mean3 = (acc_B + acc_G + acc_R) / 3`
- 定义相对均值偏差：
  - `rel_B = (acc_B - mean3) / mean3`
  - `rel_G = (acc_G - mean3) / mean3`
  - `rel_R = (acc_R - mean3) / mean3`
- 三角图中的 trait 位置不直接使用原始准确率，而使用上述相对均值偏差
- 该定义表达的是：
  - 某个 trait 更偏向 `BayesB`、`GBLUP` 还是 `RKHS` 这一类 baseline 结构
  - 而不是不同 trait 之间绝对准确率的高低
- 对 `no_prior-TabICL` 的表现，使用以下两个量进行关联分析：
  - `no_prior vs best baseline (%)`
  - `no_prior vs mean3 (%) = (no_prior - mean3) / mean3 * 100`
- 另外定义 baseline 主导程度：
  - `dominance margin = strongest baseline - second-strongest baseline`
  - 可在原始准确率口径或相对均值偏差口径下计算
- 该分析用于回答：
  - 什么样的 trait 更接近 `no_prior`
  - 什么样的 trait 更需要 prior-integrated fusion

## 7. 图表建议

### 主文图

1. `no_prior-TabICL` 与 3 个 baseline 的 trait-level 对比
2. 三条 single-prior-TabICL 相对各自 prior 的提升分布
3. `triple_two_step_ls` 相对 strongest baseline 的提升分布
4. `single / dual / triple` 的权重分布图
5. 样本量与标记量鲁棒性曲线

### 主文表

1. 主结果表：3 个 single + 1 个 triple
2. `triple_two_step_ls` 相对 strongest baseline 的逐 trait 百分比汇总

### 补充材料

1. `no_prior` 与全部 baseline 的完整数值表
2. `only-prior / single / dual / triple` 的完整 compare
3. 不同权重方案的对比
4. 全部样本量与标记量实验明细表

## 8. 当前建议固定口径

- 不把文章写成“TabICL 单独战胜统计模型”
- 把文章写成“foundation model 单独进入 GS 时不稳定，但与统计遗传 prior 融合后可以发挥作用”
- 主结果集中在：
  - `single_bayesb_two_step_ls`
  - `single_gblup_two_step_ls`
  - `single_rkhs_two_step_ls`
  - `triple_two_step_ls`
- 主结果默认 `exclude pig3534`
- 样本量和标记量实验作为 Result 4 的鲁棒性支撑

## 9. 可直接用于正文的核心表述

### 引言/讨论可用句

> 我们的目标不是证明 TabICL 单独优于所有经典 GS 模型，而是证明：即使 foundation model 本身并非 GS 原生模型、也难以单独稳定占优，只要与已有统计遗传先验有效结合，仍然能够在 GS 任务中产生稳定增益。

> 因此，本文提供的并不仅是一个针对 TabICL 的经验案例，更是一种面向 GS 的一般性思路：对于那些难以直接在 GS 中单独奏效的非原生模型，与其要求它们先替代经典模型，不如探索它们如何与经典 prior 结合并共同提升预测性能。
