# 当前固定版模型流程

本文档描述当前已经固定下来的正式模型流程。该版本基于前期多轮消融、结构比较与服务器正式实验结果确定，作为后续新性状、新数据集和论文方法部分的默认主线。

## 1. 模型总览

当前固定版模型由三个顺序模块组成：

1. `block-wise TabICLv2 表征学习`
2. `genome-level TabICLv2 预测`
3. `dual-prior 全局校准`

其核心思想是，先利用第一层 TabICLv2 从局部 SNP block 中提取非线性遗传表征，再由第二层 TabICLv2 在全基因组尺度上整合这些 block 表征，最后引入 `BayesB` 与 `GBLUP` 两个统计遗传学先验，对深度模型输出进行全局融合与校准。

基于近期关于 `group=[1,3,5,7,10]` 的系统比较，group 数对整体准确率影响极小，虽然 `group=10` 在均值上略优，但提升幅度不足 `0.1%`。因此，当前主线默认将 group 固定为 `1`，即使用全局 gate，而不再进行样本分组或多组门控。

## 2. 数据输入与预处理

基因型输入统一采用 PLINK 二进制格式，包括 `.bed`、`.bim` 和 `.fam` 文件。当前数据入口为：

- `genome/rice529/plink/rice529`

表型输入来自：

- `genome/rice529/rice529_phe.csv`

若原始 SNP 数量超过 `10000`，则首先使用固定随机种子执行可复现下采样，将总 SNP 数限制为 `10000`。后续所有 block 划分均在该下采样后的 SNP 集合上进行。

当前固定版采用：

- `window` 分组策略

即按照 SNP 当前顺序连续切分 block。若最后一个 block 中 SNP 数不足，则进行补齐以保持输入张量形状一致。

## 3. 第一层：block-wise TabICLv2 表征学习

设下采样后的 SNP 被划分为 `K` 个 block，第 `k` 个 block 的输入矩阵记为 `X^(k)`。对于每个 block，调用真实的 `TabICLv2` 回归接口进行建模，而非 mock、近似实现或预先缓存的 `.npy` 特征。

第一层的关键不在于输出标量，而在于提取样本级隐藏向量表示。当前固定版使用的 embedding 提取逻辑为：

1. block 数据输入真实 `TabICLRegressor`
2. 模型内部经过 `col_embedder`、`row_interactor` 和 `icl_predictor`
3. 取 `row_interactor` 的输出作为样本级 block representation

因此，每个 block 首先输出一个高维样本向量，而不是单个 scalar prediction。

当前主线配置为：

- `embedding_extraction_mode = legacy`
- `norm = none`
- `n_estimators = 1`
- `feat_shuffle_method = none`
- `batch_size = 1`
- `kv_cache = repr`

## 4. 第一层后的 PCA 压缩

由于第一层 block embedding 原始维度较高，直接拼接到第二层会造成输入维度过大，因此需要对每个 block 的 embedding 单独做降维。

当前固定版采用 PCA，并以累计解释率作为动态截断标准：

- `embedding_explained_variance_target = 0.99`

也就是说，对于每个 block，PCA 保留达到 `99%` 累计解释率所需的最少维度，得到该 block 的低维表示 `z^(k)`。

因此，第 `k` 个 block 最终输出的是：

- 一个 PCA 后的低维 embedding 向量

当前正式主线中：

- `include_block_scalar = false`

即不再拼接 block scalar prediction，只保留 block embedding。

## 5. 第二层：genome-level TabICLv2 预测

将所有 block 的 PCA 后向量按 block 顺序拼接，形成样本级第二层输入：

`Z = [z^(1), z^(2), ..., z^(K)]`

随后，将 `Z` 输入第二层 `TabICLv2`，学习跨 block 的全基因组组合关系，输出深度模型预测值：

- `y_tabicl`

因此，第二层模型不再直接处理原始 SNP，而是在第一层提取的局部遗传表征之上建立全局预测函数。

## 6. 统计遗传学双先验

除第二层 `TabICLv2` 外，当前固定版还同时构建两个统计遗传学先验模型：

- `BayesB`
- `GBLUP`

其中实现要求为：

- `BayesB` 必须通过 `BGLR`
- `GBLUP` 必须通过 `sommer`

记二者预测分别为：

- `y_bayesb`
- `y_gblup`

这两个先验不是单独作为 baseline 进行比较后弃用，而是被正式纳入最终融合结构，用作对深度模型的校准信息。

## 7. Dual-prior 全局校准

当前固定版已将 group 默认固定为 `1`，因此不再使用多组 gate，而是直接学习一组全局融合参数：

- `alpha`
- `w`

其中：

- `alpha` 用于在 `BayesB` 与 `GBLUP` 之间分配 prior 权重
- `w` 用于在 `TabICLv2` 与 prior mixture 之间分配最终权重

融合公式为：

`y_prior = alpha * y_bayesb + (1 - alpha) * y_gblup`

`y_final = w * y_tabicl + (1 - w) * y_prior`

等价展开后，三个分支的最终权重分别为：

- `TabICLv2 = w`
- `BayesB = (1 - w) * alpha`
- `GBLUP = (1 - w) * (1 - alpha)`

因此，当前固定版的最终输出不是“纯双层 TabICL”，而是：

- 第一层 TabICL 提供局部表征
- 第二层 TabICL 提供深度全基因组预测
- BayesB 和 GBLUP 提供统计遗传先验
- 最后通过全局 gate 进行融合

## 8. Block 超参数搜索策略

当前固定版中，唯一保留的核心结构超参数是：

- `block size`

搜索方式如下：

1. 在外层 `5-fold CV` 中，只对 `fold 1` 执行超参数搜索
2. 在 `fold 1` 的 outer-train 上做 `3-fold inner-OOF`
3. 用 inner-train 训练第一层和第二层无 prior 的 `TabICL -> TabICL`
4. 用 inner-val 平均 Pearson 作为 block size 评价标准
5. 选出当前 trait 的最优 block size

因此，block size 是：

- `trait-specific`
- 但不是每个 outer fold 都重新搜索

## 9. Gate 参数的确定方式

当前固定版严格避免直接用 in-sample 预测去拟合融合权重。具体流程如下：

1. 在 `fold 1 outer-train` 中做 `3-fold inner-OOF`
2. 对每个 inner split：
   - 用 inner-train 训练第一层 TabICL
   - 用 inner-train 训练第二层 TabICL
   - 用 inner-train 拟合 `BayesB`
   - 用 inner-train 拟合 `GBLUP`
3. 在 inner-val 上得到：
   - `y_tabicl_oof`
   - `y_bayesb_oof`
   - `y_gblup_oof`
4. 用这些 OOF prediction 估计全局 `alpha` 和 `w`
5. 将这组参数冻结

随后：

- `fold 1` 用该组参数预测 outer-test
- `fold 2-5` 复用 `fold 1` 的最优 block 和 frozen gate

因此，当前正式固定版是：

- `fold1 inner-OOF` 决定 block 与 gate
- `fold2-5` 直接复用，不再重新搜索和重新学习 gate

## 10. 交叉验证与评估

当前评估协议为：

- 外层 `5-fold CV`

每折记录：

- 测试集 Pearson
- 测试集 `R²`

正式对比的 baseline 包括：

- `GBLUP`
- `BayesA`
- `BayesB`
- `BayesLasso`

其中要求为：

- `GBLUP` 全流程围绕 `sommer`
- `BayesA/BayesB/BayesLasso` 全流程围绕 `BGLR`

主结果以 5-fold 平均指标汇总，并进行模型间横向比较。

## 11. 当前固定版默认设置

当前可以把正式主线固定为：

- 输入格式：PLINK
- 数据集：`rice529`
- 最大 SNP 数：`10000`
- 分组策略：`window`
- block 超参数：在 `fold1 inner-OOF` 中搜索
- 第一层模型：`TabICLv2`
- 第一层表示：`row_interactor hidden representation`
- PCA：每个 block 保留 `99% explained variance`
- block scalar：不使用
- 第一层 norm：`none`
- 第二层模型：`TabICLv2`
- 双先验：`BayesB + GBLUP`
- 最终融合：`global dual-prior gate`
- group 数：默认固定为 `1`
- 评估方式：外层 `5-fold CV`

## 12. 一句话总结

当前固定版模型不是简单的“双层 TabICL”，而是一个三段式结构：

`Block-wise TabICLv2 representation learning -> Genome-level TabICLv2 prediction -> Dual-prior global calibration with BayesB and GBLUP`

其中，第一层负责学习局部遗传片段中的非线性表征，第二层负责整合全基因组尺度的信息，而最终的 dual-prior gate 负责将深度模型与统计遗传学先验进行稳定融合。
