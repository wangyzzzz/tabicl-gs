# Methods 中文草稿

## 1. 研究设计概述

本研究旨在评估一个并非为 genomic selection（GS）原生设计的 table foundation model，是否能够在 GS 场景中发挥实际预测价值。我们首先将 `TabICL` 作为不依赖统计先验的独立预测器进行评估（`no_prior-TabICL`），随后进一步检验其与经典统计遗传模型的融合是否能够带来稳定增益。为避免 single/dual/triple 融合过程中对底层模型进行重复训练，我们采用了一套“解偶复用（decoupled reuse）”流程：底层只正式训练 `no_prior-TabICL`、`BayesB`、`GBLUP` 和 `RKHS`，所有 single-prior、dual-prior、triple-prior 以及 only-prior 结果均由这些留档预测直接构建。

当前正式主线为 `5.4-duli-liudang`。主结果默认排除 `pig3534`，因此正文主分析基于 `36` 个非猪 trait；如果连同 `pig3534` 一并统计，则共包含 `41` 个 trait。主文正式比较的 baseline 固定为 `BayesB`、`GBLUP` 和 `RKHS`，不再混用旧 baseline 口径。

## 2. 数据集与分析范围

主线分析包含 `rice529`、`Cotton1245`、`Soybean951`、`wheat406` 和 `pig3534` 五个多性状数据集。各 trait 均以独立任务处理，并分别完成模型训练、外层测试与融合汇总。考虑到猪数据集在当前阶段与其他作物数据集的生物学背景和结果解释口径存在差异，主文主结果统一采用 `exclude pig3534` 的设置，因此 Result 1-3 的主体统计基于 `cotton1245`、`rice529`、`soybean951` 和 `wheat406` 的 `36` 个 trait。

在补充鲁棒性分析中，我们另外构建了两个代表性 trait 子面板。样本量实验选择 4 个非猪数据集中的 8 个代表性 trait，每个数据集各选 2 个 trait，用于覆盖“融合收益较明显”和“收益较平缓”两类情形。SNP 数量实验使用同一组 8 个 trait，以便在不同 marker 密度下进行可比分析。TabPFN 补充验证也在这一 `10K SNP + 8 traits` 面板上进行，从而将其定位为对框架可扩展性的补充证据，而不是新的主结果来源。

## 3. 基因型预处理与 marker 设置

所有正式实验均基于 PLINK 格式的基因型输入，并在每个数据集中对基因型与表型样本进行严格对齐，仅保留同时具有基因型和目标表型观测的个体。主线 `5.4-duli-liudang` 统一采用 `10K SNP` 设置，并在各数据集内使用固定随机种子进行 SNP 子集抽样和缓存留档，以保证 `no_prior-TabICL` 与三条统计 baseline 使用完全一致的 marker 输入。

对于样本量实验，我们保持 marker 数量固定为主线的 `10K SNP`，只改变训练样本比例。对于 SNP 数量实验，我们设置 `2K`、`10K` 和 `50K` 三档 marker count：其中 `10K` 直接复用主线正式留档，`2K` 与 `50K` 则重新构建对应的 SNP 子集，并在该 marker 条件下重新生成底层模型留档。由此，样本量实验主要隔离“训练样本数量变化”的影响，marker-count 实验主要隔离“遗传标记密度变化”的影响。

## 4. 数据集切分、嵌套评估与信息泄露控制

### 4.1 trait 级独立建模与统一外层切分

所有主线与补充实验均在 trait 层面独立运行。也就是说，每个 trait 被视为一个单独的预测任务，分别完成数据切分、模型训练、outer-test 评估和融合汇总，不在不同 trait 之间共享目标变量信息。对于同一个 trait，`no_prior-TabICL`、`BayesB`、`GBLUP`、`RKHS` 以及后续所有 fusion 方法，均共享完全一致的外层 `5-fold cross-validation` 划分。这样可以保证不同方法的比较建立在相同的训练样本和测试样本之上，避免由于 split 不一致而带来的伪差异。

在解偶复用流程中，这一点尤为关键。因为 fusion 不是重新训练底层模型，而是直接读取 `no_prior` 和 baseline 的已留档 outer-test 预测进行组合，所以这些底层预测必须对应完全相同的 outer-test 个体顺序与真实值。当前流程在融合前会显式检查同一 trait 下不同结果目录中的 `sample_id` 和 `y_true` 是否逐项一致；只有在这些信息对齐时，才允许进入后续融合与 compare 汇总。由此，single、dual 和 triple 之间的差异可以被解释为“融合规则不同”，而不是“测试集不同”。

### 4.2 outer-test 与 inner OOF 的嵌套设计

所有最终汇报的预测性能均来自 outer-test，而不是来自 inner 验证。具体而言，对于每个 trait，我们先进行外层 `5-fold` 划分；每次用 `4/5` 样本作为 outer-train，用剩余 `1/5` 样本作为 outer-test。模型在 outer-train 上拟合后，只对该 fold 的 outer-test 个体生成预测。最终该 trait 的性能由 5 个 outer fold 的测试结果取平均得到。

除 outer-test 预测外，我们还为每个 trait 额外留档 `fold_1` 的 inner out-of-fold（OOF）预测。具体做法是：仅在 `fold_1` 的 outer-train 集合内部，再执行 `3-fold` inner cross-validation；每次只用 inner-train 子集拟合模型，并预测对应 inner-valid 子集，最终拼接得到覆盖整个 `fold_1 outer-train` 的 OOF 预测向量。这里的 inner OOF 预测有两个严格限定的用途：第一，用于搜索或确认 `TabICL` 的 block 大小；第二，用于估计 fusion 所需的权重。由于每个 inner-valid 样本的预测都来自一个未见过该样本的 inner-train 模型，因此这一 OOF 设计本身也避免了在权重学习阶段把训练样本的自拟合结果误当成泛化预测。

### 4.3 信息泄露控制原则

本研究在方法设计上重点控制了三类潜在信息泄露。

第一，`best_block` 的确定不允许使用 outer-test 信息。对于需要搜索 block 的设定，我们只在 `fold_1` 的 outer-train 内部使用 inner OOF 表现进行搜索，并以 `inner OOF Pearson` 最优的 `group_size` 作为该 trait 在该设定下的正式 block。当前 `5.4-duli-liudang` 主线中，`10K SNP` 的 `no_prior-TabICL` 直接复用此前 dual-prior 正式实验已经留档的 `best_block.json`；但该文件本身同样只来源于相同 trait、相同外层切分下的 `fold_1 outer-train` inner OOF 搜索结果，而不包含任何 outer-test 标签信息。因此，这种复用属于开发集内结果的复用，而不是测试信息回流。

第二，fusion 权重的学习不允许接触 outer-test。无论是 prior-only 的内部权重 `alpha`，还是 `two_step_ls` 中 prior 与 `TabICL` 之间的融合权重 `beta`，都只在 `fold_1` 的 inner OOF 预测上学习一次。随后，这组 trait 级权重会被冻结，并直接应用到全部 5 个 outer-test fold 上。换言之，我们没有在每个 outer fold 上重新调权重，也没有根据 outer-test 表现反向修正 fusion 规则。这样的设计会牺牲一部分“每折单独最优”的自由度，但它更严格地区分了“开发信息”与“测试信息”，也更符合把 fusion 视为一个先确定、再评估的 trait 级规则。

第三，补充实验中的子采样和 marker 改动同样不允许污染 outer-test。样本量实验中，所有子采样都只发生在每个 outer fold 的 outer-train 内部，outer-test 个体始终保持不变；而且在同一个 `trait × fold × proportion × repeat` 组合下，`TabICL` 与三条 baseline 共享完全相同的子样本索引。SNP 数量实验中，marker 子集的构造不依赖于 outer-test 表型标签，且在新的 marker 条件下，block 搜索与 fusion 权重学习仍然严格限制在 `fold_1 outer-train` 的 inner OOF 范围内。TabPFN 补充验证沿用同样的控制原则。

综上，本研究中所有超参数选择、权重估计和补充实验控制变量，均建立在“outer-test 只用于最终评估”这一原则上。主文中报告的 Pearson 和 `R^2` 均可被解释为未参与结构选择和权重学习的外层测试性能，而不是经过测试集反馈调优后的乐观估计。

## 5. no_prior-TabICL 主线

### 5.1 主线思想

`no_prior-TabICL` 用于回答最直接的问题：table foundation model 在不依赖任何统计遗传先验的情况下，是否已经能够在 GS 中建立竞争力。该模型的 outer-test 结果既作为 Result 1 的直接比较对象，也作为后续各类 prior-integrated fusion 的一个组成部分。

### 5.2 block 大小与正式口径

`TabICL` 的一个关键结构超参数是 block 大小（代码中对应 `group_size`）。在常规设定下，可以基于 `fold_1` 的 inner OOF 表现对 block 大小进行搜索，并以 `inner OOF Pearson` 最优的 block 作为该 trait 的正式设置。当前 `5.4-duli-liudang` 主线遵循一个更严格的复用原则：`10K SNP` 主线中的 `no_prior-TabICL` 直接复用此前 dual-prior 正式实验已经确定的 `best_block.json`，不再为 triple 主线重新搜索 block。这样做的目的，是保证 `no_prior`、single、dual 和 triple 在主线中使用的是同一套底层 `TabICL` 结构，而不把结构重搜带来的波动混入方法比较。

在固定 `best_block` 后，我们重新运行该 block 下的完整 outer `5-fold` 预测，并同时在 `fold_1` 外层训练集内部生成 `TabICL` 的 inner OOF 预测并留档。对于新的 marker-count 设定（如 `2K` 和 `50K`），则不再复用 `10K` 主线的 block，而是在该 marker 条件下单独执行一次 `fold_1` inner OOF block search，再用得到的最优 block 运行完整 outer-test 和 inner OOF 留档。TabPFN 补充验证线也遵循这一逻辑。

## 6. 统计 baseline：BayesB、GBLUP 与 RKHS

我们选用 `BayesB`、`GBLUP` 和 `RKHS` 作为正式 baseline，分别代表稀疏大效应建模、多基因背景建模以及核方法/非线性相似性建模三类经典 GS 先验。三条 baseline 与 `no_prior-TabICL` 使用完全相同的表型输入、marker 子集和外层数据切分，并同样在每个 trait 上生成 outer `5-fold` 测试结果。

为支持解偶复用融合，三条 baseline 还需要在 `fold_1` 的 outer-train 内部生成各自的 inner OOF 预测。具体做法与 `TabICL` 相同：在 outer-train 内再做 `3-fold` inner cross-validation，训练 baseline 并预测 inner-valid，最终留档完整的 inner OOF 预测向量和对应真实值。这样，每个 trait 最终都会形成一套可以直接复用的底层预测档案：

- `no_prior-TabICL`：outer-test 5 折预测 + `fold_1` inner OOF
- `BayesB`：outer-test 5 折预测 + `fold_1` inner OOF
- `GBLUP`：outer-test 5 折预测 + `fold_1` inner OOF
- `RKHS`：outer-test 5 折预测 + `fold_1` inner OOF

## 7. 解偶复用融合框架

### 7.1 总体思路

single-prior、dual-prior、triple-prior 以及 only-prior 结果均不再重复训练底层模型，而是完全基于上一步已经留档的 `TabICL` 和 baseline 预测进行直接构建。这样做有两个目的。第一，可以保证所有融合方案严格共享同一套底层预测，因此不同融合方式之间的差异只来自“如何组合这些预测”，而不来自底层模型重新训练带来的随机波动。第二，在需要比较不同权重求解方式时，可以在不重复训练模型的前提下进行公平横向评估。

在这一框架下，主文重点汇报的融合方法包括：

- `single_bayesb_two_step_ls`
- `single_gblup_two_step_ls`
- `single_rkhs_two_step_ls`
- `triple_two_step_ls`

同时，为了理解融合增益的来源，我们也保留以下对照：

- `only_single_bayesb`
- `only_single_gblup`
- `only_single_rkhs`
- `only_dual`
- `only_triple`
- `dual_two_step_ls`

其中，dual 结果主要用于机制分析与权重比较，而不是当前论文主文的正式主结果线。

### 7.2 prior-only 的构建

设某一 trait 在 `fold_1` inner OOF 上的真实表型向量记为 `y`，某组 prior 模型的 inner OOF 预测组成矩阵记为 `P = [p_1, p_2, ..., p_k]`。对于 only-prior 结果，我们先求解一组非负且和为 `1` 的 prior 权重 `alpha`：

`alpha = argmin ||y - P alpha||^2,  s.t. alpha_j >= 0, sum_j alpha_j = 1`

在当前实现中，这一步通过“非负、和为 1 的最小二乘”完成。对于 single-prior 情形，`k = 1`，因此 prior-only 结果就等价于该 baseline 本身；对于 dual 和 triple，则由多条 baseline 的 inner OOF 预测组合得到一个 prior-only 预测 `y_prior = P alpha`。得到 `alpha` 后，同一组权重会被固定应用到 5 个 outer-test fold 上，以构建该 trait 的 only-prior 预测结果。

### 7.3 two-step least squares 融合

主文正式结果采用 `two_step_ls` 作为统一的权重学习方式。该方法分为两步：

第一步，先在 inner OOF 上求解 prior 内部的 simplex 权重 `alpha`，得到 prior-only 预测 `y_prior = P alpha`。

第二步，再将 `y_prior` 与 `TabICL` 的 inner OOF 预测 `y_tabicl` 作为两个候选分量，求解一组非负且和为 `1` 的融合权重 `beta = (beta_prior, beta_tabicl)`：

`beta = argmin ||y - beta_prior y_prior - beta_tabicl y_tabicl||^2,`

`s.t. beta_prior >= 0, beta_tabicl >= 0, beta_prior + beta_tabicl = 1`

最终，融合模型在 outer-test 上的预测为：

`y_fusion = beta_prior * y_prior_outer + beta_tabicl * y_tabicl_outer`

在这一框架中，`w_tabicl` 定义为 `beta_tabicl`，表示 `TabICL` 在最终融合中的整体贡献；各 prior 在最终融合中的有效权重则为 `beta_prior * alpha_j`。这些权重全部只在 `fold_1` inner OOF 上学习一次，随后固定应用于该 trait 的全部 outer-test fold。也就是说，我们不会在 outer-test 上重新拟合权重，从而避免测试集信息泄露。

### 7.4 其他权重方案

除 `two_step_ls` 外，我们还对比了两类备选权重求解方式。第一类是 `two_step_clip`：prior 内部权重仍先由 simplex 约束下的最小二乘求得，但 `TabICL` 与 `y_prior` 之间的融合系数不再通过二元最小二乘求解，而是先在每个样本上计算

`w_i = clip((y_i - y_prior,i) / (y_tabicl,i - y_prior,i), 0, 1)`

再取这些 `w_i` 的均值得到全局 `w_tabicl`。第二类是 `all_ls`：将 `TabICL` 与所有 prior 同时放入一个和为 `1` 的非负最小二乘中一次性求解。当前主文不以这两类方法作为正式结果，只将其作为补充分析，用于解释不同权重定义下性能和权重分布的变化。

## 8. 评价指标与相对提升计算

主文以 `Pearson correlation` 作为主要准确率指标，并以 `R^2` 作为辅助指标。对于每个 trait 和每种方法，我们先在每个 outer-test fold 上分别计算 `Pearson` 和 `R^2`，再对 5 个 outer fold 取平均，得到该 trait 的最终表现。

在相对提升分析中，我们统一采用百分比变化：

`relative gain (%) = (acc_method - acc_reference) / acc_reference * 100`

其中 `acc_reference` 可以是对应的 prior-only 结果、单一 baseline、trait 内最优 baseline（best baseline），或者 `no_prior-TabICL`。文中所称 `strongest baseline` 或 `best baseline`，均指在 `BayesB`、`GBLUP` 和 `RKHS` 三条正式 baseline 中该 trait 表现最高者。`strongest single` 则指三条 single-prior-TabICL 中该 trait 表现最好的 single 融合。

## 9. 权重、trait 类型与 baseline geometry 分析

### 9.1 `w_tabicl` 的定义

在 single、dual 和 triple 的 `two_step_ls` 结果中，`w_tabicl` 均定义为第二步最小二乘中 `TabICL` 的系数 `beta_tabicl`。在 single-prior 情形下，该值直接反映“单一统计 prior 与 TabICL”之间的融合比例；在 triple 情形下，该值反映的是“prior 组合整体”与 `TabICL` 的融合比例，而不是某一个 baseline 与 `TabICL` 的直接对冲。因此，不同 prior 设定下的 `w_tabicl` 应被解释为“TabICL 相对于整个 prior 池的权重”，而不是跨设定可直接一一等同的局部参数。

### 9.2 baseline geometry

为分析什么样的 trait 更适合 `no_prior-TabICL`，以及什么样的 trait 更需要 prior-integrated fusion，我们进一步构建了一个基于 baseline 相对位置的 trait 几何描述。设三条 baseline 在某一 trait 上的准确率分别为 `acc_B`、`acc_G`、`acc_R`，则三 baseline 的均值定义为：

`mean3 = (acc_B + acc_G + acc_R) / 3`

相对均值偏差定义为：

`rel_B = (acc_B - mean3) / mean3`

`rel_G = (acc_G - mean3) / mean3`

`rel_R = (acc_R - mean3) / mean3`

这里的三角结构不再表达不同 trait 之间绝对准确率的高低，而是表达一个 trait 相对更偏向 `BayesB`、`GBLUP` 还是 `RKHS` 这一类 baseline 假设。我们进一步计算 `no_prior-TabICL` 与 best baseline 的相对差值，以及 `triple` 与 best baseline 的相对差值，并分析它们与 `rel_B`、`rel_G`、`rel_R` 的相关关系。

### 9.3 baseline 主导程度与分歧度

为压缩表达 trait 的 baseline 结构，我们还定义：

- `dominance margin = strongest baseline - second-strongest baseline`
- `spread = max(acc_B, acc_G, acc_R) - min(acc_B, acc_G, acc_R)`

此外，我们用 baseline 归一化 share

`share_i = acc_i / (acc_B + acc_G + acc_R)`

计算离散度指标

`entropy = - sum_i share_i * log(share_i)`

其中 `entropy` 越高，表示三类 baseline 越接近；`spread` 越大，表示三类 baseline 分歧越明显。上述指标主要用于 Result 3 的 trait 类型分层和相关性分析。

## 10. 样本量实验

样本量实验沿用主线的解偶复用逻辑，默认排除 `pig3534`，并围绕 4 个非猪数据集的 8 个代表性 trait 展开。对于每个 trait，我们固定 outer `5-fold` 划分和对应的 outer-test 个体，只在每个 outer-train 内部做子采样。当前正式设置为 `20%`、`60%` 和 `100%` 三档训练样本比例，其中 `100%` 直接复用主线 `5.4-duli-liudang` 的 full-data 留档结果，`20%` 和 `60%` 则在各 outer-train 内部重新训练 `no_prior-TabICL`、`BayesB`、`GBLUP` 和 `RKHS`，并重新生成 outer-test 预测与 `fold_1` inner OOF 留档。

为降低子采样噪声，我们对每个样本量比例设置 `repeat = 3`。在同一个 `trait × outer fold × proportion × repeat` 组合下，`TabICL` 和三条 baseline 共享完全相同的子样本索引。样本量实验的 `TabICL` block 大小不重新搜索，而是直接复用该 trait 在 `10K` 主线 full-data 条件下已确定的 `best_block`，从而使样本量实验真正隔离“训练样本数量变化”本身，而不把结构重搜带来的额外波动混入结果解释。

在当前正式整理版本中，样本量实验的 `8` 个 trait 已全部完成，因此主文 Result 4 的统计基于完整的 `8-trait` 面板汇总。

## 11. SNP 数量实验

SNP 数量消融实验使用与样本量实验相同的 8 个代表性 trait，但改变 marker 密度而不改变 outer split。我们设置 `2K`、`10K` 和 `50K` 三档 marker count，其中 `10K` 直接复用主线结果；`2K` 与 `50K` 则在各数据集内重新构建对应的 SNP 子集，并重新运行底层 `no_prior-TabICL` 与三条正式 baseline。

与样本量实验不同，marker 数量变化会改变 `TabICL` 的最优结构尺度，因此 `2K` 和 `50K` 两档不复用 `10K` 主线的 block，而是在该 marker 条件下独立执行 `fold_1` inner OOF block search，并以最优 `group_size` 重新运行完整 outer-test 与 inner OOF 留档。完成底层留档后，再按与主线一致的解偶复用逻辑构建 single-prior 与 triple-prior 结果。这样可以避免把不适配 marker 密度的旧 block 强行带入新 marker 条件。

## 12. TabPFN 补充验证

为检验 prior-integrated fusion 的思路是否仅对 `TabICL` 单一 backbone 成立，我们另外引入 `TabPFN` 作为补充验证模型。该实验限制在 `10K SNP + 8 traits` 的 marker-count 面板上，主要目的是评估“框架可扩展性”，而不是替代 `TabICL` 主线。

在这一补充线中，我们首先为 `TabPFN` 运行独立的 no-prior outer `5-fold` 预测，并同样在 `fold_1` outer-train 内部进行 inner OOF block search 与 inner OOF 留档。随后，复用主线已经正式生成的 `BayesB`、`GBLUP` 和 `RKHS` 留档结果，在相同 outer split 下构建 `only-triple-fusion` 与 `triple-fusion`。其中 `only-triple-fusion` 仅由三条 baseline 线性组合得到，`triple-fusion` 则进一步将 `TabPFN` 纳入两步式 least-squares 融合。TabPFN 这部分结果仅用于说明 prior-integrated fusion 具有一定的模型可迁移性，同时也用于展示不同 foundation model 在 GS 中的适配程度并不相同。

## 13. 当前正式入口与内部复现备注

以下内容主要用于项目内部复现与稿件整理，不建议直接写入论文正文，但可作为方法实现的对应参考。

- `no_prior` 正式入口：`scripts/run_54_duli_liudang_noprior_server_gpu1.sh`
- `baseline` 正式入口：`scripts/run_54_duli_liudang_baseline_server_gpu1.sh`
- `fusion` 正式入口：`scripts/run_54_duli_liudang_fusion_server_gpu1.sh`
- `full pipeline`：`scripts/run_54_duli_liudang_full_server_gpu1.sh`
- 解偶复用融合脚本：`scripts/run_decoupled_prior_fusion_from_archives.py`
- 样本量实验入口：`scripts/run_54_sample_size_decoupled_trait.py`
- SNP 数量实验入口：`scripts/run_54_marker_count_decoupled_trait.py`
- TabPFN 补充验证入口：`scripts/run_54_tabpfn_10k_8traits_trait.py`

当前正式服务器环境为 `server@GPU1`，正式代码目录为 `/home/server/code/git/TabICLv2-test`，正式 Python 为 `/data/yes/envs/TabICLv2-GS/bin/python`。所有正式结果均以服务器目录为准，本地目录主要用于阅读、整理、汇总与撰写。
