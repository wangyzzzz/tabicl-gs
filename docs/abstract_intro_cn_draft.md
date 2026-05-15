# Title + Abstract + Introduction 中文草稿

## 题目候选

### 候选 1
统计先验融合释放 table foundation model 在 genomic selection 中的预测价值

### 候选 2
Statistical-prior integration unlocks the predictive value of table foundation models for genomic selection

### 候选 3
Foundation models contribute to genomic selection through integration with statistical priors

## 中文摘要

基因组选择（genomic selection, GS）长期由 `GBLUP`、Bayes 家族和核方法等统计模型主导，但近年来兴起的 table foundation model 是否能够在 GS 中形成稳定价值，仍缺乏系统证据。尤其在 GS 常见的“小样本、高维标记、遗传结构复杂”条件下，foundation model 未必能够单独稳定超过强统计模型。相比简单追问其能否直接替代经典方法，一个更关键的问题是：当 foundation model 不能单独占优时，它是否仍能作为补充信息源，与现有统计遗传先验结合后提高预测准确率。

为回答这一问题，我们以 `TabICL` 为代表性 table foundation model，在统一的解偶复用框架下系统比较了 `no_prior-TabICL`、`single-prior-TabICL`、`triple-prior-TabICL` 与三条正式 baseline（`BayesB`、`GBLUP`、`RKHS`）在多数据集、多性状 GS 任务中的表现。主结果基于排除 `pig3534` 后的 `36` 个非猪 trait，并通过严格分离 `inner out-of-fold` 与 `outer test` 的方式学习 trait 级融合权重；同时，我们进一步评估了样本量变化、SNP 数量变化以及额外 backbone（`TabPFN`）条件下的结果边界与框架可迁移性。

结果表明，`TabICL` 单独用于 GS 时具有竞争力，但并不是稳定最优。`no_prior-TabICL` 在若干 trait 上能够超过 `BayesB`、`GBLUP` 或 `RKHS`，但整体平均 Pearson (`0.652`) 仍低于三条 baseline (`0.682`、`0.676` 和 `0.684`)。相比之下，与统计 prior 融合后，`TabICL` 的预测价值得到了更稳定的释放。三条 `single-prior-TabICL` 相对各自 prior-only 的平均提升分别为 `1.32%`、`1.62%` 和 `0.82%`；`triple-prior-TabICL` 则进一步成为整体最稳健的主结果，相对 `BayesB`、`GBLUP` 和 `RKHS` 的平均提升分别达到 `1.70%`、`2.61%` 和 `1.42%`，并相对 `only-triple-prior` 继续提高 `0.93%`。权重分析显示，`TabICL` 在融合中并未退化为可以忽略的小修正项，而是在不同 trait 上持续保留稳定但异质的贡献。进一步的样本量和 SNP count 实验表明，这种收益具有明显的 trait 依赖性和资源条件依赖性；`TabPFN` 的补充验证则说明，该融合框架具有一定可扩展性，但不同 foundation model 的适配程度并不相同。

综上，foundation model 在 GS 中的价值未必首先体现为“单独替代经典统计模型”，而更可能体现为“与统计遗传 prior 结合后释放补充信息”。本文因此提供的，不只是一个关于 `TabICL` 的经验结果，也是一条更一般的方法学路径：对于那些难以直接在 GS 中单独稳定奏效的非 GS 原生模型，与其要求它们先全面超越传统方法，不如探索它们如何与已有统计遗传先验协同工作，从而在 GS 中形成可重复、可量化的预测增益。

## 中文引言

基因组选择（genomic selection, GS）已经成为现代动植物育种中最核心的预测框架之一，其目标是在基因型信息可得而表型信息有限的条件下，尽可能准确地预测个体的遗传潜力。过去二十余年里，GS 的主干方法主要由具有明确遗传假设的统计模型构成，其中 `GBLUP` 通过全基因组关系矩阵描述多基因背景效应，Bayes 家族强调对稀疏大效应位点的选择性建模，而 `RKHS` 等核方法则用于吸收更复杂的非线性和高阶相似性结构。这些模型之所以长期占据中心位置，并不仅仅因为它们预测性能强，更因为它们与育种学对遗传结构的理解高度一致。

与此相对，foundation model 及更广义的 representation learning 模型正快速扩展到越来越多的数据模态和预测任务。尤其在表格学习中，这类模型展现出较强的表示能力和迁移潜力，从而引出一个自然问题：它们能否也进入 GS，并在高维基因型数据上形成新的预测优势？这一设想本身很有吸引力，因为 GS 从形式上看也是典型的高维表格预测任务。然而，GS 与常见 tabular benchmark 之间存在本质差异。GS 常常处于“小样本、超高维、marker 连锁结构复杂”的条件下，目标信号又具有明确的遗传统计背景，因此模型的优劣不仅取决于函数逼近能力，还取决于其能否在不同 trait、不同样本量和不同标记密度下稳定提取与遗传结构相关的有效信息。也正因如此，foundation model 即使在一般表格任务上表现突出，也未必能够在 GS 中单独稳定超过经典统计模型。

这就带来了本文试图回答的关键空白。现有关于 foundation model 或深度表格模型进入 GS 的讨论，往往默认以“能否单独打败经典 baseline”作为主要评价标准。但对于一个先验结构很强、经典方法已经高度成熟的领域而言，这一标准可能过于狭窄。一个新模型即使不能单独成为稳定最优的 predictor，也仍然可能保留一部分现有统计模型尚未完全编码的补充信息。换言之，真正值得追问的问题并不只是“foundation model 能否替代经典 GS 模型”，而是“当它不能单独占优时，是否仍然可以作为新的信息来源，与已有统计遗传先验结合后提高预测准确率”。如果答案是肯定的，那么 foundation model 在 GS 中的价值就不必建立在“先单独超越所有传统模型”之上，而可以体现在与经典模型的协同关系之中。

基于这一考虑，我们将研究重点从“替代”转向“融合”。本文选取 `TabICL` 作为代表性 table foundation model，首先系统评估其在不依赖任何统计先验时的独立表现（`no_prior-TabICL`），以回答它在 GS 中到底处于什么水平；随后检验它与 `BayesB`、`GBLUP` 和 `RKHS` 这三类正式 baseline 之间是否存在可量化的互补性；最后通过一套统一、低额外计算开销并严格控制信息泄露的解偶复用框架，将这种互补性转化为 single-prior 与 triple-prior 设定下的 trait-level 预测增益。在这一框架中，底层只正式训练 `no_prior-TabICL` 与三条统计 baseline，而后续 single、dual、triple 和 only-prior 结果均基于已留档预测直接构建，使不同融合方式之间的差异能够更清楚地归因于“如何组合预测”，而不是“如何重复训练模型”。

围绕这一目标，本文依次回答三个递进问题。第一，`TabICL` 单独进入 GS 后是否已经具备竞争力，以及它与三条经典统计 baseline 的相对位置如何。第二，如果 `TabICL` 不是稳定最优，那么它与不同统计 prior 是否存在可重复的互补性，这种互补性是否足以在 single-prior 和 triple-prior 设定下转化为稳定的准确率提升。第三，如果这种 prior-integrated fusion 确实有效，那么其收益是否具有明显的 trait 依赖性、sample-size 依赖性和 marker-count 依赖性，以及这种思路能否在另一类 table / foundation model（`TabPFN`）上获得一定程度的补充支持。

本文的主要贡献可以概括为四点。第一，我们证明了 table foundation model 单独迁移到 GS 时并非完全无效，但通常也不是稳定最优，这为更克制地理解 foundation model 在 GS 中的角色提供了实证基础。第二，我们提出并验证了一套与统计遗传 prior 结构化融合的框架，使 `TabICL` 的补充信息能够在 GS 中被稳定释放。第三，我们表明 single-prior 融合已经可以广泛提升对应的 prior-only 结果，而 `triple-prior-TabICL` 则提供了当前整体最稳健的主结果线路。第四，我们进一步提出一个更一般的方法学视角：对于那些并非为 GS 原生设计、也难以直接单独超过经典统计模型的新模型，其价值未必在于“替代”，而可能在于“与已有统计先验结合后共同提升预测性能”。

因此，本文的目标并不是把 `TabICL` 塑造成一个已经单独战胜所有经典 GS baseline 的新模型，而是希望说明另一件同样重要的事：**当 foundation model 直接迁移到 GS 场景时，它未必能够单独稳定奏效；但如果将其视为一个可以与统计遗传先验协同工作的补充信息源，那么它就有可能在 GS 中发挥出实际价值。** 下面我们将依次展示 `TabICL` 单独使用时的表现、其与统计 prior 融合后的稳定收益、融合权重所反映的 trait-dependent 贡献模式，以及这种思路在不同样本量、标记密度和补充 backbone 条件下的边界与可迁移性。
