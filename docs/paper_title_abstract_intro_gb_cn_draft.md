# Genome Biology 口径草稿：Title / Abstract / Introduction

## 1. Title 候选（English）

### Title 1
Integrating statistical genetic priors enables foundation models for genomic selection

### Title 2
Structured integration of statistical priors unlocks foundation models for genomic prediction

### Title 3
A prior-integrated foundation model framework improves genomic selection across diverse traits

### Title 4
Foundation models gain utility in genomic selection through structured fusion with genetic priors

### Title 5
Genomic selection with prior-integrated foundation models

## 2. Abstract（中文草稿）

基因组选择长期依赖 `GBLUP`、Bayes 家族和核方法等统计模型，它们分别代表了对多基因背景、稀疏标记效应和非线性遗传相似性的经典建模方式。与此同时，foundation model 在通用表格任务中展现出较强潜力，但其能否直接迁移到 genomic selection（GS）场景，并稳定优于经典统计遗传方法，仍缺乏系统评估。更关键的是，若这类模型不能单独占优，它们是否仍然能够以另一种方式为 GS 提供价值，仍是一个尚未被充分回答的问题。

在本研究中，我们以 `TabICL` 作为代表性 tabular foundation model，系统评估其在多数据集、多性状 GS 任务中的独立预测能力，以及其与经典统计遗传 prior 融合后的表现。我们首先比较了 `no_prior-TabICL` 与 `BayesB`、`GBLUP` 和 `RKHS` 的预测性能，发现 TabICL 单独应用于 GS 时具有竞争力，但并不是跨性状的稳定最优模型。在此基础上，我们进一步构建了一个 prior-integrated fusion 框架，将 TabICL 与统计遗传 prior 进行结构化融合，并分别评估单 prior 融合和多 prior 融合的效果。

结果表明，TabICL 的主要价值并不在于单独替代经典 GS 模型，而在于它能够与已有统计 prior 形成互补。无论是 `single-prior-TabICL` 还是 `triple-prior-TabICL`，相较于对应的 prior-only 对照，都表现出更稳定的精度提升；其中，多 prior 融合获得了最稳健的整体结果。进一步的权重分析显示，TabICL 在融合中并非简单重复已有 prior 所包含的信息，而是持续提供了可转化为预测增益的附加信号。样本量和标记量分析进一步支持了这一结论，表明该融合思路在不同资源条件下仍具有较好的鲁棒性。

总体而言，本研究说明，foundation model 直接迁移到 GS 领域时往往难以单独稳定发挥作用，但这并不意味着它们对 GS 没有价值。相反，当 foundation model 与统计遗传中的已有先验知识进行结构化结合时，它们可以在 GS 中释放出实际预测价值。该研究因此提供的，不仅是一个关于 TabICL 的结果，也是一条让更多非 GS 原生模型进入 GS 领域的潜在方法路径。

## 3. Introduction（中文草稿）

基因组选择（genomic selection, GS）已成为现代动植物遗传改良的核心工具，其关键目标是利用全基因组标记信息预测个体遗传价值，从而提高选择效率并缩短育种周期。围绕这一任务，统计遗传学界已经发展出一系列成熟方法，包括以 `GBLUP` 为代表的多基因背景模型、以 `BayesB` 为代表的稀疏效应模型，以及以 `RKHS` 为代表的核方法。尽管这些方法建模假设不同，但都在长期应用中证明了对 GS 的有效性，因此构成了当前 genomic prediction 的主要方法基础。

近年来，foundation model 在自然语言处理、计算机视觉和通用表格学习中展现出强大的表示能力，也推动了其向生物学问题的迁移尝试。在这一背景下，一个自然的问题是：能否将 table foundation model 直接引入 GS，并把它作为新的全基因组预测器来使用？ 这一问题之所以值得回答，是因为 GS 本质上同样属于高维表格预测任务；如果 foundation model 的表示能力能够迁移到这一场景，它将为 genomic prediction 提供新的建模路径。

然而，GS 与常规 tabular benchmark 也存在显著差异，包括相对有限的样本量、极高维的标记空间以及复杂而异质的遗传结构。因此，在通用任务中表现优异的 table foundation model，未必能够在 GS 中直接转化为稳定优势。对于这一类模型，首先需要回答的问题不是它是否“看起来先进”，而是它能否在 GS 中作为独立预测器真正站住脚。与此同时，如果答案是否定的或并不稳定，那么另一个同样重要的问题随之出现：一个不能稳定单独胜出的 foundation model，是否仍然可以作为新的信息来源，与已有统计遗传 prior 形成互补，并最终为 GS 带来增益？

本文正是沿着这一逻辑展开。我们以 `TabICL` 作为代表性 table foundation model，首先系统评估其在多数据集、多性状 GS 任务中的独立表现，以回答 table foundation model 是否能够单独在 GS 中建立稳定竞争力；在此基础上，再进一步引入 `BayesB`、`GBLUP` 和 `RKHS` 等经典统计遗传 prior，构建单 prior 与多 prior 融合框架，检验 TabICL 是否能够在与既有先验结合后释放额外价值。这样的设计使本文的叙事具有清晰的递进关系：先问“table foundation model 能否直接进入 GS”，再问“如果不能稳定单独取胜，它应当如何进入 GS”。

因此，本文的目标不是预设 foundation model 必须替代经典统计模型，而是检验一个更具现实意义的命题：**即使 table foundation model 难以在 GS 中单独稳定占优，只要它能够与统计遗传中的已有先验知识进行有效结合，仍然可能在 GS 中产生稳定而实际的预测增益。** 如果这一点成立，那么本文提供的就不仅是一个关于 `TabICL` 的案例，也是一条让更多非 GS 原生模型进入 GS 领域的潜在方法路径。

## 4. 当前推荐使用方式

- 如果投 `Genome Biology`，标题建议优先用更一般的方法学表述，不要把标题写得过于像单一 benchmarking。
- 摘要核心句要始终围绕：
  - `foundation model alone is not stably optimal in GS`
  - `but structured integration with statistical priors unlocks practical value`
- 引言不要急着讲太多实现细节，要先把“替代”转向“融合”的问题意识立住。
