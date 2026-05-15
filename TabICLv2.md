# TabICLv2 论文中文草稿

## 题目候选

### 候选 1

Statistical priors unlock the predictive value of table foundation models for genomic selection

### 候选 2

Integrating table foundation models with statistical genetic priors improves genomic selection



## 中文摘要

基因组选择（genomic selection, GS）是现代育种中的核心预测任务。长期以来，GS 主要由具有明确统计遗传先验的模型支撑，包括基于全基因组加性多基因假设的 `GBLUP`、通过先验分布刻画 marker effect 异质性的 Bayes 家族模型，以及通过核函数刻画非线性基因组相似性的 `RKHS`。近年来，table foundation model 在通用表格任务中展现出较强的预测准确性和表示能力，为 GS 提供了新的模型来源。我们以 `TabICL` 为代表性 table foundation model，首先系统评估其作为 standalone predictor 在 `36` 个 trait 上的表现。结果显示，`no-prior TabICL` 具有局部竞争力，但不是跨 trait 的稳定优势；其平均准确率相对 `BayesB`、`GBLUP` 和 `RKHS` 分别低 `4.41%`、`3.56%` 和 `4.62%`。这一结果进一步引出本文的核心问题：当 table foundation model 单独使用不能稳定占优时，是否可以通过吸收统计遗传先验释放更稳定的补充预测价值。为此，我们进一步比较三条 `single-prior TabICL`、一条 `triple-prior TabICL` 以及三条 baseline。与统计 prior 融合后，`TabICL` 的补充预测价值得到更稳定体现：三条 `single-prior TabICL` 相对各自 prior-baseline 的平均提升为 `1.32%`、`1.62%` 和 `0.82%`；`triple-prior TabICL` 相对 `BayesB`、`GBLUP` 和 `RKHS` 分别提升 `1.70%`、`2.61%` 和 `1.42%`。相对于每个 trait 内准确率最高的 strongest baseline，`triple-prior TabICL` 在 `30/36` 个 trait 上胜出，平均提升 `0.85%`。进一步的样本量和 marker count 实验显示，在训练样本比例和 SNP 数量变化时，prior-integrated fusion 仍整体保持稳定的正向提升趋势。我们进一步引入另一类 table foundation model `TabPFN` 进行补充验证，也观察到类似现象：foundation model 单独用于 GS 时未必稳定超过强统计模型，但与统计 prior 融合后仍可在 prior 之上进一步提高预测准确率。因此，本文提出的核心观点是：当 foundation model 或其他非 GS 原生模型进入 GS 这类强先验领域时，真正可扩展的应用路径不是绕开 GS 领域先验知识、试图直接解决 GS 预测问题，而是在尊重并利用统计遗传先验的基础上，将其通用表征能力转化为 GS 中可验证的预测增益。

## 中文引言

基因组选择（genomic selection, GS）已经成为现代动植物育种中最重要的预测框架之一，其目标是在基因型信息可得而表型信息有限的条件下，尽可能准确地预测个体的遗传潜力。【需引用：GS 概念奠基文献、GS 在动植物育种中的综述或高水平应用综述】然而，在经典统计模型已经非常成熟的背景下，如何在不同数据集和不同 trait 中进一步获得稳定且可重复的预测增益，仍然是 GS 方法研究中的核心挑战之一。长期以来，GS 主要由三类具有明确统计遗传先验的模型占据：`GBLUP` 通过全基因组关系矩阵刻画加性背景，Bayes 家族模型更适合表达稀疏或不均匀的位点效应，而 `RKHS` 则为非线性和高阶基因组相似性提供了更灵活的表达。【需引用：GBLUP/GRM 原始或经典方法文献；BayesA/BayesB/Bayesian Lasso 经典文献；RKHS 在 genomic prediction 中的经典文献】这些方法之所以长期构成 GS 的主力，主要因为它们分别对应了不同的遗传建模假设。换言之，对于 GS 而言，预测性能不仅取决于模型能力，也取决于模型是否能够利用并适配 trait 背后的统计遗传先验。【需引用：统计遗传先验、trait genetic architecture 与 genomic prediction accuracy 关系的综述或高水平实证研究】

与此同时，近年来 table foundation model 在通用表格学习任务中展现出较强的预测准确性、迁移能力和函数表示能力，为 GS 提供了一个值得检验的新模型来源。【需引用：TabPFN、TabICL 或其他 table foundation model 原始论文；tabular foundation model/表格学习 benchmark 高水平论文】这一设想是有吸引力的，因为基因型数据在形式上确实可以被视为一种高维表格输入，而 foundation model 可能捕捉到经典统计模型未完全表达的数据驱动模式。然而，GS 与常规 tabular benchmark 之间存在关键差异：GS 往往同时具有样本量有限、marker 维度极高、群体结构明显、遗传信号稀疏且 trait-specific 等特征；这些任务特点使得一般表格任务中的模型优势并不会自动转化为 GS 中的稳定优势。【需引用：GS 小样本高维、群体结构、marker density、trait-specific genetic architecture 相关综述；机器学习/深度学习用于 genomic prediction 的综述或 benchmark】因此，在 GS 中引入 table foundation model 时，首先需要回答一个最直接的问题：它作为 standalone predictor 能否在 GS 中建立稳定竞争力。

在这一问题得到系统评估之后，才有必要进一步讨论它与统计遗传先验之间的关系。对于 GS 这样一个统计先验高度成熟的领域，一个非 GS 原生模型即使不能单独稳定占优，也可能保留一部分现有统计模型尚未完全编码的数据驱动信号。因而，进一步的关键问题是：当 table foundation model 的 standalone 表现受限时，是否能够通过与 `GBLUP`、Bayes 模型或 `RKHS` 等统计 prior 结合，将这部分信号转化为更稳定的预测增益。【需引用：统计模型与机器学习/深度学习互补、ensemble/stacking/fusion 在 genomic prediction 或预测建模中的方法文献】如果答案是肯定的，那么 foundation model 在 GS 中的价值就不必建立在“全面优于经典统计模型”之上，而可以体现在与统计遗传先验的协同关系之中。

基于这一递进逻辑，本文以 `TabICL` 作为代表性 table foundation model，系统评估其在 GS 中的独立价值与融合价值。【需引用：TabICL 原始论文或技术报告；如没有正式期刊论文，需要引用可公开版本并在 Methods 中说明实现来源】我们首先考察不引入任何统计先验的 `TabICL`，以判断 table foundation model 单独用于 GS 时是否具备竞争力；随后将其与三条具有不同遗传假设的正式 baseline（`BayesB`、`GBLUP` 和 `RKHS`）结合，分别构建 single-prior 与 triple-prior 融合结果，以检验 `TabICL` 是否能够在统计 prior 之上提供补充增益。

围绕这一框架，本文依次回答三个问题。第一，`TabICL` 单独用于 GS 时是否具有竞争力，以及它相对 `BayesB`、`GBLUP` 和 `RKHS` 的真实位置如何。第二，当 `TabICL` 不是稳定最优的 standalone predictor 时，它是否仍能在 single-prior 与 triple-prior 融合中稳定提升统计 prior 的预测准确率，并在权重结构中保留可解释的贡献。第三，这种 prior-integrated fusion 的收益是否受到 trait 类型、样本量、marker density 和 foundation model backbone 的影响。通过这些分析，本文旨在检验一个更一般的方法学问题：当 foundation model 或其他非 GS 原生模型进入 GS 这类强先验领域时，能否在尊重并利用统计遗传先验的基础上，将其通用表征能力转化为 GS 中可验证的预测增益。





## Result 1. Table foundation model 可以进入 GS，但单独使用并不是稳定最优

我们首先评估了不引入统计 prior 的 `TabICL`，即 `no-prior TabICL`，作为独立预测器在 GS 场景中的表现，以回答一个最直接的问题：table foundation model 是否能够不依赖任何统计遗传先验，单独在 GS 中建立竞争力。基于 `36` 个 trait，`no-prior TabICL` 的平均 Pearson 为 `0.652`。如果分别与三条正式统计 baseline 比较，`BayesB`、`GBLUP` 和 `RKHS` 的平均 Pearson 分别约为 `0.677`、`0.671` 和 `0.677`；对应地，`no-prior TabICL` 相对三者的逐 trait 平均相对差异分别为 `-4.41%`、`-3.56%` 和 `-4.62%`。也就是说，无论与哪一类统计模型相比，`TabICL` 单独使用时都没有形成稳定的平均优势（Table S1; Table S2）。

这一结果说明，`TabICL` 单独进入 GS 时总体上仍弱于三条 baseline，但这种平均层面的劣势并不意味着它在所有 trait 上都缺乏价值。恰恰相反，在若干 trait 上，`no-prior TabICL` 已经表现出明确的局部优势。例如，在 `rice529/grain_weight` 上，`no-prior TabICL` 的 Pearson 为 `0.5741`，高于 `BayesB` 的 `0.5619`、`GBLUP` 的 `0.5554` 和 `RKHS` 的 `0.5564`，对应相对提升分别约为 `+2.18%`、`+3.38%` 和 `+3.19%`；在 `rice529/plant_height` 上，`no-prior TabICL` 达到 `0.8464`，同样高于 `BayesB` 的 `0.8427`、`GBLUP` 的 `0.8427` 和 `RKHS` 的 `0.8386`，相对提升约为 `+0.44%`、`+0.44%` 和 `+0.93%`；在 `wheat406/pl_e1` 上，`no-prior TabICL` 为 `0.8581`，也高于 `BayesB` 的 `0.8433`、`GBLUP` 的 `0.8293` 和 `RKHS` 的 `0.8429`，对应提升约为 `+1.76%`、`+3.48%` 和 `+1.80%`（Table S3）。这些例子说明，尽管 `TabICL` 作为单独预测器尚未形成跨 trait 的稳定优势，但它确实能够在部分性状上恢复出高于经典统计模型的有效预测信号。但从整体稳定性看，这种优势仍然有限：`no-prior TabICL` 相对 `BayesB`、`GBLUP` 和 `RKHS` 的逐 trait 胜率分别仅为 `4/36`、`5/36` 和 `3/36`（Table S2）。因此，Result 1 支持这样一个判断：将 table foundation model 引入 GS 是有意义的，但若希望它单独成为一个稳定优于经典统计模型的 predictor，目前证据仍不充分；这也正构成了后续引入统计 prior 的直接动机。

## Result 2. 与统计 prior 融合后，TabICL 可以稳定提升预测性能

在确认 `no-prior TabICL` 难以单独稳定胜出的基础上，我们进一步检验：如果将 `TabICL` 与经典统计遗传 prior 结合，它是否能够释放出更稳定的预测增益。我们首先考察三种 single-prior 融合，即 `BayesB-prior TabICL`、`GBLUP-prior TabICL` 和 `RKHS-prior TabICL`。结果显示，三种 single-prior TabICL 都能够在 trait 层面带来广泛的正向提升。相对于各自 prior-only 对照，`BayesB-prior TabICL`、`GBLUP-prior TabICL` 和 `RKHS-prior TabICL` 的平均相对提升分别为 `1.32%`、`1.62%` 和 `0.82%`；其中，正提升 trait 数分别为 `35/36`、`35/36` 和 `30/36`（Table S4; Table S5）。这说明 `TabICL` 并不是简单重复已有 prior 所编码的信息，而是能够提供额外预测信号。

这些 single-prior 融合的收益在若干 trait 上尤其明显。以 `wheat406/sl_e1` 为例，`BayesB-prior TabICL` 相对 `BayesB` 提升 `4.98%`，`GBLUP-prior TabICL` 相对 `GBLUP` 提升 `5.69%`，`RKHS-prior TabICL` 相对 `RKHS` 提升 `3.71%`；在 `rice529/grain_weight` 上，三条 single-prior TabICL 也都表现出一致增益，分别相对各自 prior 提升 `3.61%`、`4.52%` 和 `3.96%`。此外，`rice529/heading_date` 与 `wheat406/pl_e1` 也是较有代表性的正例：前者中 `BayesB-prior TabICL` 和 `GBLUP-prior TabICL` 分别相对对应 prior 提升 `2.54%` 和 `3.37%`，后者中三条 single-prior TabICL 分别达到 `2.53%`、`3.67%` 和 `2.42%` 的提升（Table S6）。换言之，single-prior TabICL 的价值首先体现为：**无论 prior 是 `BayesB`、`GBLUP` 还是 `RKHS`，只要将 `TabICL` 纳入融合，通常都能够把该 prior 本身再抬高一步。**

在此基础上，我们进一步评估多 prior 融合的主结果 `triple-prior TabICL`。如果先与三条正式 baseline 逐一比较，其优势已经非常清楚：相对于 `BayesB`，`triple-prior TabICL` 在 `36/36` 个 trait 上全部更优，平均相对提升为 `1.70%`；相对于 `GBLUP`，同样达到 `36/36` 个 trait 全部更优，平均相对提升为 `2.61%`；相对于 `RKHS`，则在 `30/36` 个 trait 上更优，平均相对提升为 `1.42%`（Table S7; Table S8）。这说明 triple 融合的收益并不只是体现在“偶尔超过某一条 baseline”上，而是相对于三类经典统计建模假设都保持了稳定的整体优势，尤其对 `BayesB` 和 `GBLUP` 呈现出几乎一致的全面提升。

如果进一步与三条 single-prior TabICL 比较，`triple-prior TabICL` 依然表现出更高的整体稳健性。相对于 `BayesB-prior TabICL`、`GBLUP-prior TabICL` 和 `RKHS-prior TabICL`，`triple-prior TabICL` 的平均相对提升分别为 `0.37%`、`0.97%` 和 `0.59%`；对应地，它分别在 `21/36`、`33/36` 和 `27/36` 个 trait 上不低于这三条 single-prior 结果。与此同时，`triple-prior TabICL` 相对不含 `TabICL` 的 `only-triple prior` 平均提升为 `0.93%`（Table S8; Table S11），说明 triple 的增益并不只是来自三条统计 baseline 的线性重组，而是在 triple prior 已经形成的组合基础上，`TabICL` 仍然继续提供了可转化为精度的补充信息。换言之，triple 的价值并不只是“比 prior-only 更强”，而是在多数情况下还能进一步超过单一 prior 融合，并在 `only-triple prior` 之上继续取得稳定增益。

在上述比较基础上，如果再以 strongest baseline 作为综合参照，即对每个 trait 分别取 `BayesB`、`GBLUP` 和 `RKHS` 中 Pearson 最高的一条正式 baseline，那么 `triple-prior TabICL` 仍然是当前主结果中整体最稳健的方法。在 `36` 个 trait 中，`triple-prior TabICL` 有 `30` 个 trait 超过 strongest baseline，平均相对提升为 `0.85%`。但需要强调的是，`triple-prior TabICL` 并不是在每个 trait 上都无条件最优。它在 `rice529/num_panicles`、`rice529/yield`、`soybean951/fa16c`、`soybean951/md`、`soybean951/prt` 和 `soybean951/vbn` 上仍略低于 strongest baseline，降幅约为 `0.07%` 到 `0.75%`（Table S7; Table S8）。因此，triple 的价值更准确地说是“整体最稳健”，而不是“逐 trait 全面垄断”。

按 strongest baseline 的类型分层后，当 strongest baseline 来自 `BayesB` 时，`triple-prior TabICL` 的平均提升为 `+1.10%`，并在 `17/17` 个 trait 上全部超过 strongest baseline；当 strongest baseline 来自 `GBLUP` 时，平均提升为 `+1.02%`，在 `3/3` 个 trait 上全部获胜；当 strongest baseline 来自 `RKHS` 时，平均提升下降为 `+0.56%`，赢 `10/16` 个 trait。当前 `triple-prior TabICL` 未能超过 strongest baseline 的 `6` 个 trait 全部落在 `RKHS-best` 区域（Table S10）。换言之，`triple-prior TabICL` 对 `BayesB-best` 和 `GBLUP-best` trait 的增益最稳定，而在 `RKHS` 主导的 trait 上仍存在剩余难度。这也提示 `TabICL` 所提供的补充信息更容易与加性或稀疏主效应型 prior 形成稳定协同，而面对那些更依赖复杂非线性或高阶相似性结构的 trait 时，其增益释放相对较低。

从整体上看，Result 2 支持全文最核心的经验事实：**TabICL 的价值主要体现在与统计 prior 的结合，而不是单独替代统计模型。** single-prior 融合证明 TabICL 可以稳定抬高单一统计 prior，triple-prior 融合则进一步提供了一条在大多数 trait 上都足够稳健、且无需预先判断哪条 prior 最匹配的统一主结果线路。

## Result 3. Single-prior 融合权重表明，TabICL 在不同 trait 上持续保留了稳定但异质的贡献

Result 2 已经表明，single-prior 和 triple-prior 融合都能够稳定提高预测准确率。接下来的问题不再是“融合是否有效”，而是：在与先验结合时，`TabICL` 究竟占据怎样的位置？由于 single-prior 情形只有“一个统计 prior + 一个 `TabICL`”两个组成部分，其权重含义最直接，因此我们首先聚焦于 single-prior 融合的权重结构，来判断 `TabICL` 在融合中究竟只是一个很小的修正项，还是持续保留了可观贡献（Table S13; Table S14）。

整体上看，single-prior 融合中的 `w_TabICL` 并没有系统性塌缩到接近 `0` 的区域。在 `36` 个 trait 上，`BayesB-prior TabICL`、`GBLUP-prior TabICL` 和 `RKHS-prior TabICL` 三条线的平均 `w_TabICL` 分别为 `0.493`、`0.493` 和 `0.503`，对应范围大致为 `0.433-0.553`、`0.447-0.536` 和 `0.442-0.560`。这意味着在 single-prior 融合中，`TabICL` 并没有被边缘化为一个极小的残差修正项，而是在多数 trait 上都持续保留了接近一半的权重。换言之，single-prior 融合之所以能够相对各自 prior-only 获得系统性提升，并不仅仅是因为统计 prior 本身已经足够强，还因为 `TabICL` 在融合中持续保留了可观且稳定的贡献（Table S13; Table S14）。

更重要的是，这种对 `TabICL` 的保留程度在不同 trait 间并不是固定不变的，而是呈现出明确的异质性。如果把三条 single-prior 结果的 `w_TabICL` 取平均，则其整体范围约为 `0.443-0.544`。较高权重的 trait 包括 `rice529/num_effective_panicles`（平均 `w_TabICL = 0.544`）、`wheat406/pl_e3`（`0.542`）和 `rice529/grain_weight`（`0.539`）；较低权重的 trait 则包括 `soybean951/prt`（`0.443`）、`rice529/spikelet_length`（`0.454`）和 `rice529/grain_length`（`0.457`）。从整体趋势看，single-prior 的平均 `w_TabICL` 与 `no-prior TabICL` 相对 strongest baseline 的表现呈中等正相关（`r = 0.567`）。也就是说，当 `TabICL` 单独使用时本身已经更接近 strongest baseline，在 single-prior 融合中通常也更倾向于保留更高比例的 `TabICL` 成分；反之，当 `TabICL` 单独表现较弱时，融合仍然可以带来增益，但其组合更偏向于依赖统计 prior。这说明 `TabICL` 在融合中的作用不是固定常数，而是随 trait 改变的：它在某些 trait 上保留更高份额，在另一些 trait 上则更多作为次级修正项参与组合（Table S13; Table S15）。

在此基础上，triple-prior 的权重结果可以作为一个一致性补充，而不必像 single-prior 那样被直接展开解释。由于 triple-prior 采用的是 two-step OLS，`w_TabICL` 所对应的是“`TabICL` 相对于 prior aggregate”的权重，而不是相对于某一个单独 baseline 的直接比例，因此其含义天然比 single-prior 更复杂。但即便如此，triple 融合中的 `w_TabICL` 仍然保持在一个稳定且不可忽略的区间内：在 `36` 个 trait 中，其平均值为 `0.489`，范围约为 `0.434-0.549`。同时，triple 内部三条 prior 的 share 也整体保持均衡：`BayesB` prior share 的均值为 `0.334`，范围约为 `0.328-0.341`；`GBLUP` prior share 的均值为 `0.331`，范围约为 `0.324-0.334`；`RKHS` prior share 的均值为 `0.335`，范围约为 `0.329-0.348`。这说明在更复杂的多 prior 设定下，融合框架同样不是通过机械平均来工作，而是在 prior 侧和 `TabICL` 侧都进行了 trait-dependent 的重分配。因而，Result 3 更稳妥地支持这样一个判断：**`TabICL` 在 OLS 融合中并不是一个可以忽略的小项，而是在不同 trait 上持续保留了稳定但异质的贡献；single-prior 的权重结构最清楚地揭示了这一点，而 triple-prior 的结果则从更复杂的设定中给出了方向一致的支持。**（Table S13; Table S14; Table S16; Table S17）

## Result 4. 样本量变化表明，多 prior 融合的收益具有明显的 trait 依赖性与 sample-size 依赖性

在完成主结果之后，我们进一步考察融合收益是否会随着训练样本规模变化而发生系统性改变。当前样本量分析在4个数据集上各选择了 `2` 个 trait，即 `cotton_fibelo`、`cotton_fiblen`、`grain_weight`、`grain_width`、`bbd`、`lw`、`sl_e1` 和 `sl_e2`，并比较了 `20%`、`60%` 和 `100%` 样本量下 `no-prior TabICL`、三条 baseline、三条 single-prior TabICL 以及 `triple-prior TabICL` 的表现，以检验多 prior 融合的收益是否具有 sample-size-dependent 特征（Table S18; Table S19）。

从绝对准确率看，样本量增加会同时抬升 `no-prior TabICL`、baseline 和 fusion 的表现，而不是只让某一类方法单独受益。在当前 `8` 个 trait 上，`no-prior TabICL` 的平均 Pearson 从 `20%`、`60%` 到 `100%` 分别为 `0.4867`、`0.5827` 和 `0.6097`；`BayesB`、`GBLUP` 和 `RKHS` 的平均 Pearson 也分别从 `0.5173 / 0.5073 / 0.5200` 升至 `0.5982 / 0.5913 / 0.5934`，并在 `100%` 时进一步达到 `0.6314 / 0.6208 / 0.6271`。三条 single-prior TabICL 同样同步上升：`BayesB-prior TabICL`、`GBLUP-prior TabICL` 和 `RKHS-prior TabICL` 的均值在 `20%` 时分别为 `0.5226`、`0.5191` 和 `0.5198`，在 `60%` 时为 `0.6096`、`0.6044` 和 `0.6045`，在 `100%` 时为 `0.6417`、`0.6341` 和 `0.6355`；`triple-prior TabICL` 则对应为 `0.5237`、`0.6086` 和 `0.6435`，这说明所有方法都会随着样本量增加而共同变强（Table S19; Table S20）。

从整体均值看，`triple-prior TabICL` 在三个样本量档位下都保持了相对三条 baseline 的正向增益，但这种增益并不是对所有 baseline 以同一种方式变化。相对 `BayesB`，`triple-prior TabICL` 在 `20%`、`60%` 和 `100%` 下的平均提升分别为 `+1.40%`、`+2.07%` 和 `+2.34%`；相对 `GBLUP` 的平均提升分别为 `+4.28%`、`+3.24%` 和 `+4.13%`；相对 `RKHS` 则分别为 `+0.63%`、`+2.81%` 和 `+2.71%`。若进一步以 trait 内 strongest baseline 作为综合参照，`triple-prior TabICL` 在 `20%`、`60%` 和 `100%` 下的平均相对提升分别为 `+0.25%`、`+1.79%` 和 `+1.55%`，对应胜出 trait 数为 `5/8`、`7/8` 和 `8/8`。换言之，随着训练样本增加，`triple-prior TabICL` 逐步从“部分 trait 上已可见的边际优势”发展为“在大多数甚至全部 trait 上均不低于 strongest baseline 的更稳定表现”（Table S19; Table S20）。

trait-level 轨迹进一步说明，上述整体增益并不对应所有 trait 的一致改善。部分 trait 的融合收益在小样本阶段就已出现，例如 `rice529/grain_weight` 在 `20%` 样本量下，`triple-prior TabICL` 已达到 `0.3544`，略高于 strongest baseline `RKHS` (`0.3533`)，到 `100%` 时进一步提升至 `0.5821`。但也有一些 trait 的优势释放更晚，例如 `rice529/grain_width` 和 `soybean951/lw` 在 `20%` 下尚未超过 strongest baseline，而在更高样本量下才逐步转为占优。`wheat406/sl_e1` 和 `sl_e2` 则进一步表明，小样本下并不一定立即出现正增益：其中 `sl_e1` 在 `20%` 时相对 `BayesB` 仍为负增益，但到 `100%` 时二者都转为约 `+5%` 的提升。因此，样本量增加带来的并不是所有 trait 上同步扩大的收益，而是不同 trait 上融合优势释放时机和幅度的差异（Table S18; Table S20）。

综合来看，当前 8-trait 的样本量实验表明，多 prior 融合的收益并不存在一条对所有 trait 都成立的统一规律。随着样本量增加，`triple-prior TabICL` 在整体均值和胜出 trait 数上的表现趋于更稳定，但这并不意味着融合收益会随样本量单调扩大。相反，不同 trait 的增益释放时机和幅度存在明显差异：有些 trait 在小样本阶段就已受益，有些则需要更充足的数据后才表现出稳定优势。

## Result 5. 标记密度提升增强了融合框架的整体性能上限，但增益模式并非对所有 baseline 同步扩张

与样本量实验互补，我们进一步考察了标记密度变化下融合框架的表现，以检验 TabICL 与统计 prior 的互补性是否会随着 SNP 信息增加而保留甚至增强。该分析基于 `4` 个数据集、`8` 个代表性 trait，并设置 `2K`、`10K` 和 `50K` 三档 marker count（Table S22; Table S23）。

从绝对准确率看，marker 数增加会同时抬升 `no-prior TabICL`、baseline 和 fusion 的表现，而不是只让某一类方法单独受益。当前 `8` 个 trait 上，`no-prior TabICL` 的平均 Pearson 从 `2K`、`10K` 到 `50K` 分别为 `0.5890`、`0.6097` 和 `0.6195`；`BayesB`、`GBLUP` 和 `RKHS` 的平均 Pearson 也分别从 `0.6028 / 0.6017 / 0.6147` 提升到 `0.6314 / 0.6208 / 0.6271`，并在 `50K` 时进一步达到 `0.6599 / 0.6328 / 0.6352`。三条 single-prior TabICL 同样整体上升：`BayesB-prior TabICL`、`GBLUP-prior TabICL` 和 `RKHS-prior TabICL` 的均值分别由 `0.6156 / 0.6146 / 0.6185` 提升至 `0.6417 / 0.6341 / 0.6355`，并在 `50K` 时达到 `0.6674 / 0.6461 / 0.6457`；`triple-prior TabICL` 则从 `0.6201` 进一步升至 `0.6435` 和 `0.6675`。因此，更高 marker 密度抬升的是整个比较框架的准确率上限，而不是仅仅制造某一种方法的优势。与此同时，`no-prior TabICL` 在三档 marker 下的平均表现仍低于三条正式 baseline，这也再次说明，table foundation model 单独进入 GS 虽然具有一定竞争力，但其更稳定的价值仍然来自与统计 prior 的结合（Table S22; Table S23）。

`triple-prior TabICL` 相对 `BayesB` 的优势总体保持为正，但会随着 marker 数增加而逐渐收敛，说明当标记更加丰富时，`BayesB` 本身也能更充分地提取稀疏大效应信息。例如，`cotton_fibelo` 上，`no-prior TabICL` 在 `2K / 10K / 50K` 下分别为 `0.6212 / 0.6324 / 0.6245`，`BayesB` 对应为 `0.6269 / 0.6595 / 0.6921`，`BayesB-prior TabICL` 为 `0.6381 / 0.6681 / 0.6950`，`triple-prior TabICL` 为 `0.6437 / 0.6667 / 0.6941`。可以看到，随着 marker 数提高，融合结果仍然保持在最高区间，但相对 `BayesB` 的边际优势明显缩小。`grain_weight` 也体现出类似但更强的 trait 依赖性：在 `2K` 和 `10K` 下，`triple-prior TabICL` 分别达到 `0.5743` 和 `0.5821`，高于 `BayesB` 的 `0.5483` 和 `0.5619`；但在 `50K` 下，`BayesB` 升至 `0.5828`，反而高于 `triple-prior TabICL` 的 `0.5758`。这说明相对 `BayesB` 的平均提升从 `+2.92%`、`+2.34%` 收敛到 `+1.48%`，并不是因为 fusion 失效，而是因为在部分 trait 上 `BayesB` 对高密度 marker 的利用效率更高（Table S24; Table S26）。

相比之下，`triple-prior TabICL` 相对 `GBLUP` 和 `RKHS` 的优势在三档 marker 下始终为正，并且在 `50K` 时进一步扩大到 `+6.47%` 和 `+5.64%`。这种趋势同样可以从绝对准确率轨迹中直接看到。以 `soybean951/lw` 为例，`no-prior TabICL` 从 `0.4597` 上升到 `0.5017` 和 `0.5571`，`GBLUP` 与 `RKHS` 则分别从 `0.4927 / 0.5028` 上升到 `0.5259 / 0.5318`，并在 `50K` 时维持在 `0.5421 / 0.5418`；与之对应，`BayesB-prior TabICL` 从 `0.5057` 增至 `0.5561` 和 `0.6593`，`triple-prior TabICL` 也从 `0.5062` 提升到 `0.5561` 和 `0.6593`。也就是说，更丰富的 marker 信息并没有削弱融合框架的作用，反而使其相对某些统计模型的互补收益更充分地显现。`cotton_fibelo` 也表现出类似模式：尽管 `BayesB` 在 `50K` 下非常强，但 `triple-prior TabICL` 相对 `GBLUP` 和 `RKHS` 的优势依然由 `2K` 时的有限正增益扩展到 `50K` 时的更高水平（Table S24; Table S25）。

单 prior 融合也呈现出一致但不完全相同的规律，说明 richer marker information 并不是只被 triple 融合独占利用。三条 `single-prior TabICL` 相对各自 prior 的平均增益在全部 marker 档位上均保持为正：`BayesB-prior TabICL` 相对 `BayesB` 的平均提升为 `+2.06%`、`+1.92%` 和 `+1.45%`，表现出与 triple 相似的收敛趋势；`GBLUP-prior TabICL` 相对 `GBLUP` 的平均提升则为 `+2.14%`、`+2.45%` 和 `+2.57%`，是三条 single-prior 线路中最稳定的一条；`RKHS-prior TabICL` 相对 `RKHS` 的增益在低 marker 密度下较小（`+0.51%`），但在 `50K` 时提升到 `+1.95%`。例如，在 `wheat406/sl_e2` 上，`RKHS` 从 `0.3992`、`0.4367` 提升到 `0.4431`，`RKHS-prior TabICL` 则对应为 `0.4013`、`0.4381` 和 `0.4431`，说明当 trait 更匹配 `RKHS` 这类 prior 时，single-prior 融合的增益可能较小但仍保持不劣；而在 `lw` 上，`BayesB-prior TabICL` 则从 `0.5057`、`0.5561` 增至 `0.6593`，持续高于对应的 `BayesB`。这说明 `TabICL` 并非只对某一种统计 prior 有效，而是能够在三类统计建模假设上都提供补充信号，只是补充模式并不相同（Table S25; Table S26）。

trait-level 结果进一步说明，marker 数增加的收益应被理解为“整体趋势”，而不是“所有 trait 的单调定律”。在 `10K` 条件下，`triple-prior TabICL` 在 `8/8` 个 trait 上都优于 strongest baseline；在 `2K` 下，这一数字为 `7/8`，唯一例外是 `wheat406/sl_e2`，其 `triple-prior TabICL`（`0.3956`）略低于 `RKHS`（`0.3992`）和 `RKHS-prior TabICL`（`0.4013`）；在 `50K` 下，`triple-prior TabICL` 仍在 `6/8` 个 trait 上超过 strongest baseline，但 `rice529/grain_weight` 与 `grain_width` 两个 trait 被 `BayesB` 反超，其中 `grain_width` 在 `50K` 下的 `BayesB`、`BayesB-prior TabICL` 与 `triple-prior TabICL` 已几乎重合在 `0.8422` 附近。进一步看绝对性能轨迹，`8` 个 trait 中有 `6` 个在 `2K <= 10K <= 50K` 下呈现非下降趋势，仅 `grain_weight`（`0.5743 -> 0.5821 -> 0.5758`）与 `bbd`（`0.8138 -> 0.8215 -> 0.8210`）未完全满足单调上升。由此，更高 SNP 数量通常有利于融合模型表现，但具体收益仍受到 trait 本身遗传结构和 prior 竞争关系的共同影响（Table S23; Table S26）。

综合来看，当前 8-trait SNP count 消融结果表明，更高的 marker 密度通常会提高融合模型的绝对准确率，并在相对 `GBLUP` 与 `RKHS` 的比较中保留更明显的平均增益；但相对 `BayesB` 的优势则更容易收敛，且少数 trait 在高 marker 密度下仍可能由单一强 prior 主导。因此，marker 数增加带来的并不是所有 trait 上一致的单调收益，而是一个同时受 trait 结构与 prior 竞争关系影响的整体趋势（Table S24; Table S25; Table S26）。

## Result 6. TabPFN 补充验证支持融合框架具有一定可迁移性，但其稳定性弱于 TabICL 主线

作为对这一方法学主线的进一步补充，我们还引入了一个额外的 table / foundation model，即 `TabPFN`，以检验 prior-integrated fusion 这一思路是否仅对 `TabICL` 单一 backbone 成立。该补充验证采用 `10K SNP + 8 个 traits` 的设定。需要强调的是，这条 `TabPFN` 结果线并不替代 `TabICL` 主线，也不构成新的主结果，而是用于回答一个更狭义但重要的问题：**多 prior 融合框架是否具有一定的模型可迁移性**（Table S27; Table S28）。

结果显示，`TabPFN` 在当前设定下同样符合全文主线。其 `no-prior TabPFN` 版本的 8-trait 平均 Pearson 为 `0.5986`，低于 `BayesB`（`0.6314`）、`GBLUP`（`0.6208`）和 `RKHS`（`0.6271`），说明即使换用另一类 table / foundation model，单独进入 GS 后的总体表现依然未能稳定超过强统计 baseline。进一步将 `TabPFN` 与单一统计 prior 结合后，三条 single-prior TabPFN 的平均 Pearson 分别为 `0.6364`、`0.6315` 和 `0.6309`，相对各自 baseline 的平均提升分别为 `+0.77%`、`+1.79%` 和 `+0.42%`，表明它也能够从统计 prior 中获得一定补偿（Table S28; Table S29; Table S30）。

在此基础上，`triple-prior TabPFN` 的平均 Pearson 为 `0.6352`。其相对 `BayesB`、`GBLUP` 和 `RKHS` 的平均提升分别为 `+0.6315%`、`+2.4058%` 和 `+1.0401%`；相对不含 `TabPFN` 的 `only-triple prior` 也仍有 `+0.3496%` 的平均增益。这说明多 prior 融合框架并不只对 `TabICL` 单一 backbone 有效，而是具有一定的模型可迁移性。但与此同时，`triple-prior TabPFN` 相对 trait 内最优 baseline 的 8-trait 平均提升仍为 `-0.1230%`，仅在 `4/8` 个 trait 上为正，整体稳定性明显弱于 `TabICL` 主线。因此，foundation model 单独用于 GS 时通常未必稳定优于强统计模型，但在多 prior 融合框架中，可以在一定程度上释放补充价值；同时，不同 foundation model 的收益幅度和稳定性并不相同，而当前整体更稳定、结果更完整的仍然是 `TabICL`（Table S28; Table S31）。

# Discussion 中文草稿

本研究的核心发现并不是 `TabICL` 单独进入 genomic selection（GS）后能够稳定超越经典统计模型，而是：即使 table foundation model 不是为 GS 原生设计、也难以在独立应用中形成稳定最优，它仍然可以在与统计遗传先验进行结构化结合后释放出可重复的预测价值。【需引用：机器学习/深度学习在 genomic prediction 中不稳定优于经典统计模型的综述或 benchmark；最好包含作物或多数据集比较】这一区分非常重要，因为它把研究问题从“foundation model 是否能够直接解决 GS 预测问题”转向了“foundation model 是否能够作为新的信息来源，被有效吸收进现有 GS 框架”。从这一意义上说，本文提供的不只是一个关于 `TabICL` 的经验案例，更是一条让非 GS 原生模型进入 GS 领域的可行方法路径。

这一判断首先建立在一个相对克制但更有说服力的事实之上：`TabICL` 单独使用时并非没有价值，但其优势是 trait-dependent 的，而不是跨 trait 的稳定压制。当前主结果显示，`TabICL` 在 `rice529/grain_weight`、`rice529/plant_height` 和 `wheat406/pl_e1` 等 trait 上已经能够超过 `BayesB`、`GBLUP` 或 `RKHS`，说明 table foundation model 确实能够从基因型数据中捕捉到有效预测信号；但从整体平均水平看，它仍未能稳定超过这三类经典 statistical baselines。这一点意味着，如果把评价标准简单设定为“是否直接打败经典模型”，那么像 `TabICL` 这样的模型很容易被过早判定为在 GS 中“价值有限”。而本文的结果恰恰表明，这种判断可能过于狭窄。对于 GS 这类先验结构非常强、而且已有模型长期积累的方法领域，一个新模型的价值未必首先体现为压过现有模型，也可能体现为补充现有模型尚未完全编码的信息。【需引用：GS 中 trait-dependent prediction accuracy、模型表现依赖 trait genetic architecture 的高水平综述或多数据集研究】

这种补充价值在与统计 prior 融合后得到了更清楚的体现。无论是 `single-prior TabICL` 还是 `triple-prior TabICL`，它们相对于各自 baseline 或 prior-only 的准确率提升都表现出广泛而稳定的正向趋势。特别是 `single-prior` 的结果说明，`TabICL` 并不是只对某一类统计假设有效，而是能够在 `BayesB`、`GBLUP` 和 `RKHS` 三类不同建模范式上都提供增益。这一点很关键，因为它提示 `TabICL` 并非简单复现了某一条统计模型已经编码好的信息。如果它只是在学习某一种 prior 的近似替代，那么其收益理应集中在少数与该 prior 最接近的 trait 或方法上；而当前观察到的是一种更广泛的增益格局（Table S4; Table S5; Table S6）。【需引用：模型融合、stacking、ensemble 可利用不同模型互补信息的经典方法文献；如有 genomic prediction ensemble/fusion 文献，也应加入】

`triple-prior TabICL` 的结果进一步强化了这一判断。与三条正式 baseline 分别比较时，`triple-prior TabICL` 表现出更高的整体稳健性；如果进一步以 strongest baseline 作为逐 trait 参照，它仍然在 `30/36` 个 trait 上保持领先，平均提升为 `0.85%`。换句话说，triple-prior 融合的意义不仅在于进一步提高预测准确率，也在于在无需事先判断哪一种 prior 最匹配某个 trait 的前提下，提供一个整体表现足够靠前、且跨 trait 更稳定的预测器（Table S7; Table S8; Table S10）。对于真实育种任务而言，这种“无需逐 trait 调路线、仍能获得稳定增益”的性质，往往比在少数 trait 上追求极致局部最优更具实践价值。【需引用：GS 在真实育种选择中的应用综述；跨 trait/跨环境预测稳定性的重要性文献】

从机制层面看，single-prior 融合权重提供了一条与性能结果方向一致的证据。`w_TabICL` 在三条 single-prior 线路中并没有系统性塌缩到接近零，而是在大多数 trait 上维持在接近一半的范围，并且呈现出明确的 trait 间异质性。例如，`rice529/num_effective_panicles`、`wheat406/pl_e3` 和 `rice529/grain_weight` 的 `w_TabICL` 相对更高，而 `soybean951/prt`、`rice529/spikelet_length` 和 `rice529/grain_length` 则相对更低。这并不能单独证明 `TabICL` 学到了某种全新的生物学机制，但至少表明，在当前的 OLS 融合框架下，`TabICL` 并不是一个可以忽略的小修正项。更准确地说，统计 prior 与 `TabICL` 的相对贡献会随着 trait 改变；这种 trait-dependent 的权重分配，与我们在 Result 2 中观察到的性能异质性是相互呼应的，也提示未来可以进一步把“trait 类型”与“fusion 权重模式”系统联系起来（Table S13; Table S14; Table S15）。【需引用：OLS/stacking/super learner 或 out-of-fold stacking 权重学习方法文献；可辅以 genomic prediction 中模型权重或 ensemble 权重解释文献】

样本量和 marker-count 两组补充实验，则为这种融合思路提供了比单一 full-data 场景更扎实的稳定性证据。它们共同说明，prior-integrated fusion 的正向增益并不依赖某一个固定样本量或固定 SNP 数量设定；在训练样本比例和 marker 密度变化时，融合框架仍能在整体上保持对统计 prior 的提升。与此同时，这种提升也不应被理解为对所有 trait 都成立的简单单调规律：随着样本量或 marker 数增加，baseline、`TabICL` 和 fusion 会共同变强，因而相对增益可能收敛、扩大或保持接近。更准确地说，这两组实验支持的是“融合先验的架构具有稳定增益潜力”，而不是“增加 prior 或资源后必然在所有 trait 上获得更大优势”（Table S18; Table S19; Table S20; Table S22; Table S23; Table S24; Table S25; Table S26）。【需引用：training population size、marker density、SNP 数量对 genomic prediction accuracy 影响的经典或高水平作物/动物研究】

这也引出本文在方法学上的一个更一般性启发。GS 领域长期由具有明确遗传假设的统计模型主导，而 foundation model 的优势则更多来自灵活表示、弱结构假设和跨任务泛化潜力。当前结果提示，这两类模型之间未必是简单的替代关系。对于 GS 而言，更现实也更有价值的方向，可能不是要求 foundation model 先单独打败所有经典模型，而是探索它如何与已有统计先验进行有效分工。换句话说，statistical priors 仍然承担着对遗传结构的强约束，而 foundation model 则可能提供一种补充性的 data-driven 表示或残差信息。本文的 prior-integrated fusion 框架，本质上正是对这种分工关系的一个初步实现。【需引用：foundation model/representation learning 的一般性高水平综述；domain knowledge 或 scientific prior 与机器学习结合的高水平观点文献】

`TabPFN` 的补充验证进一步支持了这一点，同时也帮助我们划清了结论边界。它表明，prior-integrated fusion 这一路线并不只对 `TabICL` 单一 backbone 有效；即使换成另一类 table / foundation model，在与统计 prior 融合后，也仍然能够获得一定程度的平均增益。但与此同时，`TabPFN` 的整体稳定性明显弱于 `TabICL` 主线，且并未形成相对 trait 内最优 baseline 的稳定领先。这一点非常重要，因为它说明本文的结论不应被夸大为“所有 foundation model 都可以同样有效地进入 GS”。更准确地说，foundation model 单独进入 GS 时通常未必稳定优于强统计模型，但其中一部分模型在与统计 prior 结构化结合后，能够释放出补充价值；只是不同 model 的适配程度、收益幅度和稳定性并不相同（Table S27; Table S28; Table S29; Table S30; Table S31）。【需引用：TabPFN 原始论文；table foundation model 在不同 tabular benchmark 中表现差异或适配性的研究】

当然，本文仍有几个需要明确承认的局限。第一，当前主线 backbone 仍然以 `TabICL` 为主，尽管 `TabPFN` 提供了一个有价值的外部支点，但还不足以把本文结论推广为“面向所有 foundation model 的一般定律”。第二，我们当前正式 prior 池固定为 `BayesB`、`GBLUP` 和 `RKHS`，它们已经覆盖了 GS 中三类非常核心的建模思路，但仍不代表全部可能的 statistical priors。第三，当前融合方式采用的是 trait 级、基于 inner OOF 的线性权重学习，这一设计有助于控制信息泄露、降低比较复杂度，也更适合当前阶段的主文主线；但它仍然是一种相对保守的融合器，未来完全可能扩展到更灵活的样本级或结构感知型融合规则。第四，尽管样本量实验现已补齐到 `8` 个 trait，但这一面板的规模仍然明显小于主线 `36` 个 trait，因此有关 sample-size dependence 的结论更适合被理解为在代表性 trait 面板上的系统证据，而不是对全部 trait 的无条件推广。

在未来工作中，最直接的延伸方向有三类。其一，是扩展更多非 GS 原生模型，检验“先验融合而非单独替代”这一思路是否对更广泛的 table / representation / foundation backbones 同样成立。其二，是扩展 prior 池本身，包括引入更多具有不同遗传结构偏好的 statistical models，进而更系统地研究“prior 匹配度”与“fusion 收益”之间的关系。其三，是把当前的 trait 级线性融合推进到更一般的融合器形式，例如结合 trait 类型、baseline geometry 或更细粒度表示信息，让不同 prior 与 foundation model 的协同关系能够以更自适应的方式表达。

总体而言，本文最重要的贡献并不在于证明 `TabICL` 已经成为一个单独优于经典 GS 模型的新基线，而在于提出并验证了一种更具现实意义的视角：**当 foundation model 直接迁移到 GS 场景时，它未必能够单独稳定奏效；但如果把它视为一个可以与统计遗传先验协同工作的补充信息源，那么它就可能在 GS 中发挥出实际价值。** 这一点不仅解释了为什么 `TabICL` 在本文中能够通过 prior-integrated fusion 获得稳定增益，也为未来更多非 GS 原生模型进入 GS 提供了一条比“直接替代经典模型”更可行的方法学路径。

# 参考文献配置建议

目标正文引用数量建议控制在 `20-30` 篇。这里不再做“删减版”筛选，而是先把可用文献池完整保留，后续再按 Intro / Results / Discussion 的落点压缩到 `24-28` 篇左右。建议优先引用原始方法文献、高水平综述、多数据集 benchmark 和 Nature/Genome Biology/PNAS/Trends/Annual Reviews/Bioinformatics 等高影响力来源，避免堆太多边缘应用文献。

## A. GS 概念与应用背景

- Meuwissen THE, Hayes BJ, Goddard ME. *Prediction of total genetic value using genome-wide dense marker maps.* Genetics, 2001. DOI: `10.1093/genetics/157.4.1819`。用于支撑 GS 概念奠基。
- Hayes BJ, Bowman PJ, Chamberlain AJ, Goddard ME. *Invited review: Genomic selection in dairy cattle: progress and challenges.* Journal of Dairy Science, 2009. DOI: `10.3168/jds.2008-1646`。用于支撑 GS 已成为现代育种核心框架。
- Heffner EL, Sorrells ME, Jannink JL. *Genomic selection for crop improvement.* Crop Science, 2009. DOI: `10.2135/cropsci2008.08.0512`。用于支撑 GS 在作物育种中的应用背景。
- Crossa J, Pérez-Rodríguez P, Cuevas J, et al. *Genomic selection in plant breeding: methods, models, and perspectives.* Trends in Plant Science, 2017. DOI: `10.1016/j.tplants.2017.08.011`。用于支撑 trait-dependent 与 scenario-dependent 挑战。
- Xu Y, et al. *Enhancing genetic gain through genomic selection: from livestock to plants.* Plant Communications, 2020. DOI: `10.1016/j.xplc.2019.100005`。用于支撑 GS 的跨物种方法学价值。

## B. 经典统计遗传模型与 baseline

- VanRaden PM. *Efficient methods to compute genomic predictions.* Journal of Dairy Science, 2008. DOI: `10.3168/jds.2007-0980`。用于支撑 `GBLUP` / GRM / mixed model 先验。
- Gianola D, van Kaam JBCHM. *Reproducing kernel Hilbert spaces regression methods for genomic assisted prediction of quantitative traits.* Genetics, 2008. DOI: `10.1534/genetics.107.084285`。用于支撑 `RKHS` 的非线性建模能力。
- de los Campos G, Naya H, Gianola D, et al. *Predicting quantitative traits with regression models for dense molecular markers and pedigree.* Genetics, 2009. DOI: `10.1534/genetics.109.101501`。用于支撑密集 marker 回归与基因组预测。
- Gianola D, de los Campos G, Hill WG, Manfredi E, Fernando R. *Additive genetic variability and the Bayesian alphabet.* Genetics, 2009. DOI: `10.1534/genetics.109.103952`。用于支撑 Bayes 家族的遗传效应假设。
- Habier D, Fernando RL, Kizilkaya K, Garrick DJ. *Extension of the Bayesian alphabet for genomic selection.* BMC Bioinformatics, 2011. DOI: `10.1186/1471-2105-12-186`。用于支撑 BayesB 及其扩展背景。
- Pérez P, de los Campos G. *Genome-wide regression and prediction with the BGLR statistical package.* Genetics, 2014. DOI: `10.1534/genetics.114.164442`。用于支撑 Bayesian genomic regression 的实现背景。

## C. Table foundation model 与 tabular learning

- Hollmann N, Müller S, Eggensperger K, Hutter F. *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second.* 2022/2023 预印本与后续发表版本。DOI: `10.48550/arXiv.2207.01848`。用于支撑 `TabPFN` 作为代表性 table foundation model。
- Qu J, et al. *TabICL: A Tabular Foundation Model for In-Context Learning on Large Data.* 预印本/技术报告。DOI: `10.48550/arXiv.2502.05564`。用于支撑本文主 backbone。
- Qu J, et al. *TabICLv2: A better, faster, scalable, and open tabular foundation model.* 预印本。arXiv: `2602.11139`。用于支撑当前实现来源。
- den Breejen F, et al. *Why In-Context Learning Transformers are Tabular Data Classifiers.* 预印本。用于支撑 tabular ICL 的方法学背景。

## D. 机器学习/深度学习用于 genomic prediction 的背景

- Pérez-Enciso M, Zingaretti LM. *A guide on deep learning for complex trait genomic prediction.* Genes, 2019. DOI: `10.3390/genes10070553`。用于支撑“深度/灵活模型并不天然优于经典统计模型”。
- Montesinos-López OA, Montesinos-López A, Crossa J, et al. *A review of deep learning applications for genomic selection.* BMC Genomics, 2021. DOI: `10.1186/s12864-020-07319-x`。用于支撑 DL / ML 在 GS 中的总体表现不稳定。
- Lourenço VM, et al. *Genomic prediction using machine learning: ... regularized regression, ensemble, instance-based and deep learning methods ...* BMC Genomics, 2024. DOI: `10.1186/s12864-023-09933-x`。用于支撑多模型 benchmark 语境。

## E. 融合、stacking 与 prior integration

- Wolpert DH. *Stacked generalization.* Neural Networks, 1992. DOI: `10.1016/S0893-6080(05)80023-1`。用于支撑 stacking 的基础逻辑。
- van der Laan MJ, Polley EC, Hubbard AE. *Super learner.* Statistical Applications in Genetics and Molecular Biology, 2007. DOI: `10.2202/1544-6115.1309`。用于支撑 out-of-fold 权重学习的统计合理性。
- Liang M, et al. *A stacking ensemble learning framework for genomic prediction.* Frontiers in Genetics, 2021. DOI: `10.3389/fgene.2021.600040`。用于支撑 genomic prediction 中的融合框架。
- Gu LL, et al. *Ensemble learning for integrative prediction of genetic values with genomic variants.* BMC Bioinformatics, 2024. DOI: `10.1186/s12859-024-05720-x`。用于支撑 ensemble / fusion 在 GS 中的互补性。
- Deng C, et al. *Integrating machine learning with human knowledge.* iScience, 2020. DOI: `10.1016/j.isci.2020.101656`。用于支撑“领域先验 + ML”的一般方法学观点。
- Novakovsky G, Dexter N, Libbrecht MW. *Obtaining genetics insights from deep learning via explainable artificial intelligence.* Nature Reviews Genetics, 2023. DOI: `10.1038/s41576-022-00532-2`。用于 Discussion 中的拔高与边界界定。

## F. 样本量与 marker density

- Meuwissen T, Hayes B, Goddard M. *Genomic selection using low-density marker panels.* Genetics, 2009. DOI: `10.1534/genetics.108.100289`。用于支撑 marker density 对预测精度的影响。
- Pocrnic I, Lourenco D, Chen C, Wiggans G, Misztal I. *Accuracy of genomic prediction using low-density marker panels.* Journal of Dairy Science, 2011. DOI: `10.3168/jds.2010-3917`。用于支撑低密度 marker 的影响。
- Zhang A, et al. *Effect of trait heritability, training population size and marker density on genomic prediction accuracy estimation in 22 bi-parental tropical maize populations.* Frontiers in Plant Science, 2017. DOI: `10.3389/fpls.2017.01916`。用于支撑样本量与 marker 数对准确率的共同作用。
- Tayeh N, et al. *Genomic prediction in pea: effect of marker density and training population size and composition on prediction accuracy.* Frontiers in Plant Science, 2015. DOI: `10.3389/fpls.2015.00941`。用于支撑训练群体规模与 marker 密度联合作用。
- Liu X, et al. *Factors affecting genomic selection revealed by empirical evidence in maize.* The Crop Journal, 2018. DOI: `10.1016/j.cj.2018.03.005`。用于支撑 empirical evidence 语境下的样本量与 marker 数分析。

建议最终正文分配为：A 类 `3-4` 篇，B 类 `6-8` 篇，C 类 `4-6` 篇，D 类 `3-5` 篇，E 类 `3-5` 篇，F 类 `2-3` 篇。这样总量大约 `24-28` 篇，既能支撑 Intro / Results / Discussion 的方法学叙事，又不至于让参考文献堆得过散。

# Methods 中文草稿

## 数据集与评价设计

本研究使用多个作物基因组选择数据集评估 table foundation model 在 GS 任务中的预测价值。主分析覆盖 `cotton1245`、`rice529`、`soybean951` 和 `wheat406` 四个数据集中的 `36` 个 trait。对于每个 trait，我们将基因型 marker 矩阵作为模型输入，将对应表型值作为预测目标；同一数据集中的不同 trait 独立建模和评估。主结果中的统计 baseline 固定为三类具有代表性的 GS 模型：`BayesB`、`GBLUP` 和 `RKHS`，分别代表稀疏或异质 marker effect、全基因组加性关系以及非线性基因组相似性三类统计遗传先验。

所有模型均在相同 trait、相同 outer split 和相同 marker 设置下比较。主评价指标为 outer test fold 上预测值与观测表型之间的 Pearson correlation。对于两个方法 `A` 和 `B`，相对提升定义为：

```text
relative improvement (%) = (Pearson_A - Pearson_B) / Pearson_B * 100
```

在逐 trait 综合比较中，`strongest baseline` 定义为该 trait 内 `BayesB`、`GBLUP` 和 `RKHS` 三者中 Pearson 最高的 baseline。除主分析外，我们还进行了三组补充实验：样本量实验在代表性 trait 上比较 `20%`、`60%` 和 `100%` 训练样本比例；marker-count 实验比较 `2K`、`10K` 和 `50K` 三档 SNP 数量；`TabPFN` 补充实验则用于检验同一 prior-integrated fusion 思路是否可以迁移到另一类 table foundation model。

## TabICL 与 prior-TabICL 框架

我们首先将 `TabICL` 作为不引入统计遗传先验的 standalone predictor 使用。对于每个 trait 和每个 outer training fold，`TabICL` 仅基于该 fold 内的基因型矩阵和表型值进行预测建模，并对对应 outer test fold 生成预测值。该设定记为 `no-prior TabICL`，用于评估 table foundation model 在 GS 中单独使用时的竞争力。

为检验 `TabICL` 是否能够在统计遗传先验之上提供补充预测信息，我们进一步构建 prior-TabICL 融合框架。设 `y_T` 表示 `TabICL` 的预测值，`y_B`、`y_G` 和 `y_R` 分别表示 `BayesB`、`GBLUP` 和 `RKHS` 的预测值。对于 single-prior 融合，我们将一条统计 prior 与 `TabICL` 组合：

```text
y_hat_single_m = (1 - w_m) * y_m + w_m * y_T
```

其中 `m` 分别对应 `BayesB`、`GBLUP` 或 `RKHS`，`w_m` 为该 trait 的 `TabICL` 融合权重。对应的 `only-prior` 对照即为不加入 `TabICL` 的单一统计 prior 预测。

对于 triple-prior 融合，我们先将三条统计 prior 聚合为一个 prior aggregate：

```text
y_prior = a_B * y_B + a_G * y_G + a_R * y_R
```

其中 `a_B + a_G + a_R = 1`。随后再将该 prior aggregate 与 `TabICL` 组合：

```text
y_hat_triple = (1 - w_T) * y_prior + w_T * y_T
```

其中 `w_T` 表示 `TabICL` 相对于聚合统计 prior 的融合权重。本文主结果采用 trait-level ordinary least squares 估计上述权重：single-prior 中直接学习 `y_m` 与 `y_T` 的组合权重；triple-prior 中先学习三条统计 prior 的聚合权重，再学习 prior aggregate 与 `TabICL` 的组合权重。所有权重均在 inner validation 的 out-of-fold 预测上学习，并在 outer test fold 上固定使用。

`TabPFN` 补充实验使用相同的 prior-integrated fusion 框架，只是将 `TabICL` 的预测值 `y_T` 替换为 `TabPFN` 的预测值。该实验用于检验融合思路的模型可迁移性，而不是替代 `TabICL` 主线结果。

## 数据分割与信息泄露控制

为避免信息泄露，所有模型训练、模型选择、统计 prior 构建和融合权重学习均严格限制在 outer training fold 内完成。outer test fold 的表型值只用于最终性能评估，不参与 `TabICL` 配置选择、baseline 模型拟合、prior 聚合权重学习或 `TabICL` 融合权重学习。

具体而言，对于每个 trait，`TabICL`、`BayesB`、`GBLUP` 和 `RKHS` 在相同 outer split 上分别生成预测。用于学习融合权重的输入不是训练集上的 in-sample 预测，而是 outer training fold 内部通过 inner validation 得到的 out-of-fold 预测。这样，权重学习所看到的每个样本预测都来自未在该样本表型上拟合过的模型，从而尽可能模拟真实 held-out 预测场景。学习得到的 trait-level 权重随后固定应用于同一 outer fold 的 `TabICL` 与统计 prior 的 outer test 预测。

样本量实验和 marker-count 实验沿用同一数据分割原则。训练样本比例变化只作用于 outer training fold 内部，outer test fold 始终保持为最终 held-out 评估集；不同 marker 数量设定下，所有比较方法使用一致的 marker 输入和一致的 outer split。通过这一设计，本文中的 `no-prior TabICL`、统计 baseline、single-prior 融合和 triple-prior 融合均在相同数据边界下比较，避免了由 split 不一致或 outer test 信息进入权重学习造成的偏差。
