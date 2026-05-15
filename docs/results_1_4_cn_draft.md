# Results 1-6 中文草稿

## Result 1. Table foundation model 可以进入 GS，但单独使用并不是稳定最优

我们首先评估了 `no_prior-TabICL` 作为独立预测器在 GS 场景中的表现，以回答一个最直接的问题：table foundation model 是否能够不依赖任何统计遗传先验，单独在 GS 中建立竞争力。基于 `5.4-duli-liudang` 主线、排除 `pig3534` 后的 `36` 个 trait，`no_prior-TabICL` 的平均 Pearson 为 `0.652`。如果分别与三条正式统计 baseline 比较，`BayesB`、`GBLUP` 和 `RKHS` 的平均 Pearson 分别约为 `0.677`、`0.671` 和 `0.677`；对应地，`no_prior-TabICL` 相对三者的逐 trait 平均相对差异分别为 `-4.41%`、`-3.56%` 和 `-4.62%`。也就是说，无论与哪一类经典统计模型相比，`TabICL` 单独使用时都没有形成稳定的平均优势。

这一结果说明，`TabICL` 单独进入 GS 时总体上仍弱于三条经典 baseline，但这种平均层面的劣势并不意味着它在所有 trait 上都缺乏价值。恰恰相反，在若干 trait 上，`no_prior-TabICL` 已经表现出明确的局部优势。例如，在 `rice529/grain_weight` 上，`no_prior-TabICL` 的 Pearson 为 `0.5741`，高于 `BayesB` 的 `0.5619`、`GBLUP` 的 `0.5554` 和 `RKHS` 的 `0.5564`，对应相对提升分别约为 `+2.18%`、`+3.38%` 和 `+3.19%`；在 `rice529/plant_height` 上，`no_prior-TabICL` 达到 `0.8464`，同样高于 `BayesB` 的 `0.8427`、`GBLUP` 的 `0.8427` 和 `RKHS` 的 `0.8386`，相对提升约为 `+0.44%`、`+0.44%` 和 `+0.93%`；在 `wheat406/pl_e1` 上，`no_prior-TabICL` 为 `0.8581`，也高于 `BayesB` 的 `0.8433`、`GBLUP` 的 `0.8293` 和 `RKHS` 的 `0.8429`，对应提升约为 `+1.76%`、`+3.48%` 和 `+1.80%`。这些例子说明，尽管 `TabICL` 作为单独预测器尚未形成跨 trait 的稳定优势，但它确实能够在部分性状上恢复出高于经典统计模型的有效预测信号。因此，table foundation model 进入 GS 仍然是有价值的；真正的问题不是“它是否完全无效”，而是“它为何不能稳定胜过现有统计方法，以及这种不足能否通过后续 prior 融合得到补偿”。换言之，Result 1 支持这样一个判断：**将 table foundation model 引入 GS 是有意义的，但如果希望它单独成为一个稳定优于经典统计模型的 predictor，目前证据并不充分。**

进一步看三条 baseline 的对照关系，这一结论会更加清楚。`no_prior-TabICL` 相对 `GBLUP` 的平均差距最小，而相对 `BayesB` 和 `RKHS` 的平均差距略大，但三者整体上都保持在相近区间内。与此同时，从逐 trait 胜率看，`no_prior-TabICL` 相对 `BayesB`、`GBLUP` 和 `RKHS` 分别仅在 `4/36`、`5/36` 和 `3/36` 个 trait 上占优。也就是说，当前 `TabICL` 单独使用时，并不是完全没有局部亮点，但也尚未对 `BayesB`、`GBLUP` 或 `RKHS` 中任何一条线形成稳定压制。因此，Result 1 给出的结论不是 “TabICL 无法用于 GS”，而是：**table foundation model 单独进入 GS 后具有竞争力，但不能稳定超越现有统计模型，这恰恰构成了后续引入统计 prior 的动机。**

## Result 2. 与统计 prior 融合后，TabICL 可以稳定提升预测性能

在确认 `no_prior-TabICL` 难以单独稳定胜出的基础上，我们进一步检验：如果将 TabICL 与经典统计遗传 prior 结合，它是否能够释放出更稳定的预测增益。我们首先考察三种 single-prior 融合，即 `single_bayesb_two_step_ls`、`single_gblup_two_step_ls` 和 `single_rkhs_two_step_ls`。结果显示，三种 single-prior-TabICL 都能够在 trait 层面带来广泛的正向提升。相对于各自 prior-only 对照，`single_bayesb_two_step_ls`、`single_gblup_two_step_ls` 和 `single_rkhs_two_step_ls` 的平均相对提升分别为 `1.32%`、`1.62%` 和 `0.82%`；其中，正提升 trait 数分别为 `35/36`、`35/36` 和 `30/36`。这说明 TabICL 并不是简单重复已有 prior 所编码的信息，而是能够在单 prior 条件下为 GS 带来额外预测信号。

这些 single-prior 融合的收益在若干 trait 上尤其明显。以 `wheat406/sl_e1` 为例，`single_bayesb_two_step_ls` 相对 `BayesB` 提升 `4.98%`，`single_gblup_two_step_ls` 相对 `GBLUP` 提升 `5.69%`，`single_rkhs_two_step_ls` 相对 `RKHS` 提升 `3.71%`；在 `rice529/grain_weight` 上，三条 single 融合也都表现出一致增益，分别相对各自 prior 提升 `3.61%`、`4.52%` 和 `3.96%`。此外，`rice529/heading_date` 与 `wheat406/pl_e1` 也是较有代表性的正例：前者中 `single_bayesb_two_step_ls` 和 `single_gblup_two_step_ls` 分别相对对应 prior 提升 `2.54%` 和 `3.37%`，后者中三条 single 融合分别达到 `2.53%`、`3.67%` 和 `2.42%` 的提升。换言之，single-prior-TabICL 的价值首先体现为：**无论 prior 是 `BayesB`、`GBLUP` 还是 `RKHS`，只要将 TabICL 纳入融合，通常都能够把该 prior 本身再抬高一步。**

在此基础上，我们进一步评估多 prior 融合的主结果 `triple_two_step_ls`。如果先与三条正式 baseline 逐一比较，其优势已经非常清楚：相对于 `BayesB`，`triple_two_step_ls` 在 `36/36` 个 trait 上全部更优，平均相对提升为 `1.70%`；相对于 `GBLUP`，同样达到 `36/36` 个 trait 全部更优，平均相对提升为 `2.61%`；相对于 `RKHS`，则在 `30/36` 个 trait 上更优，平均相对提升为 `1.42%`。这说明 triple 融合的收益并不只是体现在“偶尔超过某一条 baseline”上，而是相对于三类经典统计建模假设都保持了稳定的整体优势，尤其对 `BayesB` 和 `GBLUP` 呈现出几乎一致的全面提升。

如果进一步与三条 single-prior-TabICL 比较，`triple_two_step_ls` 依然表现出更高的整体稳健性。相对于 `single_bayesb_two_step_ls`、`single_gblup_two_step_ls` 和 `single_rkhs_two_step_ls`，`triple_two_step_ls` 的平均相对提升分别为 `0.37%`、`0.97%` 和 `0.59%`；对应地，它分别在 `21/36`、`33/36` 和 `27/36` 个 trait 上不低于这三条 single 结果。与此同时，`triple_two_step_ls` 相对 `only-triple-prior` 的平均 Pearson 提升为 `0.93%`，说明 triple 的增益并不只是来自三条统计 baseline 的线性重组，而是在 triple prior 已经形成的组合基础上，TabICL 仍然继续提供了可转化为精度的补充信息。换言之，triple 的价值并不只是“比 prior-only 更强”，而是在多数情况下还能进一步超过单一 prior 融合，并在 `only-triple-prior` 之上继续取得稳定增益。

在上述比较基础上，如果再以 strongest baseline 作为综合参照，`triple_two_step_ls` 仍然是当前主结果中整体最稳健的方法。在 `36` 个非猪 trait 中，`triple_two_step_ls` 有 `30` 个 trait 超过 strongest baseline，平均相对提升为 `0.85%`；相对于 `no_prior-TabICL`，它在 `36/36` 个 trait 上全部为正提升，平均相对提升达到 `6.84%`。但需要强调的是，`triple_two_step_ls` 并不是在每个 trait 上都无条件最优。它在 `rice529/num_panicles`、`rice529/yield`、`soybean951/fa16c`、`soybean951/md`、`soybean951/prt` 和 `soybean951/vbn` 上仍略低于 strongest baseline，降幅约为 `0.07%` 到 `0.75%`。因此，triple 的价值更准确地说是“整体最稳健”，而不是“逐 trait 全面垄断”。

如果进一步按数据集分层，`triple` 的稳定性也依然成立。在 `cotton1245` 中，`triple_two_step_ls` 相对 strongest baseline 的平均提升为 `+1.57%`，并在 `4/4` 个 trait 上全部获胜；在 `wheat406` 中，平均提升为 `+1.50%`，在 `9/9` 个 trait 上全部获胜；在 `rice529` 中，平均提升为 `+0.64%`，赢 `8/10` 个 trait；在 `soybean951` 中，平均提升为 `+0.35%`，赢 `9/13` 个 trait。特别值得注意的是，`soybean951` 虽然相对 strongest baseline 的边际提升较小，但 `no_prior-TabICL` 在该数据集上平均落后 strongest baseline `-9.06%`，而 `triple_two_step_ls` 相对 `no_prior` 的平均提升却达到 `+11.28%`。这说明在一些对 TabICL 单独更不友好的数据集上，将 TabICL 与统计先验进行融合，仍然可以进一步提高预测准确率。

按 strongest baseline 的类型分层后，这一规律更加清楚。当 strongest baseline 来自 `BayesB` 时，`triple_two_step_ls` 的平均提升为 `+1.10%`，并在 `17/17` 个 trait 上全部超过 strongest baseline；当 strongest baseline 来自 `GBLUP` 时，平均提升为 `+1.02%`，在 `3/3` 个 trait 上全部获胜；当 strongest baseline 来自 `RKHS` 时，平均提升下降为 `+0.56%`，赢 `10/16` 个 trait。当前 `triple` 未能超过 strongest baseline 的 `6` 个 trait，全部落在 `RKHS-best` 区域。换言之，`triple` 对 `BayesB-best` 和 `GBLUP-best` trait 的增益最稳定，而在 `RKHS` 主导的 trait 上仍存在剩余难度。

最后，如果把 `triple` 与 strongest single 直接比较，它的价值将更准确地体现为“稳定性与统一性”，而不是“在每个 trait 上都远超最佳 single”。在 `36` 个非猪 trait 中，最佳 single 的归属分别为：`single_bayesb` `19` 个、`single_gblup` `5` 个、`single_rkhs` `12` 个，说明不存在一条 single 方法能够在所有 trait 上始终最优。相较之下，`triple_two_step_ls` 相对 strongest single 的平均绝对差仅为 `-0.00039`，相对百分比差仅为 `-0.0746%`，中位数为 `0`；它在 `19/36` 个 trait 上不低于 strongest single，在其中 `13/36` 个 trait 上严格更优。因此，`triple` 的主要价值并不在于平均上大幅碾压 strongest single，而在于：在无需事先知道哪条 single 最适合某个 trait 的前提下，提供了一条统一且足够稳健的主结果线路。

从整体上看，Result 2 支持全文最核心的经验事实：**TabICL 的价值主要体现在与统计 prior 的结合，而不是单独替代统计模型。** single-prior 融合证明 TabICL 可以稳定抬高单一统计 prior，triple-prior 融合则进一步提供了一条在大多数 trait 上都足够稳健、且无需预先判断哪条 prior 最匹配的统一主结果线路。

## Result 3. Single-prior 融合权重表明，TabICL 在不同 trait 上持续保留了稳定但异质的贡献

Result 2 已经表明，single-prior 和 triple-prior 融合都能够稳定提高预测准确率。接下来的问题不再是“融合是否有效”，而是：在 OLS 学到的最优线性组合中，`TabICL` 究竟占据怎样的位置？由于 single-prior 情形只有“一个统计 prior + 一个 TabICL”两个组成部分，其权重含义最直接，因此我们首先聚焦于 single-prior 融合的权重结构，来判断 `TabICL` 在融合中究竟只是一个很小的修正项，还是持续保留了可观贡献。

整体上看，single-prior 融合中的 `w_tabicl` 并没有系统性塌缩到接近 `0` 的区域。排除猪数据后，`single_bayesb`、`single_gblup` 和 `single_rkhs` 三条线的平均 `w_tabicl` 分别为 `0.493`、`0.493` 和 `0.503`，对应范围大致为 `0.433-0.553`、`0.447-0.536` 和 `0.442-0.560`。这意味着在 single-prior 的 OLS 最优组合中，`TabICL` 并没有被边缘化为一个极小的残差修正项，而是在多数 trait 上都持续保留了接近一半的权重。换言之，single-prior 融合之所以能够相对各自 prior-only 获得系统性提升，并不仅仅是因为统计 prior 本身已经足够强，还因为 `TabICL` 在融合中持续保留了可观且稳定的贡献。

更重要的是，这种对 `TabICL` 的保留程度在不同 trait 间并不是固定不变的，而是呈现出明确的异质性。如果把三条 single-prior 结果的 `w_tabicl` 取平均，则其整体范围约为 `0.443-0.544`。较高权重的 trait 包括 `rice529/num_effective_panicles`（平均 `w_tabicl = 0.544`）、`wheat406/pl_e3`（`0.542`）和 `rice529/grain_weight`（`0.539`）；较低权重的 trait 则包括 `soybean951/prt`（`0.443`）、`rice529/spikelet_length`（`0.454`）和 `rice529/grain_length`（`0.457`）。从整体趋势看，single-prior 的平均 `w_tabicl` 与 `no_prior-TabICL` 相对 strongest baseline 的表现呈中等正相关（`r = 0.567`）。也就是说，当 `TabICL` 单独使用时本身已经更接近 strongest baseline，OLS 在 single-prior 融合中通常也更倾向于保留更高比例的 `TabICL` 成分；反之，当 `TabICL` 单独表现较弱时，融合仍然可以带来增益，但其组合更偏向于依赖统计 prior。这说明 `TabICL` 在融合中的作用不是固定常数，而是随 trait 改变的：它在某些 trait 上保留更高份额，在另一些 trait 上则更多作为次级修正项参与组合。

在此基础上，triple-prior 的权重结果可以作为一个一致性补充，而不必像 single-prior 那样被直接展开解释。由于 triple-prior 采用的是两段 OLS，`w_tabicl` 所对应的是“TabICL 相对于 prior aggregate”的权重，而不是相对于某一个单独 baseline 的直接比例，因此其含义天然比 single-prior 更绕。但即便如此，triple 融合中的 `w_tabicl` 仍然保持在一个稳定且不可忽略的区间内：排除猪数据后，其平均值为 `0.489`，范围约为 `0.434-0.549`。同时，triple 内部 `BayesB`、`GBLUP` 和 `RKHS` 的 prior share 也并未表现出被某一条 baseline 长期完全垄断的模式。这说明在更复杂的多 prior 设定下，融合框架同样不是通过机械平均来工作，而是在 prior 侧和 `TabICL` 侧都进行了 trait-dependent 的重分配。因而，Result 3 更稳妥地支持这样一个判断：**`TabICL` 在 OLS 融合中并不是一个可以忽略的小项，而是在不同 trait 上持续保留了稳定但异质的贡献；single-prior 的权重结构最清楚地揭示了这一点，而 triple-prior 的结果则从更复杂的设定中给出了方向一致的支持。**

## Result 4. 样本量变化表明，多 prior 融合的收益具有明显的 trait 依赖性与 sample-size 依赖性

在完成主结果之后，我们进一步考察融合收益是否会随着训练样本规模变化而发生系统性改变。为保持与主线一致，样本量实验继续沿用 `5.4-duli-liudang` 的解偶复用逻辑，默认排除 `pig3534`，并固定使用既有 `best_block`。当前样本量分析已经补齐到 `8` 个 trait，即 `cotton_fibelo`、`cotton_fiblen`、`grain_weight`、`grain_width`、`bbd`、`lw`、`sl_e1` 和 `sl_e2`，并比较了 `20%`、`60%` 和 `100%` 样本量下 `no_prior-TabICL`、三条 baseline、三条 single-prior 融合以及 `triple_two_step_ls` 的表现，以检验多 prior 融合的收益是否具有 trait-dependent 与 sample-size-dependent 特征。

从绝对准确率看，样本量增加会同时抬升 `no_prior`、baseline 和 fusion 的表现，而不是只让某一类方法单独受益。在当前 `8` 个 trait 上，`no_prior-TabICL` 的平均 Pearson 从 `20%`、`60%` 到 `100%` 分别为 `0.4867`、`0.5827` 和 `0.6097`；`BayesB`、`GBLUP` 和 `RKHS` 的平均 Pearson 也分别从 `0.5173 / 0.5073 / 0.5200` 升至 `0.5982 / 0.5913 / 0.5934`，并在 `100%` 时进一步达到 `0.6314 / 0.6208 / 0.6271`。三条 single-prior 融合同样同步上升：`single_bayesb_two_step_ls`、`single_gblup_two_step_ls` 和 `single_rkhs_two_step_ls` 的均值在 `20%` 时分别为 `0.5226`、`0.5191` 和 `0.5198`，在 `60%` 时为 `0.6096`、`0.6044` 和 `0.6045`，在 `100%` 时为 `0.6417`、`0.6341` 和 `0.6355`；`triple_two_step_ls` 则对应为 `0.5237`、`0.6086` 和 `0.6435`。这说明后文讨论的相对提升，发生在“所有方法都会随着样本量增加而共同变强”的背景下。

从整体均值看，`triple-prior-TabICL` 在三个样本量档位下都保持了相对三条 baseline 的正向增益，但这种增益并不是对所有 baseline 以同一种方式变化。相对 `BayesB`，`triple` 在 `20%`、`60%` 和 `100%` 下的平均提升分别为 `+1.40%`、`+2.07%` 和 `+2.34%`；相对 `GBLUP` 的平均提升分别为 `+4.28%`、`+3.24%` 和 `+4.13%`；相对 `RKHS` 则分别为 `+0.63%`、`+2.81%` 和 `+2.71%`。若进一步以 trait 内 strongest baseline 作为综合参照，`triple` 在 `20%`、`60%` 和 `100%` 下的平均相对提升分别为 `+0.25%`、`+1.79%` 和 `+1.55%`，对应胜出 trait 数为 `5/8`、`7/8` 和 `8/8`。换言之，随着训练样本增加，`triple-prior-TabICL` 并没有丢失其融合价值，而是逐步从“部分 trait 上已可见的边际优势”发展为“在大多数甚至全部 trait 上均不低于 strongest baseline 的更稳定表现”。

trait-level 轨迹进一步说明，这种收益仍然具有明显的结构依赖性。以 `rice529/grain_weight` 为例，在 `20%` 样本量下，`no_prior-TabICL` 已达到 `0.3449`，高于 `BayesB` (`0.3311`) 和 `GBLUP` (`0.2964`)，但仍略低于 `RKHS` (`0.3533`)；此时 `triple_two_step_ls` 为 `0.3544`，相对 `GBLUP` 的提升已达到 `+19.57%`。到 `100%` 时，该 trait 的 `no_prior-TabICL`、三条 baseline 和 `triple` 分别提高到 `0.5741`、`0.5619 / 0.5554 / 0.5564` 和 `0.5821`，说明这一困难 trait 在小样本和全样本下都能从 prior 融合中获得稳定补偿。相对地，`rice529/grain_width` 与 `soybean951/lw` 更清楚地显示了“prior 匹配度”而非“prior 数量”本身的重要性：`grain_width` 在 `20%` 下的 `triple` (`0.7437`) 仍低于 `RKHS` (`0.7504`)，到 `100%` 时才转为略高于 `BayesB` 和 `RKHS`；`lw` 在 `20%` 下的 `triple` (`0.4248`) 仍低于三条 baseline，但在 `60%` 和 `100%` 时分别升至 `0.5025` 和 `0.5561`，并开始稳定超过 strongest baseline。与此同时，新增纳入的 `wheat406/sl_e1` 和 `sl_e2` 也带来了一个重要修正：这两个 trait 在 `20%` 样本量下并不都表现为强正增益，例如 `sl_e1` 的 `triple` 相对 `BayesB` 仍为 `-2.11%`，但到 `100%` 时两者已分别转为 `+5.13%` 和 `+5.36%`。这说明部分 trait 的融合收益需要更充足样本后才会被更清楚地释放出来。

综合来看，当前 `8`-trait 样本量实验表明，多 prior 融合的价值确实具有明显的 trait 依赖性和 sample-size 依赖性，但这种依赖性并不意味着“样本量越大，融合收益必然单调扩大”，也不意味着“只有小样本才需要融合”。更准确的说法是：随着样本量增加，`triple-prior-TabICL` 的整体稳健性在增强，但不同 trait、不同 baseline 所对应的增益轨迹并不相同；因此，样本量变化更多是在重塑 fusion 与统计 prior 之间的相对关系，而不是把所有 trait 推向同一种统一模式。

## Result 5. 标记密度提升增强了融合框架的整体性能上限，但增益模式并非对所有 baseline 同步扩张

与样本量实验互补，我们进一步考察了标记密度变化下融合框架的表现，以检验 TabICL 与统计 prior 的互补性是否会随着 SNP 信息增加而保留甚至增强。该分析基于 `4` 个数据集、`8` 个代表性 trait，并设置 `2K`、`10K` 和 `50K` 三档 marker count。其中，`10K` 直接复用主线 `5.4-duli-liudang` 结果，而 `2K` 与 `50K` 则在独立的解偶复用产线中重新构建 marker 子集、重新搜索对应 `best_block`，并正式留档 `no_prior-TabICL`、`BayesB`、`GBLUP` 和 `RKHS`，再由这些留档结果直接构建全部 `single-prior` 与 `triple-prior` 融合输出。与主线一致，评价指标仍为 `5-fold outer test Pearson`。

从绝对准确率看，marker 数增加会同时抬升 `no_prior`、baseline 和 fusion 的表现，而不是只让某一类方法单独受益。当前 `8` 个 trait 上，`no_prior-TabICL` 的平均 Pearson 从 `2K`、`10K` 到 `50K` 分别为 `0.5890`、`0.6097` 和 `0.6195`；`BayesB`、`GBLUP` 和 `RKHS` 的平均 Pearson 也分别从 `0.6028 / 0.6017 / 0.6147` 提升到 `0.6314 / 0.6208 / 0.6271`，并在 `50K` 时进一步达到 `0.6599 / 0.6328 / 0.6352`。三条 single-prior 融合同样整体上升：`single_bayesb_two_step_ls`、`single_gblup_two_step_ls` 和 `single_rkhs_two_step_ls` 的均值分别由 `0.6156 / 0.6146 / 0.6185` 提升至 `0.6417 / 0.6341 / 0.6355`，并在 `50K` 时达到 `0.6674 / 0.6461 / 0.6457`；`triple_two_step_ls` 则从 `0.6201` 进一步升至 `0.6435` 和 `0.6675`。因此，更高 marker 密度抬升的是整个比较框架的准确率上限，而不是仅仅制造某一种方法的表面优势。与此同时，`no_prior-TabICL` 在三档 marker 下的平均表现仍低于三条正式 baseline，这也再次说明，table foundation model 单独进入 GS 虽然具有一定竞争力，但其更稳定的价值仍然来自与统计 prior 的结合。

在这一绝对值背景下，`triple-prior-TabICL` 相对 `BayesB` 的优势总体保持为正，但会随着 marker 数增加而逐渐收敛，说明当标记更加丰富时，`BayesB` 本身也能更充分地提取稀疏大效应信息。例如，`cotton_fibelo` 上，`no_prior-TabICL` 在 `2K / 10K / 50K` 下分别为 `0.6212 / 0.6324 / 0.6245`，`BayesB` 对应为 `0.6269 / 0.6595 / 0.6921`，`single_bayesb_two_step_ls` 为 `0.6381 / 0.6681 / 0.6950`，`triple_two_step_ls` 为 `0.6437 / 0.6667 / 0.6941`。可以看到，随着 marker 数提高，融合结果仍然保持在最高区间，但相对 `BayesB` 的边际优势明显缩小。`grain_weight` 也体现出类似但更强的 trait 依赖性：在 `2K` 和 `10K` 下，`triple_two_step_ls` 分别达到 `0.5743` 和 `0.5821`，高于 `BayesB` 的 `0.5483` 和 `0.5619`；但在 `50K` 下，`BayesB` 升至 `0.5828`，反而高于 `triple` 的 `0.5758`。这说明相对 `BayesB` 的平均提升从 `+2.92%`、`+2.34%` 收敛到 `+1.48%`，并不是因为 fusion 失效，而是因为在部分 trait 上 `BayesB` 对高密度 marker 的利用效率更高。

相比之下，`triple-prior-TabICL` 相对 `GBLUP` 和 `RKHS` 的优势在三档 marker 下始终为正，并且在 `50K` 时进一步扩大到 `+6.47%` 和 `+5.64%`。这种趋势同样可以从绝对准确率轨迹中直接看到。以 `soybean951/lw` 为例，`no_prior-TabICL` 从 `0.4597` 上升到 `0.5017` 和 `0.5571`，`GBLUP` 与 `RKHS` 则分别从 `0.4927 / 0.5028` 上升到 `0.5259 / 0.5318`，并在 `50K` 时维持在 `0.5421 / 0.5418`；与之对应，`single_bayesb_two_step_ls` 从 `0.5057` 增至 `0.5561` 和 `0.6593`，`triple_two_step_ls` 也从 `0.5062` 提升到 `0.5561` 和 `0.6593`。也就是说，更丰富的 marker 信息并没有削弱融合框架的作用，反而使其相对某些统计模型的互补收益更充分地显现。`cotton_fibelo` 也表现出类似模式：尽管 `BayesB` 在 `50K` 下非常强，但 `triple` 相对 `GBLUP` 和 `RKHS` 的优势依然由 `2K` 时的有限正增益扩展到 `50K` 时的更高水平。

单 prior 融合也呈现出一致但不完全相同的规律，说明 richer marker information 并不是只被 triple 融合独占利用。三个 `single-prior-TabICL` 相对各自 prior 的平均增益在全部 marker 档位上均保持为正：`single_bayesb_two_step_ls` 相对 `BayesB` 的平均提升为 `+2.06%`、`+1.92%` 和 `+1.45%`，表现出与 triple 相似的收敛趋势；`single_gblup_two_step_ls` 相对 `GBLUP` 的平均提升则为 `+2.14%`、`+2.45%` 和 `+2.57%`，是三条 single 线路中最稳定的一条；`single_rkhs_two_step_ls` 相对 `RKHS` 的增益在低 marker 密度下较小（`+0.51%`），但在 `50K` 时提升到 `+1.95%`。例如，在 `wheat406/sl_e2` 上，`RKHS` 从 `0.3992`、`0.4367` 提升到 `0.4431`，`single_rkhs_two_step_ls` 则对应为 `0.4013`、`0.4381` 和 `0.4431`，说明当 trait 更匹配 `RKHS` 这类 prior 时，single 融合的增益可能较小但仍保持不劣；而在 `lw` 上，`single_bayesb_two_step_ls` 则从 `0.5057`、`0.5561` 增至 `0.6593`，持续高于对应的 `BayesB`。这说明 TabICL 并非只对某一种统计 prior 有效，而是能够在三类统计建模假设上都提供补充信号，只是补充模式并不相同。

trait-level 结果进一步说明，marker 数增加的收益应被理解为“整体趋势”，而不是“所有 trait 的单调定律”。在 `10K` 条件下，`triple_two_step_ls` 在 `8/8` 个 trait 上都优于 strongest baseline；在 `2K` 下，这一数字为 `7/8`，唯一例外是 `wheat406/sl_e2`，其 `triple`（`0.3956`）略低于 `RKHS`（`0.3992`）和 `single_rkhs`（`0.4013`）；在 `50K` 下，`triple` 仍在 `6/8` 个 trait 上超过 strongest baseline，但 `rice529/grain_weight` 与 `grain_width` 两个 trait 被 `BayesB` 反超，其中 `grain_width` 在 `50K` 下的 `BayesB`、`single_bayesb` 与 `triple` 已几乎重合在 `0.8422` 附近。进一步看绝对性能轨迹，`8` 个 trait 中有 `6` 个在 `2K <= 10K <= 50K` 下呈现非下降趋势，仅 `grain_weight`（`0.5743 -> 0.5821 -> 0.5758`）与 `bbd`（`0.8138 -> 0.8215 -> 0.8210`）未完全满足单调上升。由此，更高 SNP 数量通常有利于融合模型表现，但具体收益仍受到 trait 本身遗传结构和 prior 竞争关系的共同影响。

综合来看，当前 8-trait SNP count 消融结果表明，更高的 marker 密度通常会提高融合模型的绝对准确率，并在相对 `GBLUP` 与 `RKHS` 的比较中保留更明显的平均增益；但相对 `BayesB` 的优势则更容易收敛，且少数 trait 在高 marker 密度下仍可能由单一强 prior 主导。因此，marker 数增加带来的并不是所有 trait 上一致的单调收益，而是一个同时受 trait 结构与 prior 竞争关系影响的整体趋势。

## Result 6. TabPFN 补充验证支持融合框架具有一定可迁移性，但其稳定性弱于 TabICL 主线

作为对这一方法学主线的进一步补充，我们还引入了一个额外的 table / foundation model，即 `TabPFN`，以检验 prior-integrated fusion 这一思路是否仅对 `TabICL` 单一 backbone 成立。该补充验证采用 `10K SNP + 8 个 SNP-count traits` 的设定，`no_prior-TabPFN` 的结果来自独立正式目录 `outputs/5.4-tabpfn-10k-8traits`，而融合结果则在 `outputs/5.4-tabpfn-10k-8traits-fusion` 中基于与主线一致的解偶复用逻辑构建，并继续复用 `BayesB / GBLUP / RKHS` 三条正式 baseline。需要强调的是，这条 `TabPFN` 结果线并不替代 `TabICL` 主线，也不构成新的主结果，而是用于回答一个更狭义但重要的问题：**多 prior 融合框架是否具有一定的模型可迁移性。**

结果显示，`TabPFN` 在当前设定下同样符合全文主线。其 `no_prior` 版本的 8-trait 平均 Pearson 为 `0.5986`，低于 `BayesB`（`0.6314`）、`GBLUP`（`0.6208`）和 `RKHS`（`0.6271`），说明即使换用另一类 table / foundation model，单独进入 GS 后的总体表现依然未能稳定超过强统计 baseline。进一步将 `TabPFN` 与单一统计 prior 结合后，三条 single-prior-TabPFN 的平均 Pearson 分别为 `0.6364`、`0.6315` 和 `0.6309`，相对各自 baseline 的平均提升分别为 `+0.77%`、`+1.79%` 和 `+0.42%`，表明它也能够从统计 prior 中获得一定补偿。

在此基础上，`triple-prior-TabPFN` 的平均 Pearson 为 `0.6352`。其相对 `BayesB`、`GBLUP` 和 `RKHS` 的平均提升分别为 `+0.6315%`、`+2.4058%` 和 `+1.0401%`；相对不含 `TabPFN` 的 `only-triple-fusion` 也仍有 `+0.3496%` 的平均增益。这说明多 prior 融合框架并不只对 `TabICL` 单一 backbone 有效，而是具有一定的模型可迁移性。但与此同时，`triple-prior-TabPFN` 相对 trait 内最优 baseline 的 8-trait 平均提升仍为 `-0.1230%`，仅在 `4/8` 个 trait 上为正，整体稳定性明显弱于 `TabICL` 主线。因此，这部分结果更适合作为对“框架可扩展性”的补充支持，而不是对主线结论的替代：它说明 foundation model 单独用于 GS 时通常未必稳定优于强统计模型，但在多 prior 融合框架中，可以在一定程度上释放补充价值；同时，不同 foundation model 的收益幅度和稳定性并不相同，而当前整体更稳定、结果更完整的主线仍然是 `TabICL`。

## 当前可直接挂靠的正式文件

- 主结果表：`outputs/5.4-duli-liudang/main_results_non_pig_fixed.csv`
- 全 trait compare：`outputs/5.4-duli-liudang/compare_all_41_traits.csv`
- 两个代表 trait 的权重方案对比：`outputs/5.4-duli-liudang/compare_weight_schemes_two_traits_fullsingles.csv`
- 样本量主线日志：`outputs/5.4-sample_size-decoupled/pipeline.log`
- SNP 数量实验总结：`docs/notes/2026-05-08-snp-count-experiment-summary.md`
- TabPFN 补充验证：`docs/notes/2026-05-09-tabpfn-10k-8traits-summary.md`

## 当前这版草稿的使用建议

- `Result 1-3` 已可以作为正文初稿继续精修
- `Result 4-6` 的结构已拆开，当前更适合分别作为“样本量”“标记量”“框架可扩展性”三章独立打磨
- 样本量部分目前仍应视为阶段性版本，因为 `wheat406/sl_e1` 和 `wheat406/sl_e2` 仍在补跑
- 如果后续要转英文，建议先固定每一章的主结论句，再做逐段压缩
