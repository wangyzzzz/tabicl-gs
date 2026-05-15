# 基于当前正式结果仍可直接补充的分析

本文档只基于当前已经完成的正式结果后处理，不依赖新增训练。所有主统计默认基于：

- 主结果：`outputs/5.4-duli-liudang/main_results_non_pig_fixed.csv`
- 全 compare：`outputs/5.4-duli-liudang/compare_all_41_traits.csv`
- 融合权重摘要：`outputs/5.4-duli-liudang/fusion/**/group_shared_gate_group_summary.json`
- 默认排除：`pig3534`

## 1. 按 best baseline type 分层

这是目前最值得补的一项，因为它直接回答：**TabICL/fusion 更容易在哪类统计遗传背景下发挥作用？**

### 当前统计

- 当 strongest baseline 是 `BayesB` 时：
  - `n = 17`
  - `no_prior-TabICL vs best baseline = -3.50%`
  - `triple_two_step_ls vs best baseline = +1.10%`
  - `triple` 赢 `17/17`

- 当 strongest baseline 是 `GBLUP` 时：
  - `n = 3`
  - `no_prior-TabICL vs best baseline = -3.35%`
  - `triple_two_step_ls vs best baseline = +1.02%`
  - `triple` 赢 `3/3`

- 当 strongest baseline 是 `RKHS` 时：
  - `n = 16`
  - `no_prior-TabICL vs best baseline = -7.24%`
  - `triple_two_step_ls vs best baseline = +0.56%`
  - `triple` 赢 `10/16`

### 可直接写进结果的结论

- `triple` 对 `BayesB`-best 和 `GBLUP`-best trait 的提升更稳定。
- 当前 `triple` 未能超过 strongest baseline 的 trait，全部集中在 `RKHS`-best 区域。
- 这说明融合框架并不是对所有遗传结构类型同样容易奏效；相较之下，`RKHS` 主导的 trait 可能仍保留了更多当前 prior-integrated fusion 尚未完全吸收的信息。

### 推荐图表

- Figure / Table:
  - 分三组箱线图：x 轴为 `best baseline type`，y 轴为 `triple vs best baseline (%)`
  - 或者小表格直接列 `n / mean gain / win rate`

## 2. Pearson + R2 双指标同步汇总

这项分析很重要，因为它可以证明 fusion 的收益不是某一个单一指标上的偶然现象。

### 当前统计

- `single_bayesb_two_step_ls vs only_single_bayesb`
  - Pearson: `+1.32%`
  - R2: `+3.07%`
  - 赢 `35/36`

- `single_gblup_two_step_ls vs only_single_gblup`
  - Pearson: `+1.62%`
  - R2: `+3.73%`
  - 赢 `35/36`

- `single_rkhs_two_step_ls vs only_single_rkhs`
  - Pearson: `+0.82%`
  - R2: `+2.62%`
  - 赢 `30/36`

- `dual_two_step_ls vs only_dual`
  - Pearson: `+1.39%`
  - R2: `+3.23%`
  - 赢 `35/36`

- `triple_two_step_ls vs only_triple`
  - Pearson: `+0.93%`
  - R2: `+2.26%`
  - 赢 `32/36`

### 可直接写进结果的结论

- 无论是 single、dual 还是 triple，fusion 相对对应 prior-only 的优势在 `Pearson` 和 `R2` 上均同步存在。
- `R2` 上的相对提升通常比 `Pearson` 更大，说明 fusion 不仅改善相关性，也改善解释方差能力。

### 推荐图表

- 双指标并列柱状图
- 或 Supplementary Table：每种 fusion 的 `Pearson gain / R2 gain / win rate`

## 3. triple vs best single 的“协同收益”分析

这项分析的意义不在于证明 `triple` 总是远超 best single，而在于更准确地定义它的价值：**稳定性与统一性。**

### 当前统计

- `triple - best single` 平均绝对差：`-0.000388`
- `triple >= best single`：`19/36`
- `triple > best single`：`13/36`
- 相对最佳 single 的平均百分比差：`-0.0746%`
- 中位数差：`0.0%`

### 可直接写进结果的结论

- `triple` 在平均上并未显著压倒 strongest single。
- 但其相对 best single 的差异几乎为零，中位数为 `0`，说明它在多数 trait 上可以达到与最佳 single 近似的水平。
- 因而，`triple` 的意义更多体现在：
  - 不需要事先知道哪条 single 最适合某个 trait
  - 提供一个统一、稳健的主结果入口

### 推荐图表

- `triple - best single` 的直方图或排序图
- 正负分布图

## 4. w_tabicl 的分层箱线图或排序图

这项分析已经具备完整数据基础，几乎是低成本高回报。

### 当前统计

- `single_bayesb`
  - 平均：`0.4926`
  - `Q1/median/Q3 = 0.4776 / 0.4932 / 0.5147`
  - 范围：`0.4326 - 0.5532`

- `single_gblup`
  - 平均：`0.4932`
  - `Q1/median/Q3 = 0.4752 / 0.4968 / 0.5158`
  - 范围：`0.4472 - 0.5363`

- `single_rkhs`
  - 平均：`0.5030`
  - `Q1/median/Q3 = 0.4748 / 0.5000 / 0.5401`
  - 范围：`0.4422 - 0.5602`

- `dual`
  - 平均：`0.4913`
  - `Q1/median/Q3 = 0.4702 / 0.4977 / 0.5154`
  - 范围：`0.4152 - 0.5387`

- `triple`
  - 平均：`0.4894`
  - `Q1/median/Q3 = 0.4705 / 0.4915 / 0.5118`
  - 范围：`0.4345 - 0.5492`

### 可直接写进结果的结论

- `w_tabicl` 在 single / dual / triple 三类融合中整体都稳定落在约 `0.49-0.50` 附近。
- 但其 trait-level 范围并不窄，说明不同 trait 对 TabICL 信息的依赖程度确实存在差异。
- `single_rkhs` 的上四分位略高，提示某些 trait 中 TabICL 相对 RKHS 的互补性更强。

### 推荐图表

- 箱线图：
  - x 轴：`single_bayesb / single_gblup / single_rkhs / dual / triple`
  - y 轴：`w_tabicl`
- 排序图：
  - 对 `triple` 的 `w_tabicl` 按 trait 排序

## 5. w_tabicl 和性能增益的关系图

这是解释机制最关键的一项，因为它把“权重变化”和“融合收益”联系了起来。

### 当前统计

以 `triple` 为例：

- `w_tabicl` 平均：`0.485`
- 范围：`0.434 - 0.549`
- `w_tabicl` 与 `no_prior-TabICL vs best baseline` 的相关：`r = 0.453`
- `w_tabicl` 与 `triple vs best baseline` 的相关：`r = 0.220`
- `w_tabicl` 与 `triple vs no_prior` 的相关：`r = -0.420`

将 trait 按 `w_tabicl` 分为三组后：

- 最低权重组：
  - `w_range = 0.4345 - 0.4763`
  - `triple vs best baseline = +0.57%`
  - `no_prior vs best baseline = -9.08%`
  - `triple vs no_prior = +11.67%`

- 中间权重组：
  - `w_range = 0.4777 - 0.5063`
  - `triple vs best baseline = +0.76%`
  - `no_prior vs best baseline = -4.12%`
  - `triple vs no_prior = +5.21%`

- 最高权重组：
  - `w_range = 0.5104 - 0.5492`
  - `triple vs best baseline = +1.22%`
  - `no_prior vs best baseline = -2.24%`
  - `triple vs no_prior = +3.62%`

### 可直接写进结果的结论

- 当 `no_prior-TabICL` 本身更接近 strongest baseline 时，融合倾向于分配更高的 `w_tabicl`。
- 当 `no_prior-TabICL` 本身较弱时，fusion 仍然可以带来很大的 `triple vs no_prior` 增益，但这主要通过“更多保留 prior、减少 TabICL 占比”来实现。
- 这说明 `w_tabicl` 不是一个被动的数学参数，而是 trait 对 TabICL 信息可利用程度的压缩表示。

### 推荐图表

- 散点图：
  - x 轴：`no_prior vs best baseline (%)`
  - y 轴：`w_tabicl`
- 散点图：
  - x 轴：`w_tabicl`
  - y 轴：`triple vs best baseline (%)`
- 或三分组柱状图

## 6. fusion 相对 no_prior 与相对 prior-only 的双轴比较

这项分析可以非常直观地区分：

- 哪些 trait 是 “TabICL 本身弱，但 fusion 把它救回来”
- 哪些 trait 是 “TabICL 本身不弱，因此 fusion 进一步利用了它”

### 当前象限统计

- `single_bayesb`
  - 同时优于 `no_prior` 和 `only_single_bayesb`：`35`
  - 只优于 `no_prior`：`1`
  - 只优于 prior-only：`0`
  - 两者都不优：`0`

- `single_gblup`
  - 同时优于二者：`35`
  - 只优于 `no_prior`：`1`

- `single_rkhs`
  - 同时优于二者：`30`
  - 只优于 `no_prior`：`6`

- `dual`
  - 同时优于二者：`35`
  - 只优于 `no_prior`：`1`

- `triple`
  - 同时优于 `no_prior` 和 `only_triple`：`32`
  - 只优于 `no_prior`：`4`
  - 只优于 `only_triple`：`0`
  - 两者都不优：`0`

### 可直接写进结果的结论

- `triple` 的大多数 trait 同时优于 `no_prior` 和 `only_triple`，说明它并不是在二者之间做折中，而是实质性整合了两边信息。
- 未出现“优于 prior-only 但不如 no_prior”的 trait，说明当前融合并没有以牺牲 TabICL 贡献为代价换取 prior 增益。

### 推荐图表

- 双轴散点图：
  - x 轴：`fusion - no_prior`
  - y 轴：`fusion - prior-only`
  - 每个点为一个 trait

## 7. trait 类型三角分析

这是最值得新加的一层，因为它把 “什么样的 trait 更适合 no-prior / fusion” 这个问题转化成了一个可量化的 baseline geometry 问题。

## 7.1 思路

把每个 trait 的三条 baseline 结果：

- `BayesB`
- `GBLUP`
- `RKHS`

归一化成三元坐标：

- `BayesB_share`
- `GBLUP_share`
- `RKHS_share`

这样每个 trait 都可以放到一个 3-corner simplex 里。  
这个图的含义不是“真实生物机制被完全识别”，而是用三类统计先验的相对优势，作为 trait 建模结构的经验代理。

## 7.2 当前统计相关性

### 与 no-prior 表现的相关

- `BayesB_share` 与 `no_prior vs best baseline`：`r = +0.387`
- `GBLUP_share` 与 `no_prior vs best baseline`：`r = +0.593`
- `RKHS_share` 与 `no_prior vs best baseline`：`r = -0.645`

### 与 triple 提升的相关

- `BayesB_share` 与 `triple vs best baseline`：`r = +0.344`
- `GBLUP_share` 与 `triple vs best baseline`：`r = +0.092`
- `RKHS_share` 与 `triple vs best baseline`：`r = -0.299`

### 可直接写进结果的结论

- 当一个 trait 在 baseline 几何上更偏 `BayesB / GBLUP` 一侧时，`no_prior-TabICL` 往往相对更接近 strongest baseline。
- 当 trait 更偏 `RKHS` 一侧时，`no_prior-TabICL` 相对 strongest baseline 的劣势会更明显。
- `triple` 在 `RKHS` 偏高的 trait 上依然能显著提高相对 `no_prior` 的表现，但其相对 strongest baseline 的边际优势会缩小。

## 7.3 简化分区分析

将 trait 按三 baseline 的归一化 share 做粗分区后：

- `BayesB_dominant`
  - `n = 8`
  - `no_prior vs best baseline = -4.07%`
  - `triple vs best baseline = +1.09%`
  - `triple vs no_prior = +5.46%`
  - `triple` 赢 `8/8`

- `RKHS_dominant`
  - `n = 10`
  - `no_prior vs best baseline = -9.08%`
  - `triple vs best baseline = +0.46%`
  - `triple vs no_prior = +11.76%`
  - `triple` 赢 `7/10`

- `balanced_or_weakly_separated`
  - `n = 18`
  - `no_prior vs best baseline = -3.44%`
  - `triple vs best baseline = +0.97%`
  - `triple vs no_prior = +4.72%`
  - `triple` 赢 `15/18`

### 可直接写进结果的结论

- `RKHS` 倾向更强的 trait，是 `no_prior-TabICL` 最不占优的一类 trait。
- 但也正是在这类 trait 上，`triple vs no_prior` 的提升最大，说明 fusion 在这里具有更强的“补救”作用。
- 相比之下，`BayesB`-dominant trait 中 `no_prior` 本身并不算太差，而 `triple` 更容易直接超过 strongest baseline。

## 7.4 baseline 分歧度指标

还可以给三角分析再加一个更简洁的单值指标：

- `entropy`：三 baseline 越接近，entropy 越高
- `spread`：三 baseline 差距越大，spread 越大

当前统计：

- `entropy` 与 `no_prior vs best baseline`：`r = +0.825`
- `entropy` 与 `triple vs best baseline`：`r = +0.288`
- `spread` 与 `no_prior vs best baseline`：`r = -0.430`
- `spread` 与 `triple vs best baseline`：`r = -0.262`

### 可直接写进结果的结论

- 当三类统计先验的结果更接近时，`no_prior-TabICL` 往往表现得更好。
- 当某一统计模型明显突出、baseline 分歧更大时，`no_prior` 更容易落后。
- 这为 “什么样的 trait 更适合 no-prior，什么样的 trait 更需要 fusion” 提供了一个简洁且可操作的经验判据。

## 8. 最推荐进入正文的补充分析

如果只选最值得进主文的补充项，建议优先放这 5 个：

1. 按 `best baseline type` 分层
2. `Pearson + R2` 双指标同步汇总
3. `triple vs best single` 的协同收益分析
4. `w_tabicl` 与性能增益关系图
5. `trait` 三角分区分析

## 9. 更适合放 Supplementary 的补充分析

1. `w_tabicl` 的完整排序图
2. `fusion vs no_prior` 与 `fusion vs prior-only` 的双轴象限图
3. baseline entropy / spread 与性能关系
4. dataset-level 分层结果

## 10. 一句话总结

基于当前结果，最值得强调的新信息是：

- **`no_prior-TabICL` 更适合那些三类统计 baseline 差异较小、或更偏 BayesB / GBLUP 的 trait；**
- **而当 trait 更偏 RKHS 主导时，`no_prior` 本身会明显变弱，但 `fusion` 尤其是 `triple` 反而能带来更大的补偿性提升。**

这条线非常适合把 `Result 2-3` 从“模型比拼”提升到“trait 类型与融合机制”的层面。
