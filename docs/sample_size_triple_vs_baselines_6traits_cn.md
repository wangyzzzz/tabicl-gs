# 样本量变化下 triple-prior-TabICL 相对 baseline 提升规律总结（当前 6 traits 版）

## 适用范围

- 本文档用于汇总当前样本量实验中，`triple_two_step_ls` 相对 `BayesB / GBLUP / RKHS` 的提升变化规律
- 当前口径默认：
  - 正式结果以 `server@GPU1` 为准
  - 默认 `exclude pig3534`
  - baseline 固定为 `BayesB / GBLUP / RKHS`
- 当前仅基于已经完整落盘的 6 个 trait 统计
- `wheat406/sl_e1` 和 `wheat406/sl_e2` 仍在运行中，暂未纳入本版分析

## 数据来源

- `20% / 60%`：
  - `outputs/5.4-sample_size-decoupled`
  - 每个 trait 基于 `repeat = 3` 的均值
- `100%`：
  - `outputs/5.4-duli-liudang/main_results_non_pig_fixed.csv`

说明：

- 相对提升百分比定义为：
  - `(triple_two_step_ls - baseline) / baseline × 100%`
- 正值表示 `triple-prior-TabICL` 优于对应 baseline
- 负值表示 `triple-prior-TabICL` 劣于对应 baseline

## 当前已完整完成的 6 个 traits

- `cotton1245 / cotton_fibelo_17_18_cotton_fibelo_17_18`
- `cotton1245 / cotton_fiblen_17_18_cotton_fiblen_17_18`
- `rice529 / grain_weight`
- `rice529 / grain_width`
- `soybean951 / bbd_beijing_2013_bbd_beijing_2013`
- `soybean951 / lw_beijing_2013_lw_beijing_2013`

## 主表：triple-prior-TabICL 相对 3 个 baseline 的提升（%）

| trait | sample | vs BayesB | vs GBLUP | vs RKHS |
|---|---:|---:|---:|---:|
| cotton_fibelo | 20% | +1.91 | +2.47 | +1.40 |
| cotton_fibelo | 60% | +1.46 | +2.56 | +1.70 |
| cotton_fibelo | 100% | +1.09 | +3.20 | +2.82 |
| cotton_fiblen | 20% | +2.50 | +3.24 | +4.49 |
| cotton_fiblen | 60% | +1.90 | +4.52 | +4.55 |
| cotton_fiblen | 100% | +1.78 | +3.69 | +2.63 |
| grain_weight | 20% | +7.04 | +19.57 | +0.31 |
| grain_weight | 60% | +5.14 | +3.53 | +5.47 |
| grain_weight | 100% | +3.60 | +4.82 | +4.63 |
| grain_width | 20% | +0.04 | +0.39 | -0.89 |
| grain_width | 60% | -0.07 | +0.75 | +1.01 |
| grain_width | 100% | +0.03 | +1.07 | +1.61 |
| bbd | 20% | +0.91 | +2.39 | +0.51 |
| bbd | 60% | +0.35 | +1.41 | +0.56 |
| bbd | 100% | +0.07 | +1.30 | +1.26 |
| lw | 20% | -1.16 | -0.35 | -1.46 |
| lw | 60% | +0.86 | +2.11 | +2.13 |
| lw | 100% | +1.66 | +5.75 | +4.58 |

## 可直接写进正文的规律总结

### 1. 相对 BayesB 的提升，整体更倾向于随样本量增加而收敛

在 6 个 trait 中，`triple-prior-TabICL` 相对 `BayesB` 的增益大多集中在小幅正提升区间，且在不少 trait 上会随着样本量从 `20%` 增加到 `100%` 而逐渐减弱。例如，`cotton_fibelo` 的提升从 `+1.91%` 下降到 `+1.09%`，`cotton_fiblen` 从 `+2.50%` 下降到 `+1.78%`，`bbd` 从 `+0.91%` 下降到 `+0.07%`。这说明当训练样本更加充足时，`BayesB` 本身已经能够较充分提取加性或稀疏大效应信号，`triple-prior-TabICL` 的额外增益空间会有所压缩。

### 2. 相对 GBLUP 和 RKHS 的提升，更容易在中高样本量下保持或扩大

与 `BayesB` 相比，`triple-prior-TabICL` 相对 `GBLUP` 和 `RKHS` 的提升并不简单表现为单调收缩，反而在若干 trait 上会随样本量增加而增强。例如，`cotton_fibelo` 相对 `GBLUP` 的提升从 `+2.47%` 增加到 `+3.20%`，相对 `RKHS` 的提升从 `+1.40%` 增加到 `+2.82%`；`grain_width` 相对 `RKHS` 从 `-0.89%` 逐步转为 `+1.61%`；`lw` 相对 `GBLUP` 从 `-0.35%` 增加到 `+5.75%`。这一现象说明，随着样本量上升，`TabICL` 与多 prior 之间的互补性在部分 trait 上会更加充分显现，尤其是相对于结构更“平滑”的统计模型时更明显。

### 3. 小样本场景下，融合对难 trait 的补偿可能更强，但波动也更大

`rice529 / grain_weight` 是最典型的难 trait。该 trait 在 `20%` 样本量下，`triple-prior-TabICL` 相对 `GBLUP` 的提升高达 `+19.57%`，相对 `BayesB` 也有 `+7.04%`。这说明在小样本场景下，当 baseline 对复杂信号拟合不足时，多 prior 与 `TabICL` 的融合可以显著补偿 baseline 的缺失信息。不过，这类 trait 在不同样本量下的提升幅度变化也更大，表明其收益虽然显著，但稳定性仍然受任务难度影响。

### 4. 并非所有 trait 都会随着 prior 数量增加而持续获益

`grain_width` 和 `lw` 表明，`triple-prior-TabICL` 并不是在所有性状上都天然优于单一强 baseline。`grain_width` 在 `20%` 样本量下相对 `RKHS` 仍为负提升，`lw` 在 `20%` 下相对三条 baseline 也全部为负。说明 prior 的有效性首先取决于 trait 本身的遗传结构与 prior 匹配程度，而不是 prior 数量越多越一定更优。也就是说，多 prior 融合是有条件受益，而不是无条件受益。

### 5. 更准确的主线表述应是“融合收益具有 trait 依赖性和样本量依赖性”

从当前 6 个 trait 看，`triple-prior-TabICL` 的优势并不适合被表述为“样本量越大提升越大”或“样本量越小提升越大”这样单一方向的规律。更合适的表述是：

- 相对 `BayesB`，增益通常较小，并且更容易在大样本下收敛
- 相对 `GBLUP` 和 `RKHS`，增益更具 trait 依赖性，在部分 trait 上会随样本量增加而增强
- 在难 trait 或 baseline 不完全匹配的 trait 上，多 prior 融合更容易表现出显著收益

## 建议写入正文的简洁版本

可以在结果部分写成下面这类表述：

> 为进一步评估融合收益是否受训练样本量影响，我们在当前已完整完成的 6 个性状上比较了 `triple-prior-TabICL` 在 `20%`、`60%` 和 `100%` 训练样本下相对 `BayesB`、`GBLUP` 和 `RKHS` 的表现。结果表明，融合收益具有明显的 trait 依赖性和样本量依赖性。相对 `BayesB` 的提升通常较小，且在部分性状上会随样本量增加而收敛；而相对 `GBLUP` 和 `RKHS` 的提升在若干性状上则可随样本量增加而进一步扩大。尤其是在 `grain_weight` 等较难性状上，多 prior 融合在小样本条件下可提供更显著的性能补偿。总体而言，这些结果支持本文主线，即 `TabICL` 的价值不在于单独替代统计模型，而在于能够利用多个统计 prior 的互补信息，在不同样本量条件下获得额外预测增益。

## 更保守的讨论写法

如果希望正文写得更谨慎，可以用下面这段：

> 需要指出的是，当前样本量分析尚有两个 wheat traits 正在运行，因此现阶段结论仍基于已完整完成的 6 个 traits。尽管如此，现有结果已经显示出一致趋势：`triple-prior-TabICL` 相对 baseline 的收益并非固定不变，而是同时受到 trait 遗传结构和训练样本量的影响。这一现象提示，prior 融合的价值主要体现在对不同统计建模偏好的整合，以及对 `TabICL` 表征能力的补充，而不应被简单理解为“prior 越多越一定更优”。

## 当前推荐使用方式

- 这份文档适合直接作为正文结果段和讨论段的中文素材
- 等 `wheat406/sl_e1` 和 `sl_e2` 完成后，建议把表格更新为 8 traits 完整版
- 如果要在主文中放图，最推荐的是把这张表进一步画成：
  - 每个 trait 一条折线
  - 横轴为 `20% / 60% / 100%`
  - 纵轴为 `triple` 相对 baseline 的提升百分比

