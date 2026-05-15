# SNP Count 实验总结（可纳入正文）

## 1. 实验目的

本实验用于评估在不同 SNP 数量下，解偶复用框架中 TabICL 融合策略的稳定性与收益变化。核心问题是：当标记数从低密度提升到高密度时，`single-prior-TabICL` 与 `triple-prior-TabICL` 是否仍能稳定优于对应统计基线，以及提升幅度是否随 marker 数量发生系统性变化。

## 2. 实验设计

- 实验对象：4 个数据集、共 8 个 traits，每个数据集选取 2 个代表性 traits
- 数据集与 traits：
  - `cotton1245`
    - `cotton_fiblen_17_18_cotton_fiblen_17_18`
    - `cotton_fibelo_17_18_cotton_fibelo_17_18`
  - `rice529`
    - `grain_weight`
    - `grain_width`
  - `soybean951`
    - `lw_beijing_2013_lw_beijing_2013`
    - `bbd_beijing_2013_bbd_beijing_2013`
  - `wheat406`
    - `sl_e1`
    - `sl_e2`
- SNP 数量设置：
  - `2K`
  - `10K`
  - `50K`
- `10K` 直接复用主线 `5.4-duli-liudang` 结果
- `2K / 50K` 采用新的解偶复用产线重新运行
- 对于 `2K / 50K`：
  - 先基于对应 marker 数重新搜索 `best block`
  - 再只正式留档 4 类底层结果：
    - `no_prior-TabICL`
    - `BayesB`
    - `GBLUP`
    - `RKHS`
  - 所有 `single / triple` 融合结果均从留档结果直接计算，不重复训练融合底层模型
- 评价指标：5-fold outer test Pearson 相关系数

## 3. 主结果口径

- baseline 固定为：
  - `BayesB`
  - `GBLUP`
  - `RKHS`
- 主要融合方法：
  - `single_bayesb_two_step_ls`
  - `single_gblup_two_step_ls`
  - `single_rkhs_two_step_ls`
  - `triple_two_step_ls`
- 正式汇总文件：
  - `outputs/5.4-marker_count-decoupled/marker_count_main_results.csv`

## 4. 整体均值结果

8 个 traits 的平均准确率如下：

| Marker count | no_prior | BayesB | GBLUP | RKHS | single-BayesB | single-GBLUP | single-RKHS | triple |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 0.589012 | 0.602834 | 0.601697 | 0.614662 | 0.615576 | 0.614565 | 0.618470 | 0.620091 |
| 10K | 0.609656 | 0.631417 | 0.620837 | 0.627129 | 0.641733 | 0.634098 | 0.635466 | 0.643458 |
| 50K | 0.619505 | 0.659904 | 0.632763 | 0.635205 | 0.667434 | 0.646130 | 0.645664 | 0.667544 |

可以看到，随着 marker 数量增加，`triple_two_step_ls` 的平均性能呈持续上升趋势：

- `2K`: `0.6201`
- `10K`: `0.6435`
- `50K`: `0.6675`

这说明在更高 SNP 密度下，融合框架整体仍然有效，并且上限继续提高。

## 5. Triple 融合相对 baseline 的整体提升

`triple_two_step_ls` 相对 3 个 baseline 的平均提升百分比如下：

| Marker count | vs BayesB | vs GBLUP | vs RKHS |
|---|---:|---:|---:|
| 2K | 2.920449% | 3.228987% | 0.821779% |
| 10K | 2.339403% | 4.127037% | 2.710999% |
| 50K | 1.475714% | 6.471139% | 5.644786% |

这组结果有两个值得强调的趋势：

1. `triple_two_step_ls` 在三个 marker 档位上都保持对 `GBLUP` 和 `RKHS` 的稳定正提升。
2. 当 marker 数增加到 `50K` 时，`triple_two_step_ls` 相对 `GBLUP` 和 `RKHS` 的平均优势反而进一步扩大，分别达到 `6.47%` 和 `5.64%`。

这说明 TabICL 融合框架并不是只在低 marker 或信息不足的情况下有效；当 marker 更丰富时，它仍然能够从多先验中继续提取互补信息。

## 6. Single 融合相对各自 prior 的整体提升

三个 `single-prior-TabICL` 相对各自 prior 的平均提升百分比如下：

| Marker count | single-BayesB vs BayesB | single-GBLUP vs GBLUP | single-RKHS vs RKHS |
|---|---:|---:|---:|
| 2K | 2.059867% | 2.135542% | 0.508308% |
| 10K | 1.916495% | 2.454059% | 1.515498% |
| 50K | 1.445347% | 2.574194% | 1.954497% |

可见：

- `single_gblup_two_step_ls` 的提升最稳定，而且从 `2K -> 10K -> 50K` 呈逐步增加趋势。
- `single_rkhs_two_step_ls` 在低 marker 时提升较小，但在 `50K` 时提升明显增强。
- `single_bayesb_two_step_ls` 在三档 marker 下均保持正提升，但随 marker 数增加，其平均增幅有所收敛。

因此，从单 prior 融合的角度看，TabICL 并非只对某一类先验有效，而是对三类统计模型均能提供额外增益，只是增益模式不同。

## 7. 按 trait 的稳定性观察

从 8 个 traits 的 `triple_two_step_ls` 结果看：

- 在 `10K` 条件下，`triple_two_step_ls` 在 `8/8` 个 traits 上都优于最佳 baseline。
- 在 `2K` 条件下，`triple_two_step_ls` 在 `7/8` 个 traits 上优于最佳 baseline。
  - 唯一例外是 `wheat406/sl_e2`，其 `triple` 略低于 `RKHS`。
- 在 `50K` 条件下，`triple_two_step_ls` 在 `6/8` 个 traits 上优于最佳 baseline。
  - 两个例外均来自 `rice529`：
    - `grain_weight`
    - `grain_width`
  - 这两个 traits 在 `50K` 时 `BayesB` 已非常强，因此 `triple` 虽然仍优于 `GBLUP` 与 `RKHS`，但未继续超过 `BayesB`。

进一步看 `triple` 随 marker 数量变化的趋势：

- 8 个 traits 中，有 6 个呈现 `2K <= 10K <= 50K` 的单调上升趋势：
  - `cotton_fiblen`
  - `cotton_fibelo`
  - `grain_width`
  - `lw_beijing_2013`
  - `sl_e1`
  - `sl_e2`
- 另外 2 个 traits 不完全单调：
  - `rice529/grain_weight`
  - `soybean951/bbd_beijing_2013`

这说明更高的 marker 数量通常有利于融合模型表现，但具体收益仍受 trait 本身遗传结构和 prior 竞争关系影响。

## 8. 可写入正文的核心结论

可以将本实验的主要信息概括为以下几点：

1. `triple-prior-TabICL` 在 `2K / 10K / 50K` 三档 marker 数量下都保持较强竞争力，平均性能随 SNP 数量增加而持续提升。
2. 在 `10K` 条件下，`triple-prior-TabICL` 对全部 8 个 traits 均优于最佳 baseline，显示出较强的稳定性。
3. 在 `50K` 条件下，虽然个别 traits 上 `BayesB` 变得更强，但 `triple-prior-TabICL` 相对 `GBLUP` 和 `RKHS` 的平均优势进一步扩大，说明融合框架在高密度 marker 情况下仍具有显著增益。
4. 三类 `single-prior-TabICL` 均能相对各自统计先验提供额外提升，其中 `single_gblup_two_step_ls` 的平均提升最稳定。
5. 整体上，marker 数增加并不会削弱 TabICL 融合框架的价值；相反，在更高 SNP 密度下，多先验融合仍能持续提取互补信息。

## 9. 建议正文表述

正文中建议避免把该实验表述为“单纯增加 marker 就一定单调提升所有 traits”，更稳妥的写法是：

> Across the 8 representative traits, increasing SNP density from 2K to 10K and 50K generally improved the performance of the decoupled TabICL fusion framework. The averaged accuracy of `triple-prior-TabICL` increased consistently across the three marker settings, while its relative advantage over `GBLUP` and `RKHS` remained positive and became more pronounced at 50K markers. Although a few traits were dominated by BayesB under the highest marker density, the overall results indicate that the proposed fusion framework remains effective and scalable as marker information becomes richer.

如果正文需要中文概括，可使用：

> SNP 数量消融实验表明，随着 marker 数从 2K 增加到 10K 和 50K，`triple-prior-TabICL` 的平均预测准确率持续上升，说明该融合框架能够在更高密度的遗传标记条件下继续提取有效信息。尽管在少数 traits 上高密度 marker 会进一步强化 BayesB 的单模型优势，但从整体均值看，`triple-prior-TabICL` 相对 `GBLUP` 和 `RKHS` 的提升幅度在 50K 条件下反而更大，说明 TabICL 的多先验融合能力并不会因 marker 丰富而失效，而是具有较好的可扩展性。
