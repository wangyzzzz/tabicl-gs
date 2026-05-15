# 5.4 解偶复用结果章节大纲

## 使用范围

- 本文档用于固定当前 `5.4-duli-liudang` 主线的结果章节结构
- 当前主结果默认 `exclude pig3534`
- 当前主结果表：
  - `outputs/5.4-duli-liudang/main_results_non_pig_fixed.csv`
  - `outputs/5.4-duli-liudang/main_results_non_pig_fixed.json`

## 主结果方法

- `single_bayesb_two_step_ls`
- `single_gblup_two_step_ls`
- `single_rkhs_two_step_ls`
- `triple_two_step_ls`

说明：

- `single_*_two_step_ls` 表示“单 prior + TabICL”的主结果
- `triple_two_step_ls` 表示“3 prior + TabICL”的主结果
- 当前不把 `only_triple` 作为主结果，因为其相对 strongest baseline 并不稳定占优

## 推荐章节结构

### 1. TabICL 单独表现与引入 prior 的动机

目标：

- 先交代 `no_prior-TabICL` 单独作为预测器时，结果具有竞争力，但并不稳定超过强统计模型
- 明确论文主线不是“TabICL 替代统计模型”，而是“TabICL 利用统计模型先验实现进一步增益”

建议放的内容：

- `no_prior-TabICL` vs `BayesB / GBLUP / RKHS`
- trait-level Pearson 主表或热图
- 一段简短文字说明：不同 trait 的 strongest baseline 不同，因此需要融合框架

建议图表：

- Figure 1: `no_prior-TabICL` 与 3 个 baseline 的 trait-level 对比图
- Table S1: 全部 trait 的 `no_prior` 与 baseline 数值表

### 2. 单 prior 融合结果

目标：

- 说明 `TabICL` 与单一统计 prior 融合后，通常可以稳定超过该 prior 本身
- 这是主线的第一层证据：TabICL 不是单独最强，但它能把已有 prior 再抬高

主结果方法：

- `single_bayesb_two_step_ls`
- `single_gblup_two_step_ls`
- `single_rkhs_two_step_ls`

建议放的内容：

- 各 single 方法相对自身 prior 的提升百分比
- 各 single 方法相对 strongest baseline 的结果
- 强调：
  - `single_bayesb_two_step_ls` 相对 `BayesB` 几乎全面增益
  - `single_gblup_two_step_ls` 相对 `GBLUP` 几乎全面增益
  - `single_rkhs_two_step_ls` 相对 `RKHS` 也多为正提升，但波动更大

建议图表：

- Figure 2: 三条 single-prior-TabICL 相对各自 prior 的提升分布图
- Table 1: 主结果表中的三条 single 方法
- Table S2: 每个 trait 的 single 相对 prior 百分比

### 3. 多 prior 融合结果

目标：

- 这是全文主结果章节
- 说明单 prior 之外，多 prior 融合进一步提高了整体稳定性
- 最终主结果固定为 `triple_two_step_ls`

主结果方法：

- `triple_two_step_ls`

辅助对照：

- `dual_two_step_ls`
- `only_triple`

建议放的内容：

- `triple_two_step_ls` 相对 strongest baseline 的提升
- `dual_two_step_ls` 与 `triple_two_step_ls` 的对比
- `only_triple` 与 `triple_two_step_ls` 的对比

当前已经明确的叙事点：

- `only_triple` 本身对 strongest baseline 不稳定占优
- 真正稳定更强的是 `triple prior + TabICL`
- 这说明提升不是简单来源于 prior 线性加权，而是来源于 `TabICL + prior` 的互补融合

建议图表：

- Figure 3: `single / dual / triple` 主结果对比图
- Figure 4: `triple_two_step_ls` 相对 strongest baseline 的提升分布图
- Table 2: `triple_two_step_ls` 主结果表
- Table S3: `dual / triple / only_triple` 逐 trait 百分比表

### 4. 权重与机制分析

目标：

- 解释为什么融合有效
- 展示不同 prior 与 TabICL 在不同 trait 中的相对贡献

建议放的内容：

- single / dual / triple 的权重分布
- `only_triple` 与 `triple_two_step_ls` 的差异
- 为什么 `TabICL alone` 不一定最强，但加入融合后有效

推荐聚焦问题：

- 为什么 `single + TabICL` 相对 prior-only 基本全面提升
- 为什么 `only_triple` 不稳定超过 strongest baseline，而 `triple + TabICL` 可以
- 为什么 `two_step_ls` 比 `clip` 更稳，且通常优于 `all_ls`

建议图表：

- Figure 5: single / triple 的权重分布图
- Figure 6: `only_triple` vs `triple_two_step_ls` 对比图
- Table S4: 权重统计表

## 当前建议图表清单

- Figure 1: `no_prior-TabICL` vs `BayesB / GBLUP / RKHS`
- Figure 2: 三条 single-prior-TabICL 相对各自 prior 的提升分布
- Figure 3: `single / dual / triple` 主结果对比
- Figure 4: `triple_two_step_ls` 相对 strongest baseline 的提升分布
- Figure 5: 主结果权重分布图
- Figure 6: `only_triple` vs `triple_two_step_ls`

- Table 1: 三条 single-prior-TabICL 主结果表
- Table 2: `triple_two_step_ls` 主结果表
- Table S1: 全 trait baseline 与 `no_prior-TabICL`
- Table S2: single 相对自身 prior 的提升百分比
- Table S3: dual / triple / only_triple 百分比总表
- Table S4: 权重统计表

## 当前推荐写法

- 不要把主线写成“TabICL 单独超过统计模型”
- 要写成：
  - `TabICL alone` 具有竞争力，但不是稳定最优
  - 统计模型 prior 并非被替代，而是被 `TabICL` 有效利用
  - `single-prior-TabICL` 已可稳定提高 prior-only 结果
  - `triple-prior-TabICL` 进一步提高 strongest baseline 之上的表现

## 当前建议固定结论

- 当前主结果固定为：
  - `single_bayesb_two_step_ls`
  - `single_gblup_two_step_ls`
  - `single_rkhs_two_step_ls`
  - `triple_two_step_ls`
- 默认 `exclude pig3534`
- 后续正文、图表、补充材料优先围绕这组结果展开
