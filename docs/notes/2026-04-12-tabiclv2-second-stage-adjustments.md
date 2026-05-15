# TabICLv2 第二层调整备忘

日期: 2026-04-12

## 背景

当前两层模型的主线判断已经比较清楚:

- 第一层继续使用 `TabICLv2`
- 分块方式固定为 `window`
- `group_size` 不再作为当前主矛盾, 先固定在合理范围内
- 第一层重点继续研究:
  - `PC` 维度
  - `ensemble`
- 第二层先保持稳定, 不优先扩展到三层

与此同时, 第一层现在已经能够输出和保存以下 block 级诊断量:

- `explained_variance_ratio_sum`
- `embedding_var`
- `scalar_var`
- `scalar_train_pearson`
- `scalar_train_r2`

这些量已经足够支持第二层做质量感知聚合, 不应该再把所有 block 一视同仁。

## 当前判断

第二层后续最值得做的不是直接换更复杂的模型结构, 而是先做质量感知调整。

核心原因:

1. 第一层已经提供了 block 质量画像
2. 第二层当前只是盲拼接 block 特征
3. 盲拼接会浪费稳定性/有效性信息
4. 直接引入更复杂的第二层结构, 很容易增加过拟合并降低解释性

## 推荐的第二层调整方向

### 方向 A: block reweighting

在第二层输入前, 对每个 block 的向量表示乘一个权重:

- `z_b -> w_b * z_b`

其中 `w_b` 根据 block 诊断量生成。

建议先做规则型权重, 不先上可学习 gating:

- 高质量 block: `w = 1.0`
- 中质量 block: `w = 0.7`
- 低质量 block: `w = 0.4`

这是最稳妥的第一步。

### 方向 B: diagnostics 直接拼进第二层

每个 block 不只输入:

- `block embedding`
- `block scalar`

还输入:

- `explained_variance_ratio_sum`
- `embedding_var`
- `scalar_var`
- `scalar_train_pearson`
- `scalar_train_r2`

这样第二层能显式感知:

- 哪些 block 稳定
- 哪些 block 有效
- 哪些 block 可能是高噪声块

### 方向 C: 根据 block 质量做动态维度分配

这一步比单纯 PCA 更进一步:

- 高质量 block -> 分配更多维度
- 中质量 block -> 分配中等维度
- 低质量 block -> 分配更少维度

建议后续先尝试三档离散分配, 不要直接做连续映射。

## 对 block 质量的推荐理解方式

### 1. `explained_variance_ratio_sum`

含义:

- 当前保留维度下, 这个 block 的原始表示保留了多少方差

解释:

- 高值: 说明压缩保留了较多信息
- 低值: 说明当前维度可能压得过狠

### 2. `embedding_var`

含义:

- 512 维 block embedding 在 ensemble 多个视角下的平均方差

解释:

- 低值: 表示更稳定
- 高值: 表示更敏感, 可能更复杂, 也可能更噪声

### 3. `scalar_var`

含义:

- block scalar prediction 在 ensemble 多个视角下的平均方差

解释:

- 低值: block 判断更稳
- 高值: block 结论不稳定

### 4. `scalar_train_pearson` / `scalar_train_r2`

含义:

- block scalar 对当前 trait 的训练折拟合能力

解释:

- 高值: block 对当前 trait 更有用
- 低值: block 贡献较弱

## 推荐的 block 分层逻辑

第一版先不要直接学习权重, 而是用规则型三档:

### 高质量 block

满足:

- `scalar_train_r2` 较高
- `scalar_var` 较低
- `embedding_var` 较低
- `explained_variance_ratio_sum` 较高

处理:

- 分配较高维度
- 权重较大

### 中质量 block

处理:

- 中等维度
- 中等权重

### 低质量 block

处理:

- 更低维度
- 更低权重
- 需要时可只保留 scalar

## 当前不建议优先做的事

### 不建议立刻上三层

原因:

- 当前主瓶颈仍在第一层表示质量
- 两层内部机制还没完全收敛
- 三层会显著增加解释复杂度和过拟合风险

### 不建议先上可学习 gating 或 MLP 压缩

原因:

- 新增可训练参数
- 样本规模下更容易过拟合
- 不利于判断提升到底来自质量指标还是额外容量

## 推荐的执行顺序

1. 固定当前两层主线
2. 先做 block reweighting
3. 再做 diagnostics 拼接
4. 再做动态维度分配
5. 最后才考虑更复杂的第二层结构

## 暂定最小实验版本

### Version 1

- 第二层仍为 `TabICLv2`
- block embedding 先做简单加权
- diagnostics 不进入第二层

目的:

- 验证质量感知权重本身是否有效

### Version 2

- 第二层仍为 `TabICLv2`
- block embedding 加权
- diagnostics 作为附加特征拼接进入第二层

目的:

- 验证第二层是否能够利用这些质量指标进一步提高性能

### Version 3

- 根据 block 质量做三档动态维度
- 第二层输入维度不再统一

目的:

- 验证 block 质量是否应该直接决定 block 容量

## 目前保留结论

- 第二层继续使用 `TabICLv2` 仍然是主线方案
- 但第二层不应继续盲拼接所有 block
- 第一层已有诊断量足够支持 block 质量感知聚合
- 后续应优先做质量感知版本, 而不是盲目增加层数
