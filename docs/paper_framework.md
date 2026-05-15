# TabICL-GS 论文框架

## 一句话定位

提出一个基于 Tabular Foundation Model 的两阶段基因组预测框架，通过 block-wise TabICL 表征构建数据驱动的 SNP block 表示，并结合 BayesB-guided block grouping、sample-level group feature 与 group-shared gate，将 TabICL prediction、GBLUP 的 polygenic prior 和 BayesB 的 sparse marker-effect prior 进行 OOF 校准融合，最后提供多分辨率的可解释性分析。

---

## 整体架构

```
SNP (0/1/2 加性编码)
       │
       ▼
┌─────────────────────────┐
│  Block Partitioning     │  超参数搜索确定 block size
│  (搜索最优 block size)   │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│  第一层: Block-wise     │  每个 block 独立过 TabICL
│  TabICL Encoding        │  取倒数第二层表征
│  (核心: FM 表征提取)     │  PCA 降维 → block representation
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│  第二层: Aggregation     │  拼接所有 block repr
│  Expert Prediction      │  → XGBoost / TabICL expert prediction
│  (block 级预测来源)       │  作为 data-driven prediction source
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Dual-prior Gate        │  BayesB beta → block prior groups
│  (OOF calibrated)       │  sample group feature → group-specific weights
│                         │  TabICL pred + GBLUP pred + BayesB pred
└─────────────────────────┘
       │
       ▼
    Final Prediction
```

---

## Part 1: 预测性能与模型动机

### 叙事逻辑：先展示完整方法，再解释组件

Part 1 不再从第一层 TabICL 表征开始讲，而是先让读者看到最终预测框架的表现，再解释为什么需要 dual prior，最后回到第一层表征本身做机制拆解。这样更符合 Results 的阅读顺序：先回答“模型有没有用”，再回答“为什么这样设计”。

```
Step 1: Two-stage TabICL-GS vs classical GS baselines
  → 核心问题：两层模型本身能否达到有竞争力的基因组预测性能？
  → 对比 GBLUP / BayesB / BayesA / BayesLasso 等经典 baseline
  → 结论重点：不是宣称全面 SOTA，而是支持 TabICL block representation 是有效的预测来源

Step 2: 为什么引入 dual prior
  → 观察：Two-stage TabICL-GS 在不同 trait / fold 上表现有差异
  → 动机：单一 data-driven block representation 不一定覆盖所有遗传架构
  → 设计：引入 GBLUP 的 polygenic prior 与 BayesB 的 sparse marker-effect prior
  → 进一步设计：BayesB 先定义 block prior groups，sample 再根据自己在这些 block groups 上的响应模式形成 group feature
  → 结论重点：group-shared gate 用 OOF 学到的组特异权重整合互补预测来源，提高稳定性和泛化表现

Step 3: 回到第一层 TabICL，解释两层模型为什么 work
  → 核心问题：第一层 TabICL block representation 是否比 raw SNP 或 PCA 更有预测信息？
  → 固定多个简单下游模型，只改变输入特征表示
  → 结论重点：提升来自 representation，而不只是第二层模型或 fusion

Step 4: Ablation and robustness
  → 验证 block size、PCA 维度、第一层输出类型、Bayes prior 选择等因素
```

### 1.1 Two-stage TabICL-GS vs baseline

**核心问题：** 第一层 TabICL block representation + 第二层聚合模型，是否能作为一个独立的基因组预测模型，与经典 GS baseline 达到可比或更好的预测性能？

**对比：**

| 方法 | 说明 | 角色 |
|------|------|------|
| GBLUP | 基于 genomic relationship matrix | polygenic / infinitesimal baseline |
| BayesB | marker-effect mixture prior | sparse large-effect baseline |
| BayesA / BayesLasso | 其他 Bayesian shrinkage baseline | 稳健性参照 |
| Two-stage TabICL-GS | 第一层 TabICL block 表征，第二层 expert aggregation | data-driven block representation model |

**分析重点：**
- 如果 two-stage expert prediction 在某些 trait / fold 上优于 baseline，说明 data-driven block representation 提供了有预测价值的信息。
- 如果 two-stage expert prediction 在某些 trait / fold 上不如 baseline，不直接解释为“模型失败”，而是作为后续引入 genomic priors 的动机。
- 报告 Pearson / Spearman / RMSE / MAE，并按 trait 展示性能差异，避免只看总体均值。

**推荐结论措辞：**

```text
The two-stage TabICL-GS model achieved competitive genomic prediction performance compared with classical GS baselines, indicating that TabICL-derived block representations contain useful predictive information. However, its trait-dependent performance suggests that data-driven block representations alone may not fully cover the diversity of genetic architectures.
```

### 1.2 为什么引入 dual prior 与 group-shared gate

**核心问题：** 如果 two-stage TabICL-GS 已经是一个有效模型，为什么还要融合 GBLUP 和 BayesB，并且为什么需要 group-specific gate？

**动机：** dual prior 的引入不应该写成“TabICL 捕捉非线性、GBLUP 捕捉全局、BayesB 捕捉 sparse”这种机制性断言。更稳妥的说法是：TabICL、GBLUP 和 BayesB 是三类不同的预测来源，分别代表 data-driven block representation、polygenic relationship prior 和 sparse marker-effect prior。它们对不同 trait 的贡献可能不同，因此需要用 OOF fusion 检验互补性。

| 预测来源 | 代表信息 | 论文中的安全表述 |
|----------|----------|------------------|
| TabICL two-stage | 数据驱动的 block-level SNP 表征 | predictive representation source |
| GBLUP | 全基因组亲缘关系 / polygenic 背景 | polygenic relationship prior |
| BayesB | 稀疏 marker effect 假设 | sparse marker-effect prior |

**Bayes-prior-guided group-shared gate：**

这里的 sample grouping 不是由 BayesB 直接给 outer-test 样本贴标签，而是一个“Bayes 作用于 block，再间接影响 sample grouping”的过程：

```text
BayesB beta
  → 每个 block 聚合为一个 block prior score
  → 按 prior score rank 将 block 分成低 / 中 / 高 prior groups
  → 每个样本在所有 block 上得到响应强度
  → 对同一 prior group 内的 block 响应取平均，得到 sample-level group feature
  → outer-test 样本用 group feature 匹配 OOF 训练阶段学到的 group centroids
  → 根据所属 group 使用对应的 alpha_g / w_g 融合 TabICL、GBLUP、BayesB 预测
```

具体含义：

- **Block prior score：** 当前 outer fold 的 `bayesb_beta` 按 block 内 SNP 做 L2 聚合，一个 block 得到一个 prior score；分数越大，表示 BayesB 对该 block 的 marker-effect signal 越强。
- **Block prior groups：** 将所有 block 的 prior score 排序，并按 rank 均匀切成 `num_groups=3` 组，可解释为 low / medium / high Bayes prior blocks。这里分的是 block，不是 sample。
- **Block response：** 对每个样本，将该样本在每个 block 的 stage2 embedding chunk 压缩为一个 RMS summary，得到长度等于 block 数的响应向量。
- **Sample-level group feature：** 对某个样本，把属于同一 block prior group 的 block response 取平均，得到一个 3 维 group feature，例如 `[low-prior response, mid-prior response, high-prior response]`。
- **Sample grouping：** outer-test 样本先对 group feature 做标准化，再与 OOF 训练阶段学到的 group centroids 比较，默认用 nearest-centroid 归组。因此样本组别来自“它在不同 prior block groups 上的响应模式”，不是来自 BayesB prediction 本身。
- **Group-specific fusion：** 每个 sample group 有自己的 `alpha_g` 和 `w_g`。其中 `alpha_g` 控制 Bayes-family prediction 与 GBLUP 的 prior mixture，`w_g` 控制 TabICL prediction 与 prior mixture 的最终融合。

一句话总结：

```text
BayesB 在这里扮演的是定义 block-level prior structure 的角色；sample 的归组由其在这些 prior-defined block groups 上的 response pattern 决定。
```

当前版本的一个重要特点是：`alpha_g`、`w_g` 和 group centroids 可以从 fold1 的 OOF 训练中学习并冻结；但 outer-test 的 block prior score 仍来自当前 outer fold 的 `bayesb_beta`，所以 block prior 分层和 sample group feature 会随 outer fold 有轻微变化。

**渐进式 fusion / gate 实验：**

| 方法 | 目的 |
|------|------|
| Two-stage TabICL-GS | data-driven representation baseline |
| Two-stage TabICL-GS + GBLUP | 检验 polygenic prior 是否补充 TabICL 预测 |
| Two-stage TabICL-GS + BayesB | 检验 sparse marker-effect prior 是否补充 TabICL 预测 |
| Two-stage TabICL-GS + GBLUP + BayesB | 完整 dual-prior fusion baseline |
| Two-stage TabICL-GS + GBLUP + BayesB + group-shared gate | 检验 sample response pattern 是否支持组特异融合权重 |

**需要展示的证据：**
- OOF fusion / group-shared gate 后 Pearson / Spearman / RMSE / MAE 是否提升或更稳定。
- 不同 trait 或不同 sample group 的 `alpha_g` / `w_g` 是否不同，说明不同响应模式下依赖的预测来源不同。
- 不同 group 的 centroid、group count、block prior group count 是否稳定，作为 group mechanism 的诊断。
- TabICL、GBLUP、BayesB 三者预测值的相关性是否小于 1，残差是否不完全重合，作为互补性的辅助证据。

**推荐结论措辞：**

```text
The dual-prior gate does not assign samples directly using BayesB predictions. Instead, BayesB first defines a block-level prior structure, and each sample is grouped by its response pattern across these prior-defined block groups. The group-specific fusion weights provide an empirical way to combine a data-driven TabICL representation with established genomic priors.
```

### 1.3 第一层 TabICL 表征提取能力验证

**核心问题：** TabICL 的 block-wise 表征是否比原始 SNP 和传统降维更适合作为下游预测输入？

**实验设计：** 固定下游模型，只改变输入特征表示：

| 输入 | 下游模型 | 说明 |
|------|----------|------|
| 原始 SNP (10K) | Ridge / Lasso / ElasticNet / SVR | 原始特征 baseline |
| PCA on 原始 SNP | Ridge / Lasso / ElasticNet / SVR | 传统降维 |
| Block-wise TabICL 表征 | Ridge / Lasso / ElasticNet / SVR | 本文 representation |

**关键：** 四个下游模型 × 三种输入 = 12 组实验。如果 TabICL 表征在多个下游模型上都优于其他输入，结论更稳健：提升不是只来自 XGBoost 或 fusion，而是第一层表征本身具有预测信息。

**预期结论：**
- TabICL 表征 > PCA / 原始 SNP → TabICL block representation 比传统压缩方式更适合作为预测输入。
- 即使简单线性模型也能在 TabICL 表征上获得较好结果 → 表征本身有价值，不完全依赖复杂下游模型。

**注意：** 这里仍然不直接宣称 TabICL 捕捉了非线性、epistasis 或交互效应。更安全的表述是：TabICL representation captures predictive variation not fully retained by raw-SNP baselines or PCA compression。

### 1.4 Ablation Study

| 实验 | 目的 |
|------|------|
| 不同 block size | block 划分粒度的影响 |
| 不同 PCA 维度 | 表征压缩程度的影响 |
| 第一层用 prediction vs 用中间表征 | 验证中间表征是否比第一层直接预测更适合作为第二层输入 |
| BayesB 换成 BayesA / BayesC / BayesLasso | 验证 fusion 框架对 Bayesian prior 选择的敏感性 |

### 1.5 可扩展性分析

- 10K → 100K → 1000K SNP 下各方法的运行时间
- 大规模场景下可考虑预筛选或用 VI 替代 BayesB（留作未来工作）

---

## Part 2: 可解释性分析——模型关注了什么

### 2.1 Block Importance（粗粒度）

**方法：** 结合 expert importance 与 gate diagnostics
- 如果第二层 expert 使用 XGBoost，可用 split-based / gain-based feature importance 给出每个 block 表征的重要性。
- 对 group-shared gate，可展示 BayesB-guided block prior groups、block prior group counts、sample group centroids、`alpha_g` / `w_g` 等诊断量。
- 可辅以 permutation importance 做交叉验证，避免只依赖 XGBoost split importance。

**展示：**
- Manhattan-style plot：x 轴为基因组位置，y 轴为 block importance score
- 标注 top-K 重要 block 对应的基因组区域
- 展示 low / medium / high Bayes prior block groups 与重要 block 的重叠关系

### 2.2 SNP Importance（细粒度）

对 top-K 重要 block，深入到 block 内部：

**两个来源交叉验证：**

1. **BayesB 后验 π_j**（inclusion probability）
   - BayesB 跑完直接获得
   - 反映线性框架下的 SNP 重要性

2. **第一层 TabICL permutation importance**
   - 在 block 内部 shuffle 单个 SNP，观察 block 输出表征的变化
   - 反映 TabICL block representation 对该 SNP 扰动的敏感性

**分析：**
- 两者一致 → SNP 重要性可信
- 两者不一致 → 说明 BayesB 的 sparse marker-effect prior 与 TabICL representation sensitivity 提供了不同的重要性视角，值得进一步分析

**展示：**
- 对重要 block 做 zoom-in 的 SNP importance plot
- BayesB π_j vs TabICL permutation importance 的散点图

### 2.3 Top-K SNP 预测能力验证

**核心实验：**

| 实验 | 输入 | 模型 |
|------|------|------|
| A | GWAS top-K SNP | GBLUP / Ridge |
| B | 本框架筛选的 top-K SNP | GBLUP / Ridge |
| C | 全部 SNP（baseline） | GBLUP / Ridge |

**展示：**
- K 从小到大变化（100, 500, 1000, 2000, 5000）
- 画 GWAS top-K vs 本框架 top-K 的预测精度曲线
- 预期：小 K 时本框架优势更明显（GWAS 在小样本下 top hits 噪声大）

**意义：**
- 如果 B > A → 本框架的 importance 比 GWAS 单标记检验更能找到对预测有用的 SNP
- 如果 B ≈ C → 本框架能用少量 SNP 达到全基因组预测精度，有降维价值

### 2.4 已知 QTL 数据库 Overlap 验证

- 将 top SNP 与 Animal QTLdb / GWAS Catalog 做 overlap 分析
- 验证模型找到的重要区域是否有生物学意义
- 按性状分别分析（不同性状的遗传架构不同）

---

## 论文结构草案

```
1. Introduction
   - GS 背景，传统方法的局限
   - Tabular FM 的机会
   - 本文贡献（2-3 点）

2. Methods
   2.1 Block-wise 两层架构（第一层 TabICL 表征提取 + 第二层 expert aggregation）
   2.2 Dual-prior group-shared gate（BayesB-guided block grouping + sample-level group feature）
   2.3 OOF-calibrated fusion 策略（two-stage expert prediction + GBLUP + BayesB）
   2.4 多分辨率可解释性框架（Block → SNP）

3. Results
   3.1 Two-stage TabICL-GS vs baseline 预测精度比较
   3.2 Dual-prior group-shared gate 分析（TabICL + GBLUP + BayesB）
   3.3 第一层 TabICL 表征提取能力验证（多个简单下游模型 × 三种输入）
   3.4 Ablation study
   3.5 Block importance 分析
   3.6 SNP importance 与 GWAS 对比
   3.7 Top-K SNP 预测能力验证
   3.8 QTL overlap 验证

4. Discussion
   - 为什么 data-driven block representation 与 genomic priors 互补
   - 可解释性发现的生物学意义
   - 局限性与未来方向（大规模 SNP 下的加速策略等）

5. Conclusion
```

---

## 核心卖点总结

1. **架构创新：** Block-wise 两层架构（TabICL 表征提取 + expert aggregation），将 Tabular FM 作为 data-driven SNP block representation 模块引入 GS，并验证其表征提取能力优于 raw SNP / PCA 输入
2. **融合贡献：** Dual-prior group-shared gate 先用 BayesB beta 定义 block-level prior structure，再根据样本在不同 prior block groups 上的响应模式进行组特异融合，避免把 BayesB 直接作为 sample label，同时检验 TabICL、GBLUP、BayesB 的互补性
3. **方法贡献：** 多分辨率可解释性框架（Block → SNP），结合 expert importance 与 gate diagnostics 定位关键 block，再深入到 SNP 级别，并通过 top-K 预测实验验证 importance 的实际价值
