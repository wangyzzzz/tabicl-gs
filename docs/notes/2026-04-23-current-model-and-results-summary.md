# 当前模型与结果汇总（截至 2026-04-23）

日期: 2026-04-23

## 1. 当前固定模型简介

当前正式主线模型为:

- `Two-stage TabICLv2 + dual priors (BayesB + GBLUP)`

其目标不是替代统计遗传学基线，而是:

- 让 TabICLv2 学习局部 block 上下文表示
- 再与 `BayesB/GBLUP` 先验进行全局融合

最终输出:

- `y_dual`

## 2. 架构设计（固定版）

### 2.1 数据入口与分块

- 输入统一来自 PLINK (`.bed/.bim/.fam`)
- 若 SNP 数 > 10000，固定随机种子下采样到 10000
- 分块策略固定为 `window`
- `group_size` 在 fold1 的 inner-OOF 中搜索，fold2-5 复用

### 2.2 第一层（Block-wise TabICLv2）

- 每个 block 调用真实 TabICLv2
- 提取 `row_interactor` 向量表示
- 每个 block 单独 PCA 到 `99% explained variance`
- 不拼接 block scalar (`include_block_scalar=false`)

### 2.3 第二层（Genome-level TabICLv2）

- 拼接所有 block 的 PCA 向量形成样本特征
- 第二层使用 TabICLv2 输出 `y_tabicl`

### 2.4 双先验融合（Dual-prior Gate）

先验分支:

- `y_bayesb`（BGLR）
- `y_gblup`（sommer）

融合公式:

- `y_prior = alpha * y_bayesb + (1 - alpha) * y_gblup`
- `y_dual = w * y_tabicl + (1 - w) * y_prior`

当前固定设置:

- `group = 1`（全局 gate）

### 2.5 权重来源（当前对齐口径）

`prior-only` 评估中，`alpha` 来源固定为:

- `fold1_inner_val_oof_prior`

即:

- 由 fold1 inner-val 的 OOF prior 重新计算 alpha，不使用落盘 alpha_group 直读。

## 3. 当前 compare 范围

统计范围为:

- `dual-prior` 已完成的 `20` 个 traits
- 组成: `cotton1245 (4) + pig3534 (3) + soybean951 (13)`

对比模型:

- `dual_prior`
- `prior_only` (`BayesB+GBLUP`, alpha from fold1 inner-val OOF prior)
- `no_prior` (TabICLv2-TabICLv2 without prior branch)
- `GBLUP / BayesA / BayesB / BayesLasso`

## 4. 当前总体结果（Pearson，trait 均值）

- `dual_prior`: `0.6075`
- `prior_only`: `0.6006`
- `no_prior`: `0.5644`
- `GBLUP`: `0.5965`
- `BayesA`: `0.6001`
- `BayesB`: `0.6013`
- `BayesLasso`: `0.5975`
- `best_baseline` (每 trait 取四个 baseline 最优后再平均): `0.6024`

增益统计:

- `dual_prior - prior_only`: `+0.0069` (18/20 traits 为正)
- `dual_prior - no_prior`: `+0.0431` (20/20 traits 为正)
- `dual_prior - best_baseline`: `+0.0051` (16/20 traits 为正)
- `prior_only - best_baseline`: `-0.0018` (仅 1/20 traits 为正)

## 5. 结果解读（当前阶段）

从当前 20 traits 的结果可得到以下稳定结论:

1. `no_prior` 显著弱于其余方案  
说明仅靠两层 TabICLv2 表示，整体不足以稳定超过传统先验。

2. `prior_only` 与强 baseline 接近  
说明 BayesB + GBLUP 融合本身已经较强，但并不稳定超过最优 baseline。

3. `dual_prior` 稳定优于 `prior_only` 与 `no_prior`  
说明 TabICLv2 分支在双先验基础上提供了额外可用信息，而非纯重复先验。

因此，当前最稳健的叙事是:

- `TabICLv2` 在 `BayesB/GBLUP` 先验之外提供了可叠加的增量信号。

## 6. 逐 trait 横向对比表（20 traits）

```text
dataset,trait,dual_prior,no_prior,prior_only,GBLUP,BayesA,BayesB,BayesLasso,best_baseline,best_baseline_model
cotton1245,cotton_fibelo_17_18_cotton_fibelo_17_18,0.6640,0.6324,0.6560,0.6461,0.6558,0.6595,0.6497,0.6595,BayesB
cotton1245,cotton_fiblen_17_18_cotton_fiblen_17_18,0.7636,0.7407,0.7548,0.7420,0.7563,0.7559,0.7496,0.7563,BayesA
cotton1245,cotton_fibmic_17_18_cotton_fibmic_17_18,0.6670,0.6499,0.6539,0.6514,0.6551,0.6544,0.6525,0.6551,BayesA
cotton1245,cotton_fibstr_17_18_cotton_fibstr_17_18,0.7686,0.7496,0.7582,0.7566,0.7576,0.7562,0.7573,0.7576,BayesA
pig3534,t1,0.0960,0.0791,0.0566,0.0614,0.0615,0.0520,0.0626,0.0626,BayesLasso
pig3534,t2,0.4878,0.4239,0.4850,0.4854,0.4853,0.4841,0.4850,0.4854,GBLUP
pig3534,t3,0.3216,0.2549,0.3273,0.3226,0.3262,0.3295,0.3239,0.3295,BayesB
soybean951,bbd_beijing_2013_bbd_beijing_2013,0.8191,0.7907,0.8187,0.8110,0.8176,0.8210,0.8117,0.8210,BayesB
soybean951,fa16c_beijing_2013_fa16c_beijing_2013,0.6044,0.5406,0.5989,0.5956,0.5986,0.6011,0.5965,0.6011,BayesB
soybean951,fa18c_beijing_2013_fa18c_beijing_2013,0.8008,0.7693,0.7987,0.7981,0.7991,0.7987,0.7982,0.7991,BayesA
soybean951,fac_beijing_2013_fac_beijing_2013,0.6413,0.6024,0.6378,0.6389,0.6378,0.6362,0.6363,0.6389,GBLUP
soybean951,hsw_beijing_2013_hsw_beijing_2013,0.7771,0.7405,0.7713,0.7711,0.7717,0.7704,0.7723,0.7723,BayesLasso
soybean951,ll_beijing_2013_ll_beijing_2013,0.4115,0.3508,0.4027,0.4016,0.4024,0.4032,0.3998,0.4032,BayesB
soybean951,lw_beijing_2013_lw_beijing_2013,0.5554,0.5017,0.5430,0.5259,0.5347,0.5471,0.5274,0.5471,BayesB
soybean951,md_beijing_2013_md_beijing_2013,0.6724,0.6152,0.6662,0.6598,0.6667,0.6683,0.6589,0.6683,BayesB
soybean951,plh_beijing_2013_plh_beijing_2013,0.7584,0.7033,0.7577,0.7425,0.7551,0.7617,0.7454,0.7617,BayesB
soybean951,prt_beijing_2013_prt_beijing_2013,0.2812,0.2025,0.2846,0.2817,0.2798,0.2857,0.2784,0.2857,BayesB
soybean951,sl_beijing_2013_sl_beijing_2013,0.5553,0.5048,0.5424,0.5405,0.5415,0.5436,0.5440,0.5440,BayesLasso
soybean951,st_beijing_2013_st_beijing_2013,0.7345,0.6845,0.7325,0.7322,0.7332,0.7321,0.7337,0.7337,BayesLasso
soybean951,vbn_beijing_2013_vbn_beijing_2013,0.7701,0.7517,0.7658,0.7663,0.7664,0.7651,0.7662,0.7664,BayesA
```

## 7. 当前版本定位

当前版本可以定义为:

- `production-grade predictive pipeline` 已基本稳定
- `paper second part` 需要从“精度比较”提升到“遗传信号解析”

建议后续分析主线:

- 使用 `Residual Genetic Signal Framework` 解释 dual-prior 增益来源
- 重点比较 `shared vs orthogonal` (vs GWAS) 的残差信号结构

