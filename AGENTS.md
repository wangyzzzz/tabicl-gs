# AGENTS

## 项目当前主线

- 当前正式主线是 `5.4-duli-liudang`
- 当前推荐主结果表：
  - `outputs/5.4-duli-liudang/main_results_non_pig_fixed.csv`
  - `outputs/5.4-duli-liudang/main_results_non_pig_fixed.json`
- 当前主结果默认 `exclude pig3534`

## 服务器环境

- 正式服务器：`server@GPU1`
- 服务器代码目录：`/home/server/code/git/TabICLv2-test`
- 服务器 conda 环境：`/data/yes/envs/TabICLv2-GS`
- 服务器 Python：`/data/yes/envs/TabICLv2-GS/bin/python`
- 正式结果以 `server@GPU1` 上的目录为准，不要默认本地结果是最新

## 本地环境

- 本地项目目录：`/Users/wangyuze/Desktop/NWAFU/tabicl-gs`
- 旧交接工作区：`/Users/wangyuze/Desktop/NWAFU/TabICLv2-test-handover-2026-04-29`
- 本地 shell：`zsh`
- 本地当前主要作用：
  - 阅读代码和文档
  - 编辑脚本
  - 通过 `ssh GPU1` 驱动正式任务
  - 拉取并汇总服务器正式结果

## 当前解偶复用逻辑

- 当前采用“解偶复用”产线，不再重复训练 single / dual / triple 的底层模型
- 底层正式留档只跑：
  - `no_prior-TabICL`
  - `BayesB`
  - `GBLUP`
  - `RKHS`
- 后续融合全部从留档结果直接构建：
  - `single-prior-TabICL`
  - `dual-prior-TabICL`
  - `triple-prior-TabICL`
  - `only-prior`

## 当前解偶复用数据目录

- 总目录：`outputs/5.4-duli-liudang`
- 底层留档：
  - `outputs/5.4-duli-liudang/no_prior`
  - `outputs/5.4-duli-liudang/baseline_3models`
- 融合结果：
  - `outputs/5.4-duli-liudang/fusion`
- 完整 compare 总表：
  - `outputs/5.4-duli-liudang/compare_all_41_traits.csv`
  - `outputs/5.4-duli-liudang/compare_all_41_traits.json`

## 当前 compare 实验目录

- sample-compare：
  - 总目录：`outputs/5.4-sample_size-decoupled`
  - 汇总表：`outputs/5.4-sample_size-decoupled/sample_size_main_results.csv`
  - 汇总 JSON：`outputs/5.4-sample_size-decoupled/sample_size_main_results.json`
- snp-compare：
  - 总目录：`outputs/5.4-marker_count-decoupled`
  - 汇总表：`outputs/5.4-marker_count-decoupled/marker_count_main_results.csv`
  - 汇总 JSON：`outputs/5.4-marker_count-decoupled/marker_count_main_results.json`

## TabPFN 补充验证线

- 这条线是补充验证，不替代当前主线 `5.4-duli-liudang`
- 正式服务器结果目录：
  - `outputs/5.4-tabpfn-10k-8traits`
  - `outputs/5.4-tabpfn-10k-8traits-fusion`
- 当前汇总 traits：
  - `10K SNP + 8 个 SNP-count traits`
- 本地固化结果表：
  - `outputs/5.4-tabpfn-10k-8traits-fusion/tabpfn_10k_8traits_compare.csv`
  - `outputs/5.4-tabpfn-10k-8traits-fusion/tabpfn_10k_8traits_compare.json`
- 本地结果说明：
  - `docs/notes/2026-05-09-tabpfn-10k-8traits-summary.md`
- 当前口径：
  - `no_prior-TabPFN`
  - `only-triple-fusion`
  - `triple-fusion`
  - baseline 仍固定复用 `BayesB / GBLUP / RKHS`

## 当前正式入口

- no-prior 留档：
  - `scripts/run_54_duli_liudang_noprior_server_gpu1.sh`
- baseline 留档：
  - `scripts/run_54_duli_liudang_baseline_server_gpu1.sh`
- fusion：
  - `scripts/run_54_duli_liudang_fusion_server_gpu1.sh`
- 总控：
  - `scripts/run_54_duli_liudang_full_server_gpu1.sh`

## 解偶复用融合入口

- 核心融合入口：
  - `scripts/run_decoupled_prior_fusion_from_archives.py`
- 该入口基于已留档的：
  - `no_prior` outer test + inner OOF
  - `baseline` outer test + inner OOF
- 不应再为 single / dual / triple 重复训练底层模型

## 当前结果口径

- 主结果默认排除：`pig3534`
- 正式 baseline 口径固定为：
  - `BayesB`
  - `GBLUP`
  - `RKHS`
- 不要再混入旧 baseline 口径
- 当前主推 fusion 方法为：
  - `single_bayesb_two_step_ls`
  - `single_gblup_two_step_ls`
  - `single_rkhs_two_step_ls`
  - `triple_two_step_ls`

## 任务原则

- 正式实验继续使用 `server@GPU1`
- 正式 compare 以服务器产出的新 baseline 为准
- `no_prior-TabICL` 使用既定 `best_block`
- 当前默认把 `pig3534` 从主结果中排除，若分析 pig，需单独说明
