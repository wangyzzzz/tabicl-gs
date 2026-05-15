# TabICLv2-GS 任务交接文档

更新时间：2026-04-29 15:58:23 CST

## 1. 任务目标

当前主任务是继续运行并汇总下面这条正式实验线：

- 数据范围：
  - `rice529`
  - `Cotton1245`
  - `Soybean951`
  - `pig3534`
  - `wheat406`
- 模型主线：
  - `triple-prior-TabICL`
  - `only-triple-prior`
  - `no-prior-TabICL`
  - baseline：
    - `BayesA`
    - `BayesB`
    - `BayesLasso`
    - `RKHS`
    - `GBLUP`

这里的 `triple-prior` 指三路 prior：

- `BayesB`
- `GBLUP`
- `RKHS`

并通过 `group_shared_gate` 与 `TabICL` 组合。

## 2. 当前代码状态

本地目录已经包含这轮任务需要的代码修改，重点是：

- `only-triple-prior` 的兼容修复已经完成
  - 支持优先从 `prior_cache/*.npy` 读取
  - 若历史目录没有 `.npy`，会 fallback 到：
    - `bayesb_outer/eval_fit/predictions.csv`
    - `gblup_outer/_residual_target/GBLUP/test_fit/predictions.csv`
    - `rkhs_outer/eval_fit/predictions.csv`

- baseline 正式链路已经把 `RKHS` 纳入
  - 关键修改：
    - `configs/tabicl_block/window_baseline_only_10traits.yaml`
    - `src/tabicl_gs/pipeline/experiment.py`
    - `scripts/run_rice529_10traits_baseline_only_server_gpu1.sh`
    - `scripts/run_multidataset_alltraits_baseline_only_server_gpu1.sh`

- 已新增统一汇总脚本：
  - `scripts/summarize_tripleprior_full_compare.py`

## 3. 当前正式运行环境

正式运行不在本地，而在：

- 服务器：`server@GPU1`
- 代码目录：
  - `/home/server/code/git/TabICLv2-test`
- Python 环境：
  - `/data/yes/envs/TabICLv2-GS/bin/python`
- R 环境：
  - `configs/tabicl_block/window_baseline_only_10traits.yaml`
  - `rscript_path: /data/yes/envs/r_env/bin/Rscript`

## 4. 当前服务器正在执行的任务

截至本文档更新时间，服务器上正在继续跑两条主线：

### 4.1 triple-prior 主线

脚本：

- `scripts/run_multidataset_alltraits_tripleprior_server_gpu1.sh`

说明：

- 复用之前 dual 搜好的 `best_block.json`
- 不重新搜 block
- `triple-prior`、`only-triple-prior`、`no-prior` 都从这个主目录逐 trait 继续推进

输出根目录：

- `outputs/all_datasets_alltraits_tripleprior`

### 4.2 baseline-only 主线

脚本：

- `scripts/run_rice529_10traits_baseline_only_server_gpu1.sh`
- `scripts/run_multidataset_alltraits_baseline_only_server_gpu1.sh`

说明：

- 这是“新口径 baseline”，已经包含 `RKHS`
- 旧 baseline 结果仍存在，但不能直接与新口径混用

输出根目录：

- `outputs/rice529_10traits_baseline_only`
- `outputs/multidataset_alltraits_baseline_only`

## 5. 当前正式结果进度

这是截至 `2026-04-29 15:58:23 CST` 的最新状态。

### 5.1 triple-prior 系列完成数

- `triple-prior-TabICL = 8`
- `only-triple-prior = 8`
- `no-prior-TabICL = 9`

已完成的 `triple-prior / only-triple-prior` trait：

- `cotton1245/cotton_fiblen_17_18_cotton_fiblen_17_18`
- `cotton1245/cotton_fibmic_17_18_cotton_fibmic_17_18`
- `rice529/grain_weight`
- `rice529/heading_date`
- `rice529/num_effective_panicles`
- `rice529/num_panicles`
- `rice529/plant_height`
- `rice529/yield`

已完成的 `no-prior-TabICL` trait：

- `cotton1245/cotton_fibelo_17_18_cotton_fibelo_17_18`
- `cotton1245/cotton_fiblen_17_18_cotton_fiblen_17_18`
- `cotton1245/cotton_fibmic_17_18_cotton_fibmic_17_18`
- `rice529/grain_weight`
- `rice529/heading_date`
- `rice529/num_effective_panicles`
- `rice529/num_panicles`
- `rice529/plant_height`
- `rice529/yield`

### 5.2 baseline（含 RKHS）完成数

- `baseline_with_rkhs = 38`

已完成的新口径 baseline trait：

- `cotton1245/cotton_fibelo_17_18_cotton_fibelo_17_18`
- `cotton1245/cotton_fiblen_17_18_cotton_fiblen_17_18`
- `cotton1245/cotton_fibmic_17_18_cotton_fibmic_17_18`
- `cotton1245/cotton_fibstr_17_18_cotton_fibstr_17_18`
- `pig3534/t1`
- `pig3534/t2`
- `rice529/grain_length`
- `rice529/grain_thickness`
- `rice529/grain_weight`
- `rice529/grain_width`
- `rice529/heading_date`
- `rice529/num_effective_panicles`
- `rice529/num_panicles`
- `rice529/plant_height`
- `rice529/spikelet_length`
- `rice529/yield`
- `soybean951/bbd_beijing_2013_bbd_beijing_2013`
- `soybean951/fa16c_beijing_2013_fa16c_beijing_2013`
- `soybean951/fa18c_beijing_2013_fa18c_beijing_2013`
- `soybean951/fac_beijing_2013_fac_beijing_2013`
- `soybean951/hsw_beijing_2013_hsw_beijing_2013`
- `soybean951/ll_beijing_2013_ll_beijing_2013`
- `soybean951/lw_beijing_2013_lw_beijing_2013`
- `soybean951/md_beijing_2013_md_beijing_2013`
- `soybean951/plh_beijing_2013_plh_beijing_2013`
- `soybean951/prt_beijing_2013_prt_beijing_2013`
- `soybean951/sl_beijing_2013_sl_beijing_2013`
- `soybean951/st_beijing_2013_st_beijing_2013`
- `soybean951/vbn_beijing_2013_vbn_beijing_2013`
- `wheat406/ph_e1`
- `wheat406/ph_e2`
- `wheat406/ph_e3`
- `wheat406/pl_e1`
- `wheat406/pl_e2`
- `wheat406/pl_e3`
- `wheat406/sl_e1`
- `wheat406/sl_e2`
- `wheat406/sl_e3`

## 6. 当前正在跑到哪

根据最新服务器日志：

### 6.1 triple-prior 主线

- `rice529/grain_weight` 已完成
- 当前在跑：
  - `rice529/spikelet_length`
- `cotton1245/cotton_fiblen` 已完成
- 当前在跑：
  - `cotton1245/cotton_fibelo`

### 6.2 baseline 主线

- `rice529 baseline` 已全部结束
- `multidataset baseline` 还在继续：
  - `pig3534/t3` 正在跑
  - wheat406 全部已完成

## 7. 关键脚本说明

### 7.1 triple-prior 主线

- `scripts/run_multidataset_alltraits_tripleprior_server_gpu1.sh`

用途：

- 正式逐 trait 跑：
  - `no-prior-TabICL`
  - `triple-prior-TabICL`
  - `only-triple-prior`

### 7.2 only-triple-prior 计算

- `scripts/eval_triple_prior_only_from_outputs.py`

用途：

- 从已完成的 `tabicl_tabicl_triple_prior` 目录中：
  - 用 `fold1 inner OOF` 学 prior 权重
  - 对 `fold1-5 outer-test` 形成 `only-triple-prior`

### 7.3 baseline 新口径主线

- `scripts/run_rice529_10traits_baseline_only_server_gpu1.sh`
- `scripts/run_multidataset_alltraits_baseline_only_server_gpu1.sh`

用途：

- 正式跑 `BayesA/BayesB/BayesLasso/RKHS/GBLUP`

### 7.4 横向汇总

- `scripts/summarize_tripleprior_full_compare.py`

用途：

- 汇总：
  - `triple-prior-TabICL`
  - `only-triple-prior`
  - `no-prior-TabICL`
  - `BayesA/BayesB/BayesLasso/RKHS/GBLUP`

## 8. 已知注意事项

### 8.1 不要混用旧 baseline 和新 baseline

当前存在两种 baseline：

- 旧 baseline：不含 `RKHS`
- 新 baseline：含 `RKHS`

正式 compare 只能使用“新口径 baseline”。

### 8.2 only-triple-prior 以前报错过，但现在脚本已修复

之前报错原因包括：

- 旧版脚本错误依赖 baseline 目录下的 `RKHS predictions.csv`
- 历史 triple 目录没有某些 `.npy` 文件

现在的本地脚本已经修复为：

- 优先读 `prior_cache/*.npy`
- 若缺失则 fallback 到历史 `predictions.csv`

### 8.3 triple 速度显著慢于 baseline

这是当前正常现象，不代表卡死。

## 9. 另一台机器继续接手时建议做法

解压后，建议按下面顺序接手：

### 9.1 先看本文档

路径：

- `docs/notes/2026-04-29-handover-to-new-machine.md`

### 9.2 如果继续依赖 server@GPU1 上已有任务

先登录服务器：

```bash
ssh server@GPU1
cd /home/server/code/git/TabICLv2-test
```

查看当前进度：

```bash
tail -n 40 outputs/all_datasets_alltraits_tripleprior/pipeline.log
tail -n 40 outputs/multidataset_alltraits_baseline_only/pipeline_rerun.log
```

### 9.3 如果要重新汇总当前已完成结果

用：

```bash
PYTHONPATH=src /data/yes/envs/TabICLv2-GS/bin/python scripts/summarize_tripleprior_full_compare.py \
  --triple-root outputs/all_datasets_alltraits_tripleprior \
  --rice-baseline-root outputs/rice529_10traits_baseline_only \
  --multi-baseline-root outputs/multidataset_alltraits_baseline_only \
  --output-dir outputs/tripleprior_compare_summary
```

### 9.4 如果要继续只补 only-triple-prior

直接对已完成 trait 运行：

```bash
PYTHONPATH=src /data/yes/envs/TabICLv2-GS/bin/python scripts/eval_triple_prior_only_from_outputs.py \
  --dataset rice529 \
  --trait-col Heading_date \
  --triple-root outputs/all_datasets_alltraits_tripleprior/rice529/heading_date
```

### 9.5 如果要重启正式 triple 主线

```bash
cd /home/server/code/git/TabICLv2-test
nohup bash scripts/run_multidataset_alltraits_tripleprior_server_gpu1.sh \
  > outputs/all_datasets_alltraits_tripleprior/pipeline.log 2>&1 &
```

### 9.6 如果要重启 baseline 主线

```bash
cd /home/server/code/git/TabICLv2-test
nohup bash scripts/run_rice529_10traits_baseline_only_server_gpu1.sh \
  > outputs/rice529_10traits_baseline_only/pipeline_rerun.log 2>&1 &

nohup bash scripts/run_multidataset_alltraits_baseline_only_server_gpu1.sh \
  > outputs/multidataset_alltraits_baseline_only/pipeline_rerun.log 2>&1 &
```

## 10. 本次交接最重要的一句话

另一台机器接手时，最重要的是：

- 继续使用 `server@GPU1` 上已经在跑的正式结果目录
- 正式 compare 只认“含 RKHS 的新 baseline”
- `triple-prior` 的 `best_block` 直接复用之前 dual 搜出来的 `best_block.json`
- `only-triple-prior` 已修复，可以继续从现有 triple 输出目录补算

