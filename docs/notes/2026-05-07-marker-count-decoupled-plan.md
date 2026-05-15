# 2026-05-07 marker-count decoupled 正式入口说明

## 当前决定

- 标记量实验采用独立新入口，不继续堆叠在旧样本量脚本上
- 当前 marker count 口径：
  - `2000`
  - `10000`
  - `50000`
- 其中：
  - `10000` 直接复用 `5.4-duli-liudang` 主结果
  - `2000 / 50000` 重新正式运行

## 为什么单独开新入口

主要原因有两个：

- `marker count` 变化后，`best_block` 不应继续复用 `10K` 主线结果，必须单独重搜
- 现有 `prepare_*_plink_cache.py` 会改写共享的 `*_cache_summary.json`
  - 如果直接拿来跑 `2K / 50K`
  - 会污染当前正式 `10K` 主线

因此本轮新增入口遵循：

- 不改旧主线
- 不覆盖共享 summary 文件
- 自己为不同 marker count 构造对应的子集 PLINK
- 每个 marker count 单独搜索 `best_block`

## 新增正式入口

- 单 trait 入口：
  - `scripts/run_54_marker_count_decoupled_trait.py`
- 总控入口：
  - `scripts/run_54_marker_count_decoupled_full_server_gpu1.sh`
- 汇总入口：
  - `scripts/summarize_54_marker_count_decoupled.py`

## 正式输出目录

- 总目录：
  - `outputs/5.4-marker_count-decoupled`

目录结构示意：

- `outputs/5.4-marker_count-decoupled/{dataset}/{trait_slug}/maxsnps_02000`
- `outputs/5.4-marker_count-decoupled/{dataset}/{trait_slug}/maxsnps_50000`

每个 marker count 目录下包含：

- `fold1_tabicl_block_search`
- `no_prior`
- `baseline_3models`
- `compare`
- `run_manifest.json`

## 当前正式运行口径

- 底层只跑：
  - `no_prior-TabICL`
  - `BayesB`
  - `GBLUP`
  - `RKHS`
- 融合全部从留档直接构建：
  - `single_bayesb_two_step_ls`
  - `single_gblup_two_step_ls`
  - `single_rkhs_two_step_ls`
  - `triple_two_step_ls`

## block 搜索口径

- 对 `2K / 50K`，每个 trait 都单独搜索 `best_block`
- 当前采用 `no_prior TabICL` 的 fold1 inner-OOF 搜索
- 当前默认：
  - `n_trials = 10`
  - `inner_folds = 3`

## 当前 block 搜索边界

当前固定为：

- `2K`
  - `50 ~ 400`
- `50K`
  - `1000 ~ 5000`

当前默认：

- `optuna trials = 10`
- `inner_folds = 3`

说明：

- 该范围已经固定到 marker-count 独立入口中
- 如有需要，仍可通过命令行手动覆盖 `--block-min / --block-max`

## 当前正式任务 trait 范围

与样本量实验保持一致，共 8 个 trait：

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

## 当前 GPU1 启动方式

正式后台命令：

```bash
cd /home/server/code/git/TabICLv2-test
nohup bash scripts/run_54_marker_count_decoupled_full_server_gpu1.sh \
  >> outputs/5.4-marker_count-decoupled/pipeline.log 2>&1 < /dev/null &
```

## 当前汇总输出

总控结束后会生成：

- `outputs/5.4-marker_count-decoupled/marker_count_main_results.csv`
- `outputs/5.4-marker_count-decoupled/marker_count_main_results.json`

说明：

- 该汇总会自动把 `10000` 主结果从
  - `outputs/5.4-duli-liudang/main_results_non_pig_fixed.csv`
  补进来
- 因而最终可直接用于比较：
  - `2K / 10K / 50K`

## 当前状态

- 2026-05-07 已在 `GPU1` 上正式启动
- 起跑 trait 为：
  - `cotton1245 / cotton_fiblen_17_18_cotton_fiblen_17_18`
  - `cotton1245 / cotton_fibelo_17_18_cotton_fibelo_17_18`
