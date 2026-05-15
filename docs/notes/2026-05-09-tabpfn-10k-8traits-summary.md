# TabPFN 10K SNP 8 Traits 补充验证总结

## 结果来源

- 正式服务器：`server@GPU1`
- 服务器代码目录：`/home/server/code/git/TabICLv2-test`
- no-prior 结果目录：`outputs/5.4-tabpfn-10k-8traits`
- fusion 结果目录：`outputs/5.4-tabpfn-10k-8traits-fusion`
- baseline 复用目录：`outputs/5.4-duli-liudang/baseline_3models`

本地固化总表：

- `outputs/5.4-tabpfn-10k-8traits-fusion/tabpfn_10k_8traits_compare.csv`
- `outputs/5.4-tabpfn-10k-8traits-fusion/tabpfn_10k_8traits_compare.json`

说明：

- 这是一条补充验证线，不替代当前主线 `5.4-duli-liudang`
- 当前口径为 `10K SNP + 8 个 SNP-count traits + no-prior-TabPFN`
- 融合方式复用现有解偶逻辑，比较对象为 `BayesB / GBLUP / RKHS / only-triple-fusion / triple-fusion`

## 已完成 trait

- `cotton1245 / cotton_fibelo_17_18_cotton_fibelo_17_18`
- `cotton1245 / cotton_fiblen_17_18_cotton_fiblen_17_18`
- `rice529 / grain_weight`
- `rice529 / grain_width`
- `soybean951 / bbd_beijing_2013_bbd_beijing_2013`
- `soybean951 / lw_beijing_2013_lw_beijing_2013`
- `wheat406 / sl_e1`
- `wheat406 / sl_e2`

## 核心结论

`TabPFN` 单独使用时，8 个 trait 的平均 Pearson 为 `0.5986`，低于 `BayesB (0.6314)`、`GBLUP (0.6208)` 和 `RKHS (0.6271)`。这说明在当前 GS 设定下，`TabPFN` 的 no-prior 版本仍然不能稳定胜过强统计基线。

在融合后，`only-triple-fusion` 的平均 Pearson 提升到 `0.6325`，`triple-fusion` 进一步提升到 `0.6352`。从均值看，`triple-fusion` 相对 `only-triple-fusion` 仍有 `+0.3496%` 的增益，说明把 `TabPFN` 放进多 prior 融合框架后并非没有贡献。

但如果与每个 trait 的最优 baseline 相比，`triple-fusion` 的 8-trait 平均相对提升为 `-0.1230%`。也就是说，这条 `TabPFN` 补充线目前还没有像主线 `TabICL` 那样，形成更稳定的整体领先优势。

## 相对 baseline 的平均提升

`only-triple-fusion` 相对各 baseline 的平均提升：

- 相对 `BayesB`：`+0.2915%`
- 相对 `GBLUP`：`+2.0425%`
- 相对 `RKHS`：`+0.6709%`

`triple-fusion` 相对各 baseline 的平均提升：

- 相对 `BayesB`：`+0.6315%`
- 相对 `GBLUP`：`+2.4058%`
- 相对 `RKHS`：`+1.0401%`

整体看，`TabPFN` 融合后对 `GBLUP` 的补益最明显；相对 `BayesB` 和 `RKHS` 也有平均正增益，但稳定性明显弱于当前 `TabICL` 主线。

## trait 级别表现

`triple-fusion` 相对 trait 内最佳 baseline 为正的 trait 有 4 个：

- `cotton_fibelo`
- `cotton_fiblen`
- `grain_weight`
- `lw`

相对 trait 内最佳 baseline 为负的 trait 也有 4 个：

- `grain_width`
- `bbd`
- `sl_e1`
- `sl_e2`

其中 `lw` 的增益最明显，`triple-fusion` 相对最佳 baseline `BayesB` 提升 `+4.0014%`。而 `wheat406` 的两个 trait 明显拖累整体稳定性：这两个 trait 上最强 baseline 都是 `RKHS`，而 `triple-fusion` 分别落后 `-3.2763%` 和 `-3.6572%`。

## 当前判断

这条 `TabPFN` 验证线支持一个比较清楚的判断：把 foundation model 放进多 prior 融合框架本身是可行的，`TabPFN` 也能在若干 trait 上给出正向贡献；但在当前 10K SNP、8 个 trait 的设定下，它的整体稳定性和平均表现仍未达到 `TabICL` 主线的水平。

因此，这条结果更适合作为“融合框架具备可扩展性、但不同 foundation model 的适配程度并不相同”的补充证据，而不适合作为替换主线叙事的核心结果。
