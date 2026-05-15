# 给 boss 的提示词：在 Results 中新增 TabPFN 补充验证小节

请基于下面这些现有正文草稿与结果文件，在论文 `Results` 主体中新增一节或一个小节，用于纳入 `TabPFN` 的补充验证结果。

## 你要修改的正文草稿

- 主文件：
  - `docs/results_1_4_cn_draft.md`

## 你需要阅读的 TabPFN 补充结果文件

- 结果说明：
  - `docs/notes/2026-05-09-tabpfn-10k-8traits-summary.md`
- 总表 CSV：
  - `outputs/5.4-tabpfn-10k-8traits-fusion/tabpfn_10k_8traits_compare.csv`
- 总表 JSON：
  - `outputs/5.4-tabpfn-10k-8traits-fusion/tabpfn_10k_8traits_compare.json`

## 需要知道的正式结果目录

- 正式服务器来源：
  - `server@GPU1:/home/server/code/git/TabICLv2-test`
- TabPFN no-prior：
  - `outputs/5.4-tabpfn-10k-8traits`
- TabPFN fusion：
  - `outputs/5.4-tabpfn-10k-8traits-fusion`
- baseline 复用来源：
  - `outputs/5.4-duli-liudang/baseline_3models`

## 写作任务

请把 `TabPFN` 结果自然整合进现有 `Results` 主体，不要写成实验日志，也不要写成与主线并列竞争的新主结果。它的定位是：

- 用一个额外的 table / foundation model 进行补充验证
- 说明“多 prior 融合框架”具有一定可扩展性
- 但同时强调：不同 foundation model 在 GS 中的适配程度并不相同
- 当前主线仍然是 `TabICL`

## 推荐写法

建议不要新增一个完全独立、与 Result 1-4 平行的大章节；更合适的做法是：

- 在 `Result 4` 末尾增加一个补充小节，或
- 在 marker count / sample size 那一章后面增加一个短小的 “framework extensibility” 小节

核心是：不打乱现在的 4 章结构，但把 `TabPFN` 作为一个有价值的补充验证嵌进去。

## 必须保持的主线

1. `TabICL` 仍然是全文主线与主结果来源。
2. `TabPFN` 不是来替代 `TabICL`，而是作为补充模型验证“prior-integrated fusion` 这条思路不是只对单一 foundation model 成立。
3. 该结果不能被写成“所有 foundation model 都能同样有效地进入 GS”。
4. 更准确的表述应是：
   - foundation model 单独用于 GS 时，未必稳定优于强统计模型
   - 但在多 prior 融合框架下，它们可以在一定程度上释放补充价值
   - 不同 foundation model 的收益幅度和稳定性并不相同

## 建议你在正文里明确写出的事实

请基于现有总表，把以下事实写进去：

- 当前 TabPFN 补充实验基于 `10K SNP + 8 个 SNP-count traits`
- `no_prior-TabPFN` 的平均 Pearson 为 `0.5986`
- 三个 baseline 的平均 Pearson 分别为：
  - `BayesB = 0.6314`
  - `GBLUP = 0.6208`
  - `RKHS = 0.6271`
- `only-triple-fusion` 的平均 Pearson 为 `0.6325`
- `triple-fusion` 的平均 Pearson 为 `0.6352`
- `triple-fusion` 相对 `only-triple-fusion` 的平均提升为 `+0.3496%`
- 但 `triple-fusion` 相对 trait 内最优 baseline 的平均提升为 `-0.1230%`

还请点出 trait-level 的异质性：

- `triple-fusion` 相对 trait 内最优 baseline 为正的 trait 有 4 个：
  - `cotton_fibelo`
  - `cotton_fiblen`
  - `grain_weight`
  - `lw`
- 为负的 trait 也有 4 个：
  - `grain_width`
  - `bbd`
  - `sl_e1`
  - `sl_e2`

还请写出相对各 baseline 的平均提升：

- `only-triple-fusion` 相对：
  - `BayesB`：`+0.2915%`
  - `GBLUP`：`+2.0425%`
  - `RKHS`：`+0.6709%`
- `triple-fusion` 相对：
  - `BayesB`：`+0.6315%`
  - `GBLUP`：`+2.4058%`
  - `RKHS`：`+1.0401%`

## 语气和结论边界

请用论文正文语气来写，不要用汇报口吻。

请避免以下写法：

- “TabPFN 证明了所有 foundation model 都适合 GS”
- “TabPFN 结果与 TabICL 同样强”
- “融合后稳定超过最优 baseline”

更合适的落点是：

- 该补充验证支持融合框架具有一定模型可迁移性
- 但模型适配程度存在明显差异
- `TabPFN` 在当前设定下可以从多 prior 中获益，但整体稳定性仍弱于 `TabICL`
- 因而这部分结果加强的是“框架可扩展性”，而不是替换主线结论

## 输出要求

请直接产出可并入 `docs/results_1_4_cn_draft.md` 的正式学术中文段落。

最好给出两种形式之一：

- 方案 A：新增一个短小节标题 + 2 到 4 段正文
- 方案 B：直接在 `Result 4` 末尾续写 2 到 3 段自然衔接的正文

优先目标是：

- 不破坏现有主线
- 自然抬高“融合框架可扩展性”的方法学价值
- 同时诚实保留 `TabPFN` 不如 `TabICL` 稳定这一事实
