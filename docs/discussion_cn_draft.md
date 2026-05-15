# Discussion 中文草稿

本研究的核心发现并不是 `TabICL` 单独进入 genomic selection（GS）后能够稳定超越经典统计模型，而是：即使 table foundation model 不是为 GS 原生设计、也难以在 no-prior 设定下形成稳定最优，它仍然可以在与统计遗传先验进行结构化结合后释放出可重复的预测价值。这一区分非常重要，因为它把研究问题从“foundation model 是否能够直接替代经典 GS 模型”转向了“foundation model 是否能够作为新的信息来源，被有效吸收进现有 GS 框架”。从这一意义上说，本文提供的不只是一个关于 `TabICL` 的经验案例，更是一条让非 GS 原生模型进入 GS 领域的可行方法路径。

这一结论首先建立在一个相对克制但更有说服力的事实上：`TabICL` 单独使用时并非没有价值，但其优势是局部的、trait-dependent 的，而不是跨 trait 的稳定压制。当前主结果显示，`no_prior-TabICL` 在若干 trait 上已经能够超过 `BayesB`、`GBLUP` 或 `RKHS`，说明 table foundation model 确实能够从基因型数据中恢复出一部分有效信号；但从整体平均水平看，它仍未能稳定超过这三类经典 statistical baselines。这一点意味着，如果我们把评价标准简单设定为“是否直接打败经典模型”，那么像 `TabICL` 这样的模型很容易被过早判定为在 GS 中“价值有限”。而本文的结果恰恰表明，这种判断可能过于狭窄。对于 GS 这类先验结构非常强、而且已有模型长期积累的方法领域，一个新模型的价值未必首先体现为“替代”，也可能体现为“补充”。

这种补充价值在与统计 prior 融合后得到了更清楚的体现。无论是 `single-prior-TabICL` 还是 `triple-prior-TabICL`，它们相对对应 baseline 或 prior-only 的准确率提升都表现出广泛而稳定的正向趋势。特别是 `single-prior` 的结果说明，`TabICL` 并不是只对某一类统计假设有效，而是能够在 `BayesB`、`GBLUP` 和 `RKHS` 三类不同建模范式上都提供增益。这一点很关键，因为它提示 `TabICL` 并非简单复现了某一条统计模型已经编码好的信息。如果它只是在学习某一种 prior 的近似替代，那么其收益理应集中在少数与该 prior 最接近的 trait 或方法上；而当前观察到的是一种更广泛但幅度不完全相同的增益格局。换言之，`TabICL` 的作用更像是为不同类型的统计 prior 提供了一个共享但不完全重叠的补充信息源。

`triple-prior-TabICL` 的结果进一步强化了这一判断。与三条正式 baseline 分别比较时，triple 在平均水平和 trait 覆盖面上都表现出更高的整体稳健性；而与最优 single 直接比较时，其优势并不体现为“在每个 trait 上都显著更强”，而更多体现为一种统一性与稳健性。也就是说，triple 的意义不在于绝对压制所有 single-prior 方案，而在于在无需事先判断哪一种 prior 最匹配某个 trait 的前提下，给出一条整体表现足够靠前、且跨 trait 更稳定的统一主结果线路。对于真实育种任务而言，这种“无需逐 trait 调路线、仍能获得稳定增益”的性质，往往比在少数 trait 上追求极致局部最优更具实践价值。

从机制层面看，single-prior 融合权重提供了一条与性能结果方向一致的证据。`w_tabicl` 在三条 single-prior 线路中并没有系统性塌缩到接近零，而是在大多数 trait 上维持在接近一半的范围，并且呈现出明确的 trait 间异质性。这并不能单独证明 `TabICL` 学到了某种全新的生物学机制，但至少表明，在当前的 OLS 融合框架下，`TabICL` 并不是一个可以忽略的小修正项。更准确地说，统计 prior 与 `TabICL` 的相对贡献会随着 trait 改变：在某些 trait 上，OLS 更倾向于保留较高比例的 `TabICL`；在另一些 trait 上，统计 prior 占据更主导的位置。这种 trait-dependent 的权重分配，与我们在 Result 2 中观察到的性能异质性是相互呼应的，也提示未来可以进一步把“trait 类型”与“fusion 权重模式”系统联系起来。

样本量和 marker-count 两组补充实验，则为这种融合思路提供了比单一 full-data 场景更扎实的边界信息。它们共同说明，多 prior 融合的收益不是一条简单的单调规律，而是明显依赖于 trait 本身的遗传结构以及当前可用信息的丰富程度。样本量实验显示，相对 `BayesB` 的增益更像是一种边际补充，往往在样本增多后收敛；而相对 `GBLUP` 与 `RKHS` 的增益则在部分 trait 上反而会随着样本量增加而进一步显现。marker-count 实验则表明，增加 SNP 密度通常会提高融合模型的绝对准确率，但并不会保证它在所有 trait 上都同步扩大相对优势。特别是 `grain_weight`、`grain_width` 和 `lw` 这类 trait，反复说明“prior 数量增加”并不自动等于“无条件更优”，真正关键的是 prior 与 trait 结构之间是否存在有效匹配。因而，本文更愿意把融合收益描述为一种有条件的、结构依赖的优势，而不是一个普适、单调、无需前提的提升规律。

这也引出本文在方法学上的一个更一般性启发。GS 领域长期由具有明确遗传假设的统计模型主导，而 foundation model 的优势则更多来自灵活表示、弱结构假设和跨任务泛化潜力。当前结果提示，这两类模型之间未必是简单的替代关系。对于 GS 而言，更现实也更有价值的方向，可能不是要求 foundation model 先单独打败所有经典模型，而是探索它如何与已有统计先验进行有效分工。换句话说，statistical priors 仍然承担着对遗传结构的强约束，而 foundation model 则可能提供一种补充性的 data-driven 表示或残差信息。本文的 prior-integrated fusion 框架，本质上正是对这种分工关系的一个初步实现。

`TabPFN` 的补充验证进一步支持了这一点，同时也帮助我们划清了结论边界。它表明，prior-integrated fusion 这一路线并不只对 `TabICL` 单一 backbone 有效；即使换成另一类 table / foundation model，在与统计 prior 融合后，也仍然能够获得一定程度的平均增益。但与此同时，`TabPFN` 的整体稳定性明显弱于 `TabICL` 主线，且并未形成相对 trait 内最优 baseline 的稳定领先。这一点非常重要，因为它说明本文的结论不应被夸大为“所有 foundation model 都可以同样有效地进入 GS”。更准确的说法应是：foundation model 单独进入 GS 时通常未必稳定优于强统计模型，但其中一部分模型在与统计 prior 结构化结合后，能够释放出补充价值；只是不同 model 的适配程度、收益幅度和稳定性并不相同。

当然，本文仍有几个需要明确承认的局限。第一，当前主线 backbone 仍然以 `TabICL` 为主，尽管 `TabPFN` 提供了一个有价值的外部支点，但还不足以把本文结论推广为“面向所有 foundation model 的一般定律”。第二，我们当前正式 prior 池固定为 `BayesB`、`GBLUP` 和 `RKHS`，它们已经覆盖了 GS 中三类非常核心的建模思路，但仍不代表全部可能的 statistical priors。第三，当前融合方式采用的是 trait 级、基于 inner OOF 的线性权重学习，这一设计有助于控制信息泄露、降低比较复杂度，也更适合当前阶段的主文主线；但它仍然是一种相对保守的融合器，未来完全可能扩展到更灵活的样本级或结构感知型融合规则。第四，尽管样本量实验现已补齐到 `8` 个 trait，但这一面板的规模仍然明显小于主线 `36` 个非猪 trait，因此有关 sample-size dependence 的结论更适合被理解为在代表性 trait 面板上的系统证据，而不是对全部 trait 的无条件推广。

在未来工作中，最直接的延伸方向有三类。其一，是扩展更多非 GS 原生模型，检验“先验融合而非单独替代”这一思路是否对更广泛的 table / representation / foundation backbones 同样成立。其二，是扩展 prior 池本身，包括引入更多具有不同遗传结构偏好的 statistical models，进而更系统地研究“prior 匹配度”与“fusion 收益”之间的关系。其三，是把当前的 trait 级线性融合推进到更一般的融合器形式，例如结合 trait 类型、baseline geometry 或更细粒度表示信息，让不同 prior 与 foundation model 的协同关系能够以更自适应的方式表达。

总体而言，本文最重要的贡献并不在于证明 `TabICL` 已经成为一个单独优于经典 GS 模型的新基线，而在于提出并验证了一种更具现实意义的视角：**当 foundation model 直接迁移到 GS 场景时，它未必能够单独稳定奏效；但如果把它视为一个可以与统计遗传先验协同工作的补充信息源，那么它就可能在 GS 中发挥出实际价值。** 这一点不仅解释了为什么 `TabICL` 在本文中能够通过 prior-integrated fusion 获得稳定增益，也为未来更多非 GS 原生模型进入 GS 提供了一条比“直接替代经典模型”更可行的方法学路径。
