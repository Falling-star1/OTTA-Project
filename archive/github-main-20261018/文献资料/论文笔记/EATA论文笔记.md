# EATA 初读笔记

## 基本信息

- 论文名：Efficient Test-Time Model Adaptation without Forgetting
- 方法简称：EATA
- 年份/会议：ICML 2022
- 链接：https://proceedings.mlr.press/v162/niu22a.html
- arXiv：https://arxiv.org/abs/2204.02610
- 代码：https://github.com/mr-eggplant/EATA
- 阅读人：
- 阅读日期：2026-07-29

## 一句话总结

EATA 在 Tent 式熵最小化基础上加入“可靠且非冗余”的主动样本选择，并用 Fisher 正则缓解测试时更新导致的遗忘。

## 它解决的问题

传统 TTA 方法常对每个测试样本反向传播，带来高计算开销；同时，模型在 OOD 数据上适应后，可能在 ID 数据上性能下降，即 catastrophic forgetting。

## 方法流程

1. 接收测试样本。
2. 判断样本是否可靠，过滤掉不适合更新的样本。
3. 判断样本是否与已有样本冗余，避免重复更新。
4. 对保留样本做熵最小化更新。
5. 加入 Fisher regularizer，限制重要参数剧烈变化，缓解遗忘。

## 样本选择机制

- 选择依据：可靠性 + 非冗余性。
- 是否需要标签：不需要。
- 是否需要源域数据：不需要源训练数据，但 Fisher 重要性可由测试样本伪标签估计。
- 是否需要验证器：不需要。
- 是否考虑优化器状态：不直接考虑。

## 模型更新方式

- 更新参数：视觉模型参数，具体依实现而定。
- 优化目标：sample-efficient entropy minimization + Fisher regularization。
- 是否在线：是。
- 是否适合 LLM：原始方法主要用于视觉，但思想可迁移。

## 实验设置

PMLR 页面说明 EATA 在 CIFAR-10-C、ImageNet-C、ImageNet-R 上验证，目标是同时提升 OOD 性能、减少反向传播开销、缓解 ID 遗忘。

## 可以借鉴的地方

1. “不是所有测试样本都值得更新”是本项目的重要论据。
2. 可靠性和非冗余过滤可以作为 OTTA 的辅助模块。
3. Fisher regularizer 的抗遗忘思想可用于后续阶段。
4. 可以与 OTTA 的优化器感知选择形成互补：EATA 解决可靠/冗余，OTTA 解决更新方向价值。

## 不足或可改进点

1. 面向视觉分类，不是 LLM。
2. 样本选择基于不确定性和冗余，不考虑 optimizer-induced utility。
3. 熵最小化仍可能引入过度自信问题。
4. Fisher regularizer 增加额外实现复杂度。

## 和 OTTA 的关系

EATA 适合放在“样本选择类 TTA 方法”的相关工作中。它支持本项目的一个关键论点：

> 测试时适应不能无差别使用所有测试样本，样本选择会影响效率、性能和遗忘。

OTTA 可以进一步提出：

> 可靠性和非冗余性还不够，还应考虑样本是否沿着当前优化器诱导的有效更新方向贡献改进。

## 读完后的下一步

- [x] 作为样本选择类 TTA 参考。
- [x] 作为抗遗忘和非冗余机制参考。
- [ ] 后续可读 EATA-C，了解校准版如何处理过度自信。
- [ ] 如果 OTTA 误差累积严重，可引入 EATA 式过滤作为增强。

## 参考来源

- PMLR：https://proceedings.mlr.press/v162/niu22a.html
- arXiv：https://arxiv.org/abs/2204.02610
- GitHub：https://github.com/mr-eggplant/EATA

