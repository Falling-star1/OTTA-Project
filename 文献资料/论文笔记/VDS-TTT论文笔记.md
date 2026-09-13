# VDS-TTT 初读笔记

## 基本信息

- 论文名：Continuous Self-Improvement of Large Language Models by Test-time Training with Verifier-Driven Sample Selection
- 方法简称：VDS-TTT
- 年份：2025
- 链接：https://arxiv.org/abs/2505.19475
- OpenReview：https://openreview.net/forum?id=6ahliSpvQ0
- 阅读人：
- 阅读日期：2026-07-29

## 一句话总结

VDS-TTT 让 LLM 在测试时生成多个候选答案，用 verifier 给候选答案打分，只把高可信伪标签样本用于 LoRA 测试时训练，从而实现连续自我改进。

## 它解决的问题

LLM 在训练分布外的推理任务上可能表现下降。VDS-TTT 试图不用人工标签，而是利用模型自己生成的答案和一个 learned verifier，在测试阶段构造伪训练样本，让模型持续适应新任务或新分布。

## 方法流程

对每个测试 query：

1. LLM 生成多个候选回答。
2. verifier 对每个候选回答打可靠性分数。
3. 选择分数最高且超过阈值的回答作为伪标签。
4. 将 `(query, selected_answer)` 加入测试时训练数据。
5. 只微调 LoRA adapter 参数。
6. 更新后的模型继续处理后续测试样本。

## 样本选择机制

- 选择依据：verifier reliability score。
- 是否需要标签：不需要人工标签，但需要 verifier。
- 是否需要源域数据：不需要。
- 是否需要验证器：需要，这是核心组件。
- 是否考虑优化器状态：不考虑。

## 模型更新方式

- 更新参数：LoRA 低秩适配器。
- 优化目标：把 verifier 选出的高置信伪标签作为训练目标进行 fine-tuning。
- 是否在线：是。
- 是否适合 LLM：是，直接面向 LLM。

## 实验设置

arXiv 摘要称，VDS-TTT 在三个 benchmark 和三个先进 LLM 上实验，最高相对 base model 提升 32.29%，相比不做 test-time training 的 verifier-based 方法提升 6.66%。

## 可以借鉴的地方

1. 是本项目最直接的 LLM-TTA baseline。
2. LoRA 测试时更新流程可以参考。
3. “选择高质量伪标签样本再更新”的流程和 OTTA 的“选择-适应-反馈”闭环相似。
4. 可以作为实验对照：verifier score selection vs optimizer-aware utility selection。

## 不足或可改进点

1. 样本选择依赖 verifier，搭建成本较高。
2. 使用固定阈值，可能对不同任务/分布不稳。
3. 选择标准主要衡量答案可靠性，没有显式考虑这个样本对当前优化器更新方向是否有用。
4. 可能偏向模型已经会的高置信样本，存在确认偏误风险。

## 和 OTTA 的关系

VDS-TTT 可以作为 OTTA 的直接对照。区别可以写成：

| 方法 | 选择依据 | 更新方式 | 局限 |
|---|---|---|---|
| VDS-TTT | verifier 高分伪标签 | LoRA | 不考虑优化器更新方向 |
| OTTA | 优化器感知效用 | LoRA | 需要设计低开销效用估计 |

OTTA 可以尝试替换 VDS-TTT 的 verifier 阈值选择模块：

```text
VDS-TTT: select if verifier_score(x, y) > threshold
OTTA: select if utility_optimizer_aware(x, y) is high
```

## 读完后的下一步

- [x] 作为 LLM-TTA baseline。
- [x] 写入文献综述“现有 LLM-TTA 样本选择较简单”的证据。
- [ ] 查找是否有官方代码。
- [ ] 如果 verifier 成本太高，先用模型置信度或自一致性分数模拟。

## 参考来源

- arXiv：https://arxiv.org/abs/2505.19475
- OpenReview：https://openreview.net/forum?id=6ahliSpvQ0

