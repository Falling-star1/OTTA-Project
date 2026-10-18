# OPUS 初读笔记

## 基本信息

- 论文名：OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration
- 年份：2026
- 链接：https://arxiv.org/abs/2602.05400
- 代码：https://github.com/gszfwsb/OPUS
- 阅读人：
- 阅读日期：2026-07-29

## 一句话总结

OPUS 提出了一种面向 LLM 预训练的数据选择方法：样本价值不再用原始梯度估计，而是在 AdamW/Muon 等优化器实际塑造的“有效更新空间”中衡量。

## 它解决的问题

LLM 预训练正在从“更多 token”转向“更好的 token”。传统数据选择方法有两类局限：

1. 静态过滤器忽略训练过程中的模型状态变化。
2. 动态方法常用原始梯度衡量样本效用，但现代 LLM 训练主要使用 AdamW、Muon 等优化器，优化器会改变真实更新方向。

OPUS 把这个问题称为 raw-gradient scoring 和 optimizer-induced update 之间的 misalignment gap。

## 方法流程

1. 构造一个稳定的高质量 proxy 集合作为目标分布。
2. 计算 proxy 方向，表示当前模型应该朝哪个方向改进。
3. 对候选训练样本估计其优化器诱导后的有效更新方向。
4. 用候选样本有效更新方向与 proxy 目标方向的投影/对齐程度作为 utility score。
5. 使用 Ghost technique 与 CountSketch 避免显式保存完整 per-sample gradient。
6. 用 Boltzmann sampling 选择样本，兼顾高效用和多样性。

## 样本选择机制

- 选择依据：样本在优化器诱导更新空间中对 proxy 方向的一步改进贡献。
- 是否需要标签：不强调标签，主要用于预训练语料选择。
- 是否需要源域数据：需要训练语料池和 proxy 数据。
- 是否需要验证器：不需要。
- 是否考虑优化器状态：是，核心创新就是 optimizer-aware。

## 模型更新方式

- 更新场景：LLM 预训练或继续预训练。
- 优化器：AdamW、Muon 等现代优化器。
- 更新参数：完整预训练参数，不是 LoRA 场景。
- 关键技术：Ghost technique、CountSketch、Boltzmann sampling。

## 实验设置

论文报告了多种语料、质量层级、优化器和模型规模下的实验。arXiv 摘要中提到，OPUS 在 GPT-2 Large/XL 的 30B tokens 预训练中超过工业级 baseline，甚至超过完整 200B-token 训练效果；继续预训练 Qwen3-8B-Base 时，也用更少 token 获得更好效果。

## 可以借鉴的地方

这是本项目最重要的理论来源。可借鉴：

1. 样本效用不应只看原始梯度，而要看优化器实际更新方向。
2. 用投影或对齐度衡量样本是否推动模型朝目标方向改进。
3. 用低维投影近似高维梯度，控制额外开销。
4. 选择策略不要只取 top-k，可以引入随机性保持多样性。

## 不足或可改进点

相对本大创课题，OPUS 有明显迁移缺口：

1. 它面向预训练，不是测试时自适应。
2. 它依赖 proxy 分布，而 TTA 场景通常只有无标签测试流。
3. 它的目标是训练效率，不是在线推理阶段的持续适应。
4. 完整实现可能对算力要求较高，不适合直接照搬。

## 和 OTTA 的关系

OPUS 是 OTTA 的核心思想来源。可以把它从预训练场景改造成：

```text
OPUS: 候选训练样本 -> 对 proxy 目标方向的优化器诱导贡献
OTTA: 候选测试样本 -> 对当前 LoRA/优化器有效更新方向的贡献
```

第一版 OTTA 可以不复刻完整 Ghost+CountSketch，而是先做简化：

```text
utility(x) = cosine(projected_lora_gradient(x), current_optimizer_update_direction)
```

## 读完后的下一步

- [x] 写入理论来源。
- [x] 作为 OTTA 效用分数设计依据。
- [ ] 进一步读方法章节，明确 AdamW effective update 的近似公式。
- [ ] 检查官方代码里 utility scorer 的实现。

## 参考来源

- arXiv：https://arxiv.org/abs/2602.05400
- arXiv HTML：https://arxiv.org/html/2602.05400
- GitHub：https://github.com/gszfwsb/OPUS

