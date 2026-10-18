# TLM 初读笔记

## 基本信息

- 论文名：Test-Time Learning for Large Language Models
- 方法简称：TLM
- 年份/会议：ICML 2025
- 链接：https://proceedings.mlr.press/v267/hu25z.html
- arXiv：https://arxiv.org/abs/2505.20633
- 代码：https://github.com/Fhujinwu/TLM
- 阅读人：
- 阅读日期：2026-07-29

## 一句话总结

TLM 把 LLM 测试时学习表述为输入困惑度最小化：用无标签测试数据在测试阶段更新 LoRA，并主动选择高困惑度样本提升样本效率。

## 它解决的问题

LLM 面对专业领域、语言风格变化等分布偏移时性能下降。TLM 希望只利用无标签测试数据，让模型在测试阶段动态适应目标域。

## 方法流程

1. 输入一批无标签测试样本。
2. 计算样本输入困惑度。
3. 选择或强调高困惑度样本，因为它们被认为对优化更有信息量。
4. 以输入困惑度最小化作为自监督目标。
5. 通过 LoRA 更新模型，避免全参数更新带来的遗忘和高开销。
6. 在 AdaptEval 等 benchmark 上评估领域知识适应效果。

## 样本选择机制

- 选择依据：高输入困惑度。
- 是否需要标签：不需要。
- 是否需要源域数据：不需要。
- 是否需要验证器：不需要。
- 是否考虑优化器状态：不考虑。

## 模型更新方式

- 更新参数：LoRA。
- 优化目标：input perplexity minimization。
- 是否在线：是，测试时动态适应。
- 是否适合 LLM：是。

## 实验设置

PMLR 页面说明 TLM 提出了 AdaptEval benchmark，并报告在领域知识适应任务上相比原始 LLM 至少提升 20%。这对本项目很有参考价值，因为开题文本里也把 AdaptEval 作为候选评估基准。

## 可以借鉴的地方

1. TLM 是本项目最适合优先复现的 baseline。
2. 它不依赖 verifier，工程门槛比 VDS-TTT 低。
3. 高困惑度选择可以作为 OTTA 的强对照。
4. AdaptEval 和 LoRA 测试时更新流程值得直接参考。

## 不足或可改进点

1. 高困惑度不一定等于高价值样本。
2. 高困惑度样本可能是噪声、异常输入或模型无法可靠学习的坏样本。
3. 选择机制没有考虑当前优化器状态、更新方向和历史动量。
4. 如果一直强调高困惑度样本，可能造成过拟合或误差累积。

## 和 OTTA 的关系

TLM 是 OTTA 的最关键 baseline 之一。OTTA 的动机可以直接对比 TLM：

```text
TLM: 这个样本难，所以值得学。
OTTA: 这个样本不只是难，而且它的梯度方向对当前优化器更新有帮助，所以值得学。
```

可设计实验：

| 策略 | 分数 |
|---|---|
| TLM-style | perplexity(x) |
| OTTA v0 | alignment(projected_gradient(x), optimizer_update_direction) |
| Hybrid | perplexity(x) * positive_alignment(x) |

## 读完后的下一步

- [x] 作为优先复现 baseline。
- [x] 记录官方代码地址。
- [ ] 克隆并检查 TLM 代码依赖。
- [ ] 确认 AdaptEval 数据集下载方式。
- [ ] 设计 TLM vs OTTA 的对比实验。

## 参考来源

- PMLR：https://proceedings.mlr.press/v267/hu25z.html
- arXiv：https://arxiv.org/abs/2505.20633
- GitHub：https://github.com/Fhujinwu/TLM

