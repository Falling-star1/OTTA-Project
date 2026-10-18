# Tent 初读笔记

## 基本信息

- 论文名：Tent: Fully Test-Time Adaptation by Entropy Minimization
- 年份/会议：ICLR 2021
- 链接：https://arxiv.org/abs/2006.10726
- OpenReview：https://openreview.net/forum?id=uXl3bZLkr3c
- 代码：https://github.com/DequanWang/TENT
- 阅读人：
- 阅读日期：2026-07-29

## 一句话总结

Tent 是经典 TTA baseline：测试时只用无标签目标数据，通过最小化预测熵来更新模型的一部分参数，使模型对目标域预测更自信。

## 它解决的问题

模型训练好后，在测试时遇到分布偏移会性能下降。Tent 研究 fully test-time adaptation 设置：测试阶段只有测试数据和模型参数，没有源数据和标签。

## 方法流程

1. 接收一个测试 batch。
2. 前向计算预测分布。
3. 用预测熵作为无监督损失。
4. 在线更新模型中的归一化统计量和通道级仿射参数。
5. 用更新后的模型进行当前或后续 batch 推理。

## 样本选择机制

- 选择依据：原始 Tent 基本不做复杂样本选择，通常对 batch 做熵最小化。
- 是否需要标签：不需要。
- 是否需要源域数据：不需要。
- 是否需要验证器：不需要。
- 是否考虑优化器状态：不考虑。

## 模型更新方式

- 更新参数：主要是 normalization statistics 和 channel-wise affine transformations。
- 优化目标：预测熵最小化。
- 是否在线：是，batch-by-batch。
- 是否适合 LLM：原始方法主要用于视觉分类，不直接适配 LLM。

## 实验设置

论文主要在图像分类分布偏移任务上验证，如 corruption、simulation-to-real 等。官方 GitHub 提供 PyTorch 示例，强调每个 batch 做一次更新以避免明显打断推理。

## 可以借鉴的地方

1. 是 TTA 领域最基础的思想起点。
2. 可以用作“熵最小化类方法”的理论背景。
3. 对说明 TTA 设置很有帮助：无标签、无源数据、测试时在线更新。

## 不足或可改进点

1. 原始 Tent 不做样本选择，可能对坏样本也更新。
2. 熵最小化容易让模型过度自信。
3. 主要用于视觉模型，LLM 迁移需要重新定义目标。
4. 不考虑灾难性遗忘、样本冗余和优化器状态。

## 和 OTTA 的关系

Tent 不是本项目直接复现的最高优先级，但适合作为文献综述里的经典 baseline。

可以写成：

> Tent 证明了测试时无监督更新可以缓解分布偏移，但其样本使用策略较粗，不区分哪些测试样本真正有利于更新。后续 EATA、TLM、VDS-TTT 和本项目 OTTA 都可以看作在“测试时更新”基础上进一步改进样本选择。

## 读完后的下一步

- [x] 作为 TTA 背景和经典 baseline。
- [x] 不建议第一阶段优先复现，除非需要传统视觉 TTA 对比。
- [ ] 可在综述中用于引出“熵最小化”和“过度自信”问题。

## 参考来源

- arXiv：https://arxiv.org/abs/2006.10726
- OpenReview：https://openreview.net/forum?id=uXl3bZLkr3c
- GitHub：https://github.com/DequanWang/TENT

