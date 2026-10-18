# OTTA v0.2 方法与实现定义

> 状态：中期版本冻结  
> 更新时间：2026-09-12  
> 目标：在最小在线 TTA 实验中检验“优化器状态能否改善样本选择”这一假设。

## 1. 研究假设

高困惑度表示模型难以解释一个输入，但不代表用该输入更新一定有利。OTTA v0.2 检验：

> 在短测试流中，与历史有效更新方向一致的候选样本，可能比单纯的高困惑度样本带来更稳定的 LoRA 测试时更新。

这是待验证假设。中期材料不得把“可能有用”写成“已经证明有用”。

## 2. 在线无标签设定

模型由冻结基座和可训练 LoRA 参数 θ 组成。测试样本按 batch 到达：

```text
B_t 到达
  -> 用更新前模型预测 B_t 并记录结果
  -> 隐藏标签，仅用 B_t 输入计算选择分数
  -> 选择 k 个样本并更新 LoRA
  -> 更新后的模型仅用于 B_(t+1) 及以后
```

标签只允许由 evaluator 在预测保存后读取，不进入 prompt、选择器或适应损失。该 predict-before-adapt 协议避免在同一 batch 上先更新再汇报性能。

## 3. 自监督适应目标

对不包含答案的输入文本 x_i，使用 token-average causal language modeling loss：

```text
ℓ_i = -(1/T_i) Σ_j log p(x_i,j | x_i,<j)
```

输入文本由题干和全部选项组成，但不包含正确选项标记。token average 用于降低文本长度对选择分数的影响。

## 4. 优化器感知分数

### 4.1 样本梯度

仅对 LoRA 参数计算单样本梯度：

```text
g_i = ∇_θ ℓ_i
```

打分时不调用 `optimizer.step()`，不改变模型与 AdamW 状态。

### 4.2 统一的 AdamW 预条件空间

读取打分时刻 AdamW 的一阶矩和二阶矩，并进行 bias correction：

```text
m_hat_t = m_t / (1 - β1^t)
v_hat_t = v_t / (1 - β2^t)
```

候选方向与历史参考方向统一定义为：

```text
u_i   = g_i     / (sqrt(v_hat_t) + ε)
d_ref = m_hat_t / (sqrt(v_hat_t) + ε)
```

两者都采用相同二阶矩缩放，因此余弦位于同一优化器诱导空间。AdamW 参数更新带负号，但候选梯度和历史梯度同时取梯度方向，余弦排序不受共同负号影响。中期版本设 `weight_decay=0`，不处理与样本无关的衰减项。

### 4.3 块级余弦

```text
utility(x_i) = <u_i, d_ref> / (||u_i||₂ ||d_ref||₂ + δ)
```

实现时不拼接全部 LoRA 参数，也不构造稠密随机投影矩阵。对每个 LoRA 参数块累计点积与平方范数：

```text
dot    = Σ_p <u_i^(p), d_ref^(p)>
norm_u = sqrt(Σ_p ||u_i^(p)||²)
norm_d = sqrt(Σ_p ||d_ref^(p)||²)
score  = dot / max(norm_u * norm_d, 1e-12)
```

第一版删除随机投影。只有主实验完成且实测表明块级余弦是瓶颈时，才研究稀疏投影或 CountSketch。

## 5. 冷启动与异常回退

第一个适应 batch 尚无 AdamW 历史状态，因此使用 High-PPL 选择并完成一次 LoRA 更新。从第二个 batch 开始启用 OTTA。

以下情况当前 batch 回退为 High-PPL，并在日志中记录 `fallback_reason`：

- AdamW 状态尚未初始化；
- `d_ref` 范数小于 `1e-12`；
- 任一 utility 为 NaN/Inf；
- 可用候选数小于 k。

主方法不加入困惑度保护带，避免实验提升无法归因。若发现极端 PPL 样本导致明显不稳定，再将 `OTTA-gated` 作为后续消融，而不是修改主方法定义。

## 6. 选择与更新

对候选 batch 中所有样本计算 utility，取最高的 k 个：

```text
S_t = TopK({utility(x_i)}, k)
L_t = mean({ℓ_i | i ∈ S_t})
θ_(t+1) = AdamW(θ_t, ∇_θ L_t)
```

为减少图复用错误，单样本打分梯度和最终更新梯度分开计算。打分结束后重新前向选中样本并计算更新 loss。

## 7. 完整伪代码

```text
initialize frozen backbone + LoRA θ
initialize AdamW optimizer
set model state identical for each method

for t, B_t in test_stream:
    # A. 先预测；标签仅在 evaluator 中使用
    predictions = predict_choices(model, B_t.inputs)
    evaluator.record(predictions, B_t.labels)

    # B. 无标签适应
    losses = token_average_nll(model, B_t.inputs_without_labels)

    if t == 0 or optimizer_state_is_empty():
        selected = topk(losses, k)
        fallback_reason = "cold_start"
    else:
        m_hat, v_hat = read_bias_corrected_adam_state(optimizer)
        d_ref = m_hat / (sqrt(v_hat) + eps)

        if block_norm(d_ref) < 1e-12:
            selected = topk(losses, k)
            fallback_reason = "zero_reference"
        else:
            for each sample i:
                g_i = autograd_grad(loss_i, lora_params)
                u_i = g_i / (sqrt(v_hat) + eps)
                score_i = blockwise_cosine(u_i, d_ref)

            if any score is NaN or Inf:
                selected = topk(losses, k)
                fallback_reason = "invalid_score"
            else:
                selected = topk(scores, k)

    # C. 重新前向，只用选中样本更新一次
    update_loss = mean_nll(model, selected_inputs)
    optimizer.zero_grad()
    update_loss.backward()
    optimizer.step()

    save predictions, losses, scores, selected_ids,
         fallback_reason, time, and peak_memory
```

## 8. 必做对比与唯一优先消融

| 方法 | 选择规则 | 更新 | 研究作用 |
|---|---|---|---|
| No Adaptation | 无 | 否 | 原始性能 |
| Random | seed 固定的均匀随机 | LoRA | 随机下界 |
| High-PPL | token-average NLL top-k | LoRA | TLM 风格直接对照 |
| OTTA v0 | optimizer-aware cosine top-k | LoRA | 项目方法 |
| OTTA-raw | raw gradient 与 raw moment 余弦 | LoRA | 主表完成后，验证二阶矩缩放的贡献 |

中期主表没有完成前，不实现 verifier、Boltzmann、PPL gate、随机投影和多参考方向。

## 9. 实现接口

```python
def select_otta_v0(
    sample_losses,
    lora_params,
    optimizer,
    k: int,
    norm_floor: float = 1e-12,
):
    """
    Returns:
        selected_idx: list[int]
        utilities: list[float | None]
        fallback_reason: str | None
    """
```

实现必须读取 PyTorch AdamW 状态中的 `exp_avg`、`exp_avg_sq` 和 `step`，且只处理当前确实参与优化的 LoRA 参数。

## 10. 每个 batch 必须记录

- batch_id、sample_ids、stream_position；
- 每个样本的 token 数与 NLL/PPL；
- 每个样本的 utility，冷启动时为 null；
- selected_ids 与 fallback_reason；
- 更新前预测和正确性；
- update_loss、累计 accuracy；
- 推理、打分、更新耗时；
- 峰值显存。

## 11. 可证伪标准

- 若 OTTA 与 High-PPL 选样几乎完全相同，说明 optimizer-aware 分数没有提供独立信息。
- 若 OTTA-raw 与 OTTA v0 无差异，说明二阶矩缩放暂未产生贡献。
- 若 OTTA alignment 高但后续窗口 accuracy 更低，说明“方向一致等于有用”的假设不成立或发生方向锁定。
- 若性能相近但成本显著增加，应将结果表述为效率不划算，而非有效方法。

## 12. 中期表述边界

可以说：提出并实现了一个 optimizer-aware 样本选择原型，并完成初步验证。

不能说：证明了该方法普遍优于困惑度选择、解决了 LLM TTA 数据选择问题，或完整复现了 OPUS/TLM。

