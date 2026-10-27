# Dolly 与 GSM8K 复现排查

更新：2026-10-27 08:45（北京时间）。这是排查记录，不是“修复完成”声明。原始权重、回答和评分均保留；机制对照不改名为论文原方法。

## 已确认的全量结果

Llama 三组均已完成 5000 条评测：GeoSignal 为 Source 23.98、TLM 29.32；Dolly 为 Source 33.33、TLM 24.07；GSM8K 为 Source 77.60%、TLM 77.98%。Qwen GeoSignal 全量为 Source 24.94、TLM 29.64，另两组仍按各自实际状态推进。

论文同型号 Llama3.2-3B 的 GSM8K 为 77.56% → 90.96%，当前未复现该提升。不能再把差距解释成“评测没跑完”。本轮 Qwen 为 3B、论文为 7B，不能直接对齐其绝对分数。

## GSM8K 为什么净提升很小

全量重新使用固定官方函数评分，5000 条 Source 和 TLM 均与保存分数一致。TLM 3899 题正确，Source 3880 题正确；559 题由错变对，同时540题由对变错，最后净增19题。

350 条只输出 `Let's think step by step.` 就结束，Source 同类为0；其中258题原来 Source 能答对。这不是空字符串，也不是评分器把正确数字漏掉了。实际输入的结尾恰好包含这句开场白，随后是用户轮次结束符。

其他检查：学习率1e-6、batch1、一轮、5000条独立索引、阈值和权重公式均符合固定配方；Llama 5000次优化器更新全部成功，输入裁剪为0。31道错题达到512 token生成上限，即使全答对最多增加0.62个百分点，不能填平与论文的差距。

官方数字提取器还有格式限制，例如带空格的千分位、答案后的其他数字以及带逗号的参考值。仅对参考标签去逗号，TLM 为78.74%、Source为78.44%，依然没有论文中的大幅提升。这个值仅作诊断，原官方口径77.98% / 77.60%不改。

## 已保存检查点的变化

探针没有训练。先用最终权重精确重放第0、99条保存回答，再加载底模和1000 / 2500 / 5000步权重。固定100条问题，强制给出完整推理开场白后，观察下一步；这不是实际空答率。

| 训练步数 | 开场白后结束符排第一 | 结束符平均概率 |
|---|---:|---:|
| 0 | 0/100 | 0.59% |
| 1000 | 4/100 | 6.96% |
| 2500 | 14/100 | 18.30% |
| 5000 | 24/100 | 23.87% |

趋势支持“适应增强了说完开场白就停止的倾向”，还不能单凭它认定原因。每阶段另有4条固定生成，仅用于检查行为，不能选最高分检查点作为正式结果。

因此另建三组2500步对照：只屏蔽开场白、只屏蔽紧随其后的用户EOT、同时屏蔽两者。输入、原5000条采样器、5000步调度、1e-6学习率、Gaussian初始化和完整输入底模CE权重不变；保留项仍除以原目标数。原2500步权重作为保存的基线复用，不冒充独立重训。开始训练前先精确重放最终预测，再重新创建并逐张量确认相同Gaussian初始权重。

## Dolly 对照

200步基线不足以复现全量退化，所以延长到1000步。两模型的1000步原训练重放均与原记录逐值一致，step-100权重也完全相同。

以下是固定100条下一token探针及8条固定生成的诊断，8条分数不是全量ROUGE-Lsum。1000步Qwen的提前结束40/100，也不是最终5000步的空答率。

| 对照 | 首token为格式标题 | 首token为结束符 | 8条生成ROUGE×100 | 调度核查 |
|---|---:|---:|---:|---|
| llama3b__all | 84/100 | 0/100 | 32.19 | 逐次一致 |
| llama3b__role_masked | — | — | — | FAILED，尚未给出可解释结果 |
| llama3b__wrapper_masked | 0/100 | 0/100 | 37.23 | 逐次一致 |
| llama3b__both_masked | — | — | — | FAILED，尚未给出可解释结果 |
| qwen3b__all | 3/100 | 40/100 | 21.45 | 逐次一致 |
| qwen3b__role_masked | — | — | — | FAILED，尚未给出可解释结果 |
| qwen3b__wrapper_masked | — | — | — | ACTIVE，尚未给出可解释结果 |
| qwen3b__both_masked | — | — | — | ACTIVE，尚未给出可解释结果 |

只屏蔽数据自带格式词的 Llama 组已把 `###` 起始从84/100降至0/100，但8条分数仍低于配对 Source 的47.71，不能声称退化已经解决。

部分掩码组改变了FP16跳步时机，被早期“逐次LR必须一致”的检查拒绝，训练权重完整保留。例如原组在第12次访问跳步、role_masked在第33次访问跳步，两者都成功更新999次。新入口独立验证同一cosine定义下每次LR只能随真实优化器更新推进0或1步，并核对最终Adam与scheduler计数；能解释才补评测。它不改写 `learning_rates_exact=False` 或原失败记录，也不重训。梯度大小、裁剪和AMP差异仍作为干预的一部分记录，不能将改善都归为纯语义效应。

## 复现边界与代码

模板和真实结束符进入输入损失，也存在于固定官方处理器的 `labels=input_ids` 实现中；屏蔽目标是独立机制对照，不能静悄悄代替原文方法。论文写Gaussian，固定公开代码普通LoRA路径未显式指定Gaussian；本轮按论文显式选Gaussian。这些差异不能通过挑选更高分结果来掩盖。

已确认的停止符差异和标题空壳统计已提供独立检查代码。此前两组Dolly已扫描前缀按官方停止符解码，改变回答数为0；Source没有生成token ID，不能据单侧检查认证完整官方解码配对。原主实验评分保留。

- [模板损失位置诊断](../../../offline_tta/tlm_template_diagnostic_v1/README.md)
- [1000步对照调度](../../../offline_tta/tlm_template_diagnostic_runner_v1/README.md)
- [已训练GSM检查点探针](../../../offline_tta/tlm_gsm_milestone_probe_v1/README.md)
- [GSM开场白/EOT对照](../../../offline_tta/tlm_gsm_transition_control_v1/README.md)
- [保留AMP差异的评测补救](../../../offline_tta/tlm_template_evaluation_recovery_v1/README.md)
- [官方停止符与无实质回答检查](../../../offline_tta/tlm_reproduction_protocol_fix_v1/README.md)
- [配置、部署哈希与原始核查证据](tlm-dolly-gsm-diagnostic-evidence.json)

来源：[原文Table2/3及附录C](https://arxiv.org/html/2505.20633v1)、[官方输入目标](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/src/llamafactory/data/processor/self_supervised.py)、[官方LoRA构建](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/src/llamafactory/model/adapter.py)、[官方答案提取](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/scripts/eval/eval_utils.py)。
