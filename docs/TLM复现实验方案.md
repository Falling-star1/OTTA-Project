# TLM 复现实验方案

更新：2026-10-26 08:10（北京时间）。**v6 六组完整预检全部通过，六组各 100 条验证正在并行执行。全部通过后自动开始全量；当前尚不能报告全量已启动。** [本次核查与目录](records/20261026/TLM_v6预检与全量准备.md)记录了代码身份、检查结果和运行路径。

旧 v1/v2 六组已停止，权重、预测、检查点和日志保留，状态文件也已改成 STOPPED。已完成的 TENT 结果继续作为对照。

## 1. 这轮要完成什么

在两种 3B 模型、三个数据集上补齐 TLM。每格都是独立实验：从原模型新建 Gaussian LoRA，用 5000 条问题适应一遍，再回答同样 5000 条问题。

| 模型 | GeoSignal | Dolly | GSM8K |
|---|---:|---:|---:|
| Llama-3.2-3B-Instruct | 5000 条 | 5000 条 | 5000 条 |
| Qwen2.5-3B-Instruct | 5000 条 | 5000 条 | 5000 条 |

适应只读问题和上下文，参考答案留到评分时使用。不同数据集之间不继承适应权重。旧 Kaiming 结果单独记录，不能改名并入这轮 Gaussian 主实验。

本轮是 **3B / FP16 条件下的方法复现**。论文 Qwen 主结果使用 7B，官方示例使用 BF16，和当前 2080 Ti 的条件不同；结果中会写明这些差异。

## 2. TLM 怎么更新模型

对一条问题，依次做四件事：

1. 套用模型的固定提示模板，得到输入 token。
2. 关闭 LoRA，用冻结的底模计算输入平均交叉熵 `c0`。
3. 只选中 `c0 > 3` 的样本，权重为 `0.1 × exp(c0 - 3)`；`c0` 和权重不参与反向传播。
4. 打开 LoRA，计算当前模型在同一输入上的平均交叉熵，乘上权重并更新 LoRA。

前一个位置预测下一个输入 token，排除 padding；真实 EOS 不因 pad 与 EOS 同 ID 而被删掉。未选中的样本保留零损失图和固定源码中的优化器调用；已有 Adam 动量仍可能推动参数变化。

损失和 offline 流程核对官方提交 `496c9ed97d0fb8bc4854ed2479259e7e9e70a790`。初始化遵照论文，显式使用 Gaussian A、零 B。可核对[论文](https://arxiv.org/html/2505.20633v1)、[官方配置](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/examples/train_lora/offline_ttl.yaml)和[官方损失实现](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/src/llamafactory/train/ttl/trainer.py)。

## 3. 固定设置

| 项目 | 本轮设置 |
|---|---|
| 数据 | 固定版本的三个数据集，各 5000 条，核验文件哈希 |
| LoRA | q_proj、v_proj；rank=8、alpha=16、dropout=0 |
| 初始化 | Gaussian A、零 B；seed=42 |
| 数值类型 | 冻结底模 FP16，LoRA 参数和 CE 为 FP32；实际 Trainer 使用 AMP 与 GradScaler |
| 输入 | 最多 4096 token；裁剪用户内容尾部后重建完整模板，不读取答案决定截断 |
| 优化器 | AdamW；betas=(0.9, 0.999)、eps=1e-8、weight decay=0 |
| 学习率 | Geo/Dolly：5e-5；GSM8K：1e-6 |
| 调度 | cosine；warmup 前 10%；梯度裁剪 1.0 |
| 训练 | 一轮，batch=1，梯度累积=1；Trainer 按 seed=42 采样 |
| 生成 | 固定模板与停止符，greedy，最多 512 个新 token |
| 指标 | Geo/Dolly：ROUGE-Lsum；GSM8K：固定官方答案提取器和 Exact Match |
| 保存 | 完整检查点每 25 步一次、保留最近两个；另存 100/500/1000/2500/5000 步 adapter |

模型与哈希见 [profiles.json](../offline_tta/tlm_offline_v6_bundle/profiles.json)，数据与哈希见 [datasets.json](../offline_tta/tlm_offline_v6_bundle/datasets.json)。Torch 2.5.1+cu118、Transformers 4.46.1、PEFT 0.12.0、Accelerate 1.0.1 等版本由运行时核验。

## 4. 为什么有 v6

2080 Ti 显存有限。之前把输出层矩阵乘法分块，真实 GPU 反向梯度与整块实现不完全一致。v6 保留一次完整的 FP16 输出层计算，交叉熵分块计算，完整 FP16 logits 暂存到 CPU，反向时逐块取回。原始输出层矩阵形状、因果目标、TLM 权重和学习率保持一致。

v6 的六组真实预检已通过：零 B 和非零 B、不同长度共 32 组比较，LoRA 梯度及输出层反向梯度逐值一致；六组断点恢复一致；最长输入都完成 16 步实际混合精度训练，4096 条件下分配显存峰值约 8.0 GiB。12 份最终检查点共 120 个文件的哈希已独立核验。

这些检查不等于 5000 条成绩已经出来。新版本仍需完成六组各 100 条验证，才能进入全量。详细证据见 [真实 GPU 预检记录](records/20261026/tlm-v6-real-preflight.json)。

## 5. 当前怎么运行

训练包是 [tlm_offline_v6_bundle](../offline_tta/tlm_offline_v6_bundle/README.md)，统一控制器是 [tlm_full_run_gate](../offline_tta/tlm_full_run_gate/README.md)。冻结代码身份：

```text
b6da067b55f5c65241ae6fbd8c706b88523f64b27dd9568ad5661255b89d7999
```

顺序固定为：

```text
六组完整预检全部通过
  → 六组各 100 条适应、100 条生成全部通过
  → 一次核对所有代码、回执、实际训练配置和质量结果
  → 从底模重新初始化六份 Gaussian LoRA
  → 六组各 5000 条适应，再各 5000 条生成和评分
```

六组分别使用 GPU 0—5，顺序为 Llama Geo/Dolly/GSM、Qwen Geo/Dolly/GSM。所有组通过同一阶段后才进入下一阶段；一组未通过就阻止进入全量。100 条试跑权重不接入正式训练。

当前控制器已运行，不要重复启动。查看：

```bash
TLM_RUN=/data/chuwenjing/results/tlm_offline_v6_ready/run_20261026_000443_2ca89d
cat "$TLM_RUN/status.json"
cat "$TLM_RUN"/*.status.json
nvidia-smi
```

`pilot` 表示百条验证，`formal` 才是全量。`ready_for_formal.json` 是启动前六组检查的记录；还要确认 formal 任务确实启动，不能只凭文件存在报告全量在跑。

要看单组日志，例如 Qwen Dolly：

```bash
tail -n 30 "$TLM_RUN/qwen3b__dolly_5k.pilot.log"
```

生成期间会写 `tlm.jsonl`，其中每行是一条最终回答。适应结束会写 `training_complete.json`，含实际成功与跳过的优化器步数。适应期间看日志和最新已提交的 `trainer/checkpoint-*`；`run_manifest.json` 的 `initializing` 并不一定表示模型仍在加载。

新环境接手时，按包内说明先核验源码、输入与依赖，再做预检。旧包 `launch.py --mode all` 保留了早期 16 条试跑逻辑，不作为本轮入口。

## 6. 怎么判断结果能用

每组要有完整的 5000 条适应记录和 5000 条连续索引的回答；训练值、adapter 和 optimizer 有限，底模哈希不变，评分文件能从已保存回答重新计算。最终汇总同时给出 Source、TENT、TLM 和差值，负结果照实保留。

复用的 Source 来自已完成的同模型 TENT 运行。六份 Source 均有完整 5000 条，预检生成样例一致，输入身份、原始提示哈希和使用 token 数会继续核对。只复用底模 Source，不复用 TENT 适应权重。另外用 CPU 逐条核对了六组共 30000 条输入：token ID 与冻结的 TENT 编码器相同，所有 Source 提示元信息一致。见 [完整输入核对](records/20261026/tlm-v6-source-input-audit.json)。

生成每满 100 条，和配对 Source 检查空答及明显循环重复。当前工程报警规则是空答/重复超过 Source 5 个百分点，或同一非空答案占一半以上且超过 Source 10 个百分点。报警后保留权重、预测和日志，停止该组评测，不删坏样本或自动改参数重跑。

FP16 初始缩放过大时，GradScaler 可能跳过更新；因此记录 global_step 之外的实际成功步数。出现非有限 adapter/optimizer、长期没有有效更新、输入或代码身份变化，都会报错。单条日志里的初期 grad_norm 非有限值，须结合 GradScaler 回退和最终有限性检查判断。

## 7. 时间和论文差异

此前 Qwen 的 100 条生成约 36 分钟。新六组并行验证预计约 40—90 分钟，全部通过后自动进入全量。5000 条生成预计约 1—3 天，主要取决于生成长度；运行后按实际速度更新。

需要在结果里保留四点说明：Qwen 使用 3B；2080 Ti 使用 FP16；输入截断不受答案长度影响；CPU 暂存 logits 是工程省显存措施。已有 TENT 与本轮 TLM 的训练配方也有差异，分数变化不能全部归因于损失函数。

短验证检查是否能稳定运行，不保证全量会提高分数。此前 Qwen v4 百条结果为 TLM 27.05、Source 28.01，未触发退化报警；本轮 Gaussian 初始化和论文学习率不按这次分数取舍。

## 历史记录

- [旧运行核查与论文差异](records/20261025/TLM运行核查与风险_20261025.md)
- [Gaussian 初始化与 v3 验证](records/20261025/TLM修复选择与v3验证.md)
- [完整梯度诊断](records/20261025/TLM完整梯度诊断结果_20261025.md)
- [实际 GradScaler 与 16 步更新](records/20261025/TLM缩放与16步更新结果_20261025.md)
- [v4 预检及 Qwen 百条入口](records/20261026/TLM_v4预检结果与Qwen试跑.md)
- [v5 候选的当时状态](records/20261026/TLM_v5候选与Qwen试跑进度.md)
- [v6 六组实测与统一启动](records/20261026/TLM_v6预检与全量准备.md)

历史文件保留当时的结论，当前状态以本文开头和服务器运行文件为准。
