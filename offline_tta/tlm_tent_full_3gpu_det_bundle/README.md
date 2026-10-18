# Tent：Qwen 三卡确定性计算版本：全量实验及长输入预检

这是 Tent-text-LoRA 的文本任务工程复现。本次修订针对默认GPU注意力反向算子重复执行不完全一致的问题，补充严格确定性计算设置；保留逐元素精确比较，不使用数值容差绕过验收。旧双卡版本在 GeoSignal 的4096-token长输入反向传播时GPU1显存不足，三卡分摊模型保持不变；训练目标、输入上限、数据、学习率与可训练参数保持原设置。先完成长输入预检，再运行 GeoSignal、Dolly、GSM8K 各 5000 条的 Source/Tent 对照。此前的 3×16 实验只证明流程可运行，不能代替全量结果。

使用已安装的 Python3.10、torch2.5.1+cu118、Transformers4.46.1、PEFT0.12.0，无需安装新包。每次启动先验证包内 SHA256，再在项目环境运行 CPU 测试。代码复制到 `/home/chuwenjing/TTA_LLM/TLM/local_repro/tent_full_3gpu_det`；已存在且内容不同的文件会停止安装。结果、adapter、Adam/RNG 检查点在 `/data/chuwenjing/results`。基础模型与旧实验包保持原样。

## 1. 本地 Windows PowerShell 上传

```powershell
scp -r "D:\大创\进度\offline_tta\tlm_tent_full_3gpu_det_bundle" chuwenjing@10.126.62.55:/data/chuwenjing/tta_offline/
```

## 2. 服务器先跑长输入预检

```bash
cd /data/chuwenjing/tta_offline/tlm_tent_full_3gpu_det_bundle
nvidia-smi
nohup bash run_tent_full.sh preflight 0,1,2 >> preflight.log 2>&1 < /dev/null &
tail -n 30 preflight.log
```

使用三张空闲 GPU；如 0、1、2 被其他人占用，修改为另外三个物理编号。空闲检查会拒绝占用中的 GPU。`nohup` 允许关闭 SSH 后继续，脚本锁防止同一包重复启动。

三组各取最长输入（原始索引 3396、1404、1445），使用和全量相同的输入上限4096、Source生成512、适应生成80。预检验证真实三卡梯度、参数变化、冻结基础权重以及 adapter/Adam/Python/NumPy/CPU/CUDA 随机状态恢复；恢复后下一步必须与不中断路径完全一致。

所有预检通过后输出 `TENT_FULL_PREFLIGHT=PASS`，并生成绑定代码、配置与报告哈希的 `preflight_receipt.json`。只看到 CPU 测试通过不代表 GPU 预检通过。预检失败请保留日志，不能直接降低门槛或启动全量；重试预检会使用新的结果目录。

## 3. 预检通过后启动全量

```bash
cd /data/chuwenjing/tta_offline/tlm_tent_full_3gpu_det_bundle
nohup bash run_tent_full.sh run 0,1,2 >> full.log 2>&1 < /dev/null &
tail -n 30 full.log
```

记下开头的 `OUTPUT_DIR=/data/chuwenjing/results/tent_full_3gpu_det/qwen25-7b/...`。每个数据集使用新进程、原模型和同种子新 LoRA/Adam，依次完成：5000条 Source 预测 → 全部5000条无标签适应 → 5000条 Tent 预测。三组总计30000次最终预测、15000次适应生成及更新；耗时需根据服务器实际速度估计。

每10条在控制台报告进度，每25条提交训练检查点。最终应看到 `TENT_FULL_RUN=PASS`；对照表在输出根目录的 `comparison.md`、`comparison.json`。每组保存 `source.jsonl`、`adaptation.jsonl`、`tent.jsonl`、`metrics.json` 和 `adapter/`。

## 4. 中断后的恢复

只有旧进程已经停止后才执行恢复。将下面最后一个参数替换为日志里**完整的原 OUTPUT_DIR**，不能填父目录或新建目录。

```bash
cd /data/chuwenjing/tta_offline/tlm_tent_full_3gpu_det_bundle
nohup bash run_tent_full.sh resume 0,1,2 /data/chuwenjing/results/tent_full_3gpu_det/qwen25-7b/替换为原运行目录 >> full.log 2>&1 < /dev/null &
tail -n 30 full.log
```

恢复会验证协议、模型和文件身份，复用已经写好的 Source/最终预测，加载 adapter、Adam、随机状态，从已提交的 `next_index` 继续。硬中断后，检查点之后最多24条未提交更新会从旧状态重新计算，不在已经更新过的状态上重复累加。正常 SIGTERM/SIGINT 在当前样本边界提交退出；如需暂停整个任务，对根清单 `run_manifest.json` 中的 `pid` 使用 `kill -TERM PID`，不要猜 PID。首次模型加载尚未完成时需等待其完成后才能响应暂停。

Source/评测逐条持久化，只有末尾不完整的一行可自动修复；中间文件损坏、配置变更、错误模型、缺失梯度或非有限值均停止，不静默跳过。预检要求实际更新；全量训练中的有限零变化允许作为收敛状态记录。

## 固定口径与边界

- Qwen2.5-7B-Instruct，FP16基础权重；q/v LoRA r8、alpha16、dropout0，FP32可训练参数2523136。GeoSignal/Dolly学习率5e-5，GSM8K为1e-6，seed42，batch1，greedy。
- 输入上限4096，超过上限的 GeoSignal 2条、Dolly 1条使用固定尾部截断并记录。Source与Tent完全同口径；旧的1024上限 Source 不能混作本次对照。
- 适应使用最多80个自生成 token 的全词表预测熵，不接收参考答案；评测最多生成512。后缀 logits 优化仅减少长 prompt 的词表投影，CPU 测试对照完整 logits 的损失、梯度和一步更新。
- GeoSignal/Dolly按固定 TLM 评分流程算 ROUGE-Lsum，GSM8K使用原提取器及 exact match；输出记录生成上限命中、空输出和输入截断。
- r8 q/v LoRA和当前学习率是已公开的工程设置；作者对 Tent 的全部参数细节尚未完全核实，不能称完全逐项一致。分数下降也须如实记录，PASS只表示运行及一致性校验通过。
- 本包完成 Qwen 路线；Llama-3.2-3B-Instruct 的访问权限、模型和同口径实验仍待补齐。接下来继续完善 Tent，不跳到 EATA/SAR/DeYO。

硬件修订说明：新包使用独立的代码、预检收据和结果目录。不要覆盖旧包，不要对旧双卡或旧三卡的失败预检目录执行 resume；请在本包重新运行 preflight。Source和Tent都必须在本次三卡配置下重新运行，旧小样本得分不能代替新对照。三卡版本仍需真实GPU预检确认显存及恢复一致性；CPU测试通过不等于OOM已解决。

## 确定性修订的依据与验收

旧三卡预检已经完成首步真实更新，熵0.573185→0.440132；恢复后下一步的参数精确比较失败。检查点SHA、112组Adam状态及三卡RNG完整。服务器上固定输入SDPA诊断显示：默认FP16高效注意力反向重复3次，Q梯度每次约4万个元素不同；严格确定性模式下连续3次全部相等。该诊断只证明算子重复性，完整模型恢复仍需本包GPU预检。

启动脚本在Python启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；Runtime在模型加载/CUDA计算前调用 `torch.use_deterministic_algorithms(True, warn_only=False)`，清单记录实际开关。若算子不支持确定性执行，将直接报错，不能用warn_only绕过。确定性内核可能更慢，不改变熵损失、学习率、LoRA、数据及4096/80/512口径。

预检新增保存点的adapter/Adam恢复后立即精确比较，并继续验证RNG、贪心token和下一步adapter/Adam逐元素相等。任何不一致仍停止；错误将包含张量路径、不同元素数及最大绝对差。

实现依据：[PyTorch2.5.1高效注意力反向的确定性分支](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/transformers/cuda/attention_backward.cu#L613-L624)、[PyTorch2.5.1可复现性说明](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/randomness.rst)。本版本仍需完整三卡预检，不能因为CPU测试或算子诊断通过就启动正式全量。
