# Tent-text-LoRA：3×16 条 offline 对照

本轮按 GeoSignal、Dolly、GSM8K 顺序串行运行，各 16 条。同一数据集先完成 16 次无标签更新，再统一生成 16 条回答评分。
每个数据集由独立 Python 子进程重新加载原模型，并重新初始化 LoRA/Adam。不是逐条更新后立即评分的 online 设置。

## 固定设置与结果含义

- 基础模型为 `/data/chuwenjing/models/Qwen2.5-7B-Instruct`，FP16；q/v LoRA 为 FP32，rank=8、alpha=16、dropout=0。
- **适应生成上限 80 token；评测生成上限 512 token。** 输入上限为 1024，batch=1、greedy、seed=42。
- Adam 学习率提前固定：GeoSignal/Dolly 为 5e-5，GSM8K 为 1e-6。参考 TLM 主文的 benchmark 设置，不是已经查明的作者 Tent 专用配置；不根据本轮答案调参。
- 用生成前缀的全词表预测熵做更新，适应函数只接收 instruction/input。提示位置、停止符/PAD 及之后的位置排除。
- GeoSignal/Dolly 用 ROUGE-Lsum，GSM8K 沿用原仓库数字提取 + Exact Match。保持和 Source 完全相同的主指标口径。
- Source 固定为已完成的 512-token 运行 `/data/chuwenjing/results/source_smoke/qwen25-7b/20261018_110656_386331`，校验 5 个文件的 SHA256 和样本/配置匹配。
- 每个数据集在更新前对第 0 条做一次零 LoRA 的 Source 一致性探针，输出必须匹配。此探针不更新参数。
- 相同 16 条输入既用于无标签适应又用于后续评分，这是本轮 offline transductive 设置；参考答案不参与更新或超参数选择。

本轮仍称 Tent-text-LoRA 工程适配，作者 Tent 的参数集合尚未完整核实。[TLM 主文与附录 C.2](https://arxiv.org/html/2505.20633v1#A3.SS2)
分数降低也是合法实验结果；PASS 仅表示流程和校验通过，不表示效果提升或论文全量复现完成。

## 1. 本机 PowerShell 上传

```powershell
scp -r "D:\大创\进度\offline_tta\tlm_tent_offline_smoke_bundle" chuwenjing@10.126.62.55:/data/chuwenjing/tta_offline/
```

只上传这个新代码包。模型、数据、环境均复用已有文件；不会覆盖预检代码包和 Source 结果。

## 2. 服务器启动

```bash
cd /data/chuwenjing/tta_offline/tlm_tent_offline_smoke_bundle
nvidia-smi
```

选择两张空闲 GPU。以下以 0、1 为例：

```bash
nohup bash run_tent_smoke.sh 0,1 >> run.log 2>&1 < /dev/null &
tail -n 30 run.log
```

只启动一次。三个数据集自动依次运行，不需要分别启动三次。SSH 断开后 nohup 任务继续。
新包锁防止重复启动；每个数据集启动前重新检查 GPU。脚本会等待最多约 10 秒让前一子进程的 NVML 状态刷新，仍被占用就停止。
运行副本复制到 `/home/chuwenjing/TTA_LLM/TLM/local_repro/tent_offline_smoke`；存在不同版本时停止，不覆盖。
先激活已有项目环境并跑 CPU 测试，然后检查实际 Source 对照和启动模型。无需 pip 安装，不修改 base。

## 3. 看进度和结果

日志中的关键阶段：

- `PINNED_SOURCE_COMPARISON=PASS`：基线配置、样本和文件校验通过。
- `source_probe_matches`：零 LoRA 第 0 条回答与 Source 相同。
- `adapted`：已完成多少次更新；只到 16 次之后才出现 `evaluated`。
- `TENT_DATASET=PASS`：一个数据集的 16 次更新、16 次评测完成。
- **`TENT_OFFLINE_SMOKE=PASS`**：三个数据集全部完成，比较文件已保存。

```bash
tail -n 40 run.log
```

`OUTPUT_DIR` 行给出本轮输出目录：
`/data/chuwenjing/results/tent_offline_smoke/qwen25-7b/上海时区时间戳/`。

根目录有 `comparison.json`、`comparison.md`、`source_summary.json` 和总 `run_manifest.json`。
每个数据集子目录有 `adaptation.jsonl`、`predictions.jsonl`、`metrics.json`、`source_probe.json`、参数清单、事件日志、manifest，及仅含 LoRA 的 `adapter/` checkpoint。
所有结果和 checkpoint 均在 `/data/chuwenjing`，不写回基础模型目录。

遇到非有限数、空有效生成前缀、无参数更新、基础权重变化或 Source 不匹配时停止，保留 `failure.txt` 与已经生成的日志。
不自动跳过样本、改变学习率、重试选最好分数或在失败后打印 PASS。保留日志后排查，不反复启动。

## 已验证与待验证

已通过本机 CPU 阶段顺序/标签隔离/基线拒绝测试，以及原有熵、掩码、LoRA 更新和梯度检查点测试。
兼容性测试使用服务器版本 Transformers 4.46.1、PEFT 0.12.0；本机 PyTorch 2.14.0 与服务器 2.5.1 不同，服务器启动时会重跑 CPU 测试。
复用的单步核心已在真实 Qwen 双 RTX 2080 Ti 上通过。该新 48 条实验在交付时尚未执行，不预报其分数或宣称 GPU PASS。
