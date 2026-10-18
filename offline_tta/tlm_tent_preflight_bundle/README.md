# Tent 文本工程适配：单步预检

用途：验证 Qwen2.5-7B-Instruct 在两张 RTX 2080 Ti 上能执行一次无标签生成熵更新，且只更新 q/v LoRA。
不是正式方法比较，不计算任务得分，不把预检 adapter 保存为后续实验起点。

## 本次配置

- 环境：`/data/chuwenjing/conda-envs/tta-llm`；torch 2.5.1+cu118、Transformers 4.46.1、PEFT 0.12.0。
- 模型：`/data/chuwenjing/models/Qwen2.5-7B-Instruct`，基础权重 FP16、只读。
- 数据：固定版本 GSM8K 第 0 条，只向适应函数传入 instruction/input。
- TLM qwen 模板；最多 1024 输入 token；greedy、seed=42、batch=1。
- 最多生成 80 token，固定生成前缀上重算全词表预测熵，排除提示位置、停止符、PAD 及其后续位置。
- LoRA：q_proj/v_proj、rank=8、alpha=16、dropout=0、A 高斯初始化、B=0，2,523,136 个 FP32 可训练参数。
- Adam：lr=1e-6、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，一步更新。
- 非重入梯度检查点，反向传播时禁用 KV cache；生成只在 no_grad 下进行。

TLM 附录 C.2 给出 80-token 熵适配，但作者 Tent 的可更新参数集合尚未从基线代码确认。
LoRA 参数化及未单独披露的超参数均为本工程明确记录的选择。因此名称为 Tent-text-LoRA engineering adaptation。
80 是适应前缀预算，与 Source/后续评测的 512 个生成 token 是不同参数。

## 1. 在本机 PowerShell 上传

```powershell
scp -r "D:\大创\进度\offline_tta\tlm_tent_preflight_bundle" chuwenjing@10.126.62.55:/data/chuwenjing/tta_offline/
```

这个包只含代码、配置、文档和校验和，不需要重新上传模型或安装依赖。

## 2. 在服务器终端运行

```bash
cd /data/chuwenjing/tta_offline/tlm_tent_preflight_bundle
nvidia-smi
```

选择两张空闲 GPU，以下以 0、1 为例。若有其他人在用，请替换卡号。

```bash
nohup bash run_tent_preflight.sh 0,1 >> run.log 2>&1 < /dev/null &
tail -n 30 run.log
```

只启动一次。日志采用追加方式，重复启动会被锁拦截，GPU 被占用也会停止。
脚本先校验文件，再把运行副本复制到 home 的项目目录，激活已有项目环境，运行 CPU 测试，然后加载真实模型。
不在 base 安装包，不修改上游源码，不保存或覆盖基础模型。SSH 断开后 nohup 任务继续运行。

## 3. 看结果

只有日志出现 `TENT_SINGLE_STEP=PASS` 才表示所有真实模型检查通过：

- 损失及梯度有限，两张 GPU 的 adapter 都收到非零的汇总梯度。
- 至少部分 LoRA 元素实际更新；零初始化 B 时，第一步部分 A 不变化属于预期。
- 更新前后全部基础参数的流式 SHA256 摘要相同，基础参数没有梯度。
- Adapter 精确恢复初值，Adam 状态已清空；预检状态不延续到正式实验。

损失下降不等于任务得分提高，单步通过也不等于论文复现完成。

`OUTPUT_DIR` 行显示实际输出目录：
`/data/chuwenjing/results/tent_single_step/qwen25-7b/上海时区时间戳/`。
其中保存 `run_manifest.json`、`events.jsonl`、`trainable_parameters.json`、`sampled_prefix.json`、`step_report.json`。
异常时 manifest 标记 FAILED，并保留 `failure.txt`，程序非零退出；不自动改模型、量化、改学习率或跳过检查。
校验全部基础权重会短暂传输约一份权重的数据量到 CPU，按小块处理，不复制整份模型占用额外显存。

代码运行目录：`/home/chuwenjing/TTA_LLM/TLM/local_repro/tent_preflight`。
若该目录已有不同版本文件，脚本会因 cmp 检查停止，避免覆盖旧代码；此时保留日志再排查。

## 验证状态

本机 CPU 数学、掩码、答案隔离、小随机 Llama/PEFT 更新、重置、异常路径和梯度检查点测试已通过。
兼容性测试使用 Transformers 4.46.1、PEFT 0.12.0；本机 PyTorch 为 2.14.0，与服务器的 2.5.1 不同。
服务器启动时会使用项目的实际版本重新运行整套 CPU 测试，然后执行真实 Qwen 双卡验证。
截至交付，真实 Qwen 单步预检尚未运行，不宣称 GPU PASS。

正式 offline Tent 小样本对照将另行实现：每个数据集先用其无标签输入更新，再统一评测，数据集之间重置 adapter 和 optimizer。
