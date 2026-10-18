# Tent-text-LoRA offline 小样本对照

用户在已通过真实 Qwen 单步验证后回复“继续”，确认进行此前提出的三个数据集各 16 条 offline 对照。沿用已确认的生成熵 + q/v LoRA 工程适配；不声称作者 Tent 参数集合已完全查明。

## 固定协议

- 模型：Qwen2.5-7B-Instruct；固定 TLM 源码提交 496c9ed97d0fb8bc4854ed2479259e7e9e70a790；FP16 基础权重、FP32 q/v LoRA，r=8、alpha=16、dropout=0。
- 数据：GeoSignal、Dolly、GSM8K 各原顺序前 16 条。每个数据集用新的子进程重新加载原始模型并初始化 adapter/Adam，seed=42；不跨数据集继承状态。
- 设置：offline。对某数据集先依次处理完 16 条无标签输入，每条最多生成 80 token 并执行一次生成熵更新，然后统一对同样 16 条输入生成回答、评分。没有独立有标签训练集；这是无标签 transductive 测试时适应。
- 学习率提前固定：GeoSignal/Dolly 为 5e-5，GSM8K 为 1e-6，参考 TLM 对三个 benchmark 的主文设置，不冒充已确认的作者 Tent 专用参数。不根据这 48 条的参考答案调参。
- 优化器、熵定义、因果掩码、停止符规则、梯度检查点、冻结参数检查均复用已通过真实 GPU 验证的单步核心，按字节复制而不修改原包。
- 评测：greedy，输入上限 1024，输出上限 512，batch=1，TLM qwen 模板、相同 ROUGE-Lsum 与原仓库 GSM8K 答案提取器。原始答案只在评测函数中读取。
- Source 比较固定使用 `/data/chuwenjing/results/source_smoke/qwen25-7b/20261018_110656_386331`，不自动寻找最新结果；校验配置、文件 SHA256、索引、提示指纹、参考答案、统计一致性。
- 每个数据集开始更新前，零初始化 LoRA 对第 0 条生成 512-token 回答，要求与已存 Source 文本一致；若不一致则停止，避免比较不同推理条件。该探针不计入适应步数，不做分数选择。
- 适应函数不接收答案；评测必须等全部适应输入处理完才运行。记录每一步前缀、熵、梯度、变化量、更新索引、评测预测和逐条指标。
- 全部基础参数的流式摘要在每个数据集更新前后保持一致；评测开始前清空梯度，关闭梯度检查点并进入 eval 模式。异常、非有限值、空有效前缀或无更新均停止并保留失败日志，不静默跳过或自动改超参数。
- 结果和仅含 LoRA 的 checkpoint 放在 `/data/chuwenjing/results/tent_offline_smoke/qwen25-7b/独立时间戳/数据集/`。不覆盖原始模型、Source、预检结果或旧代码。

## 架构与验收

`offline_flow.py` 提供可测试的阶段顺序、标签隔离及基线验证；`run_tent_dataset.py` 执行单个数据集；`run_tent_smoke.py` 在三个独立子进程上调度并汇总差值。
CPU 测试使用无输出字段访问权的输入记录验证更新先于评测、禁止中途评分、全部索引正确；使用伪造基线文件验证配置/提示/指标不一致会被拒绝。适应数学继续由原包的小随机 PEFT 模型测试覆盖。
shell 包装器检查校验和、重复启动锁、项目环境和 CPU 测试；每个子进程启动前重新检查 GPU 占用。

仅当三个数据集均完成 16 次更新、16 次评测，基础权重未变，比较配置匹配且所有分数有限，输出 `TENT_OFFLINE_SMOKE=PASS`。
性能低于 Source 仍是合法结果。此标记只表示 48 条工程流程完成，不表示方法提高性能或完成论文全量复现。

本地离线包为 `D:\大创\进度\offline_tta\tlm_tent_offline_smoke_bundle`，运行代码为 `/home/chuwenjing/TTA_LLM/TLM/local_repro/tent_offline_smoke`。不自动上传、不在 base 安装依赖。
