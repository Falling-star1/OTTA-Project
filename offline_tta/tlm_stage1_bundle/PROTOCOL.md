# 本轮实验范围与准备状态

目标模型：meta-llama/Llama-3.2-3B-Instruct、Qwen/Qwen2.5-7B-Instruct。
目标方法：Tent、EATA、SAR、DeYO；另保留不更新参数的 Source 基线。
四种方法的文本适配尚未实现。完成四种方法的比较后，参考 OPUS，基于同一批数据构造 proxy pool。

| 类别 | 本轮选择 | TLM 数据别名 | 正式样本数 | 主指标 |
|---|---|---|---:|---|
| DomainBench | Geography / GeoSignal | geosignal_5k | 5000 | ROUGE-Lsum |
| InstructionBench | Dolly | dolly_5k | 5000 | ROUGE-Lsum |
| ReasoningBench | GSM8K | gsm8k_5k | 5000 | Exact Match |

模型与数据范围来自用户指定的文档。三个具体数据集由本轮选择。
主指标对应 [TLM 论文 §5.1](https://arxiv.org/html/2505.20633v1#S5.SS1)。
BLEURT、BERTScore 等为后续扩展评测，不作为本轮主指标流程的启动条件。

源码固定提交 496c9ed97d0fb8bc4854ed2479259e7e9e70a790。
三份 JSON 均从该提交的 data/AdaptEval 复制原始字节，核对 SHA256，不替换成其他 GSM8K 或 Dolly 发布版本。
正式数据保持源码给定的全部 5000 条及原顺序。每份额外生成前 16 条调试样本，保留原始索引。
16 条版本仅验证读取、生成、输出和评测流程，其结果不用于论文结论或完整复现比较。

代码目录：/home/chuwenjing/TTA_LLM/TLM/local_repro/stage1。
数据目录：/data/chuwenjing/datasets/AdaptEval/496c9ed97d0f。
NLTK 缓存：/data/chuwenjing/cache/nltk_data。
后续模型、日志、预测、checkpoint 分别放到 /data/chuwenjing 下的专用目录。

正式方法实现之前需写清：采用 offline 或 online 设置、token 级损失及其掩码、可更新参数、
阈值与归一化、优化器/学习率、模型和优化器重置、样本顺序及随机种子。
测试参考答案仅用于评测，不能进入测试时适应损失或调参选择。Proxy pool 的划分和标签访问也需单独记录。

论文 Appendix C.2 为 Tent、EATA 给出了基于 80 个生成 token 的熵的文本适配描述；
SAR、DeYO 的文本适配须另对照各自原论文和官方实现，明确保留项及改动项。
不能把替换模型名称、降低熵或安装依赖本身记为四种方法复现完成。

服务器 GPU 为 RTX 2080 Ti，使用 FP16。Qwen 7B 的完整 FP16 权重不能放入单卡 11 GiB，
后续需要验证多卡放置方案。8 张卡不会自动合并显存，DDP 也不会自动拆分单份模型权重。

Llama 模型尚待用户通过官方页面申请访问；Qwen 权重尚未下载。正式模型加载和基线尚未验证。
