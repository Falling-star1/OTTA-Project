# 已交付源码包

当前训练固定使用 [TLM v6](tlm_offline_v6_bundle/README.md)，新建六组运行使用 [控制器 v2](tlm_full_run_gate_v2/README.md)。四组全量评测继续，Llama Dolly 质量报警停评，Qwen Dolly 由 [单组重新验证](tlm_single_recovery/README.md) 补齐。见 [当前方案](../docs/TLM复现实验方案.md)。

以下为 TENT 交付时的历史入口。

当前正式入口：[`tlm_tent_full_3gpu_det_bundle`](tlm_tent_full_3gpu_det_bundle/README.md)。三组真实 GPU 长输入和精确恢复预检已通过；全量启动与完成状态见[实验记录](../docs/实验记录.md)。

其余 stage1、source_smoke、tent_preflight、tent_offline_smoke 包用于追溯已完成验证。包内 README、DESIGN、PROTOCOL 是各交付时刻的冻结文档，其中历史待办不表示当前状态。

全部包保留原字节、目录名与 SHA256SUMS，便于与服务器 cmp/哈希核对。修改代码需另立新包版本并重新验证。模型、数据、依赖 wheel、Conda 安装包和当前服务器收据不在此提交。

运行和部署前先读[环境与运行](../docs/环境与运行.md)。
