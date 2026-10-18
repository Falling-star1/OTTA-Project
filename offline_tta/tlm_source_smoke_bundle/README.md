# Qwen Source 小样本基线

目的：检查三套数据的读取、生成、结果保存和主指标评测流程。
Geography、Dolly、GSM8K 各用官方固定版本前 16 条，共 48 条。原始权重冻结，不创建优化器，不进行参数更新。
复用 TLM 的 Qwen 提示模板和模型加载器；直接调用 model.generate。
原因：固定源码提交中的 hf_engine.py 按旧参数调用多模态插件，与当前 BasePlugin 的 audios 参数不匹配。
本脚本面向纯文本任务，无须更改上游仓库、依赖版本或模型权重。

配置：两张 RTX 2080 Ti、FP16、batch size 1、greedy、seed 42、最多 1024 个输入 token 和 128 个生成 token。
过长输入保留用户文本前缀并重新套用完整 TLM 模板，逐条记录截断和生成达到上限的情况。
这些限制用于调试，结果不能当作论文的完整复现成绩；正式实验需另行冻结完整配置。

Geography/Dolly 用 ROUGE-Lsum，处理方式对应 TLM 的 scripts/eval/eval_similarity.py；
GSM8K 调用原仓库 extract_gsm8k_answer_number，以与参考答案相等作为 Exact Match。
参考答案只在生成之后用于评测，不传入生成提示。

## 上传并启动

本机 PowerShell：

```powershell
scp -r "D:\大创\进度\offline_tta\tlm_source_smoke_bundle" chuwenjing@10.126.62.55:/data/chuwenjing/tta_offline/
```

服务器终端先查看 nvidia-smi，选择两张空闲卡。以 0、1 为例：

```bash
cd /data/chuwenjing/tta_offline/tlm_source_smoke_bundle
nohup bash run_source_smoke.sh 0,1 > run.log 2>&1 < /dev/null &
tail -n 40 run.log
```

不要重复启动。脚本会在卡已被占用时停止。SSH 断开后 nohup 任务仍会运行。
看到 SOURCE_SMOKE=PASS 才表示完整 48 条流程通过；遇到错误请保留日志和已生成的部分结果。

代码复制到 /home/chuwenjing/TTA_LLM/TLM/local_repro/source_smoke。
每次运行输出独立目录：/data/chuwenjing/results/source_smoke/qwen25-7b/上海时区时间戳。
run.log 中 OUTPUT_DIR 行给出实际结果目录。
目录中含 3 个预测 JSONL、metrics.json 和 run_manifest.json。不会覆盖以前的运行结果。

截至打包时：正式 Qwen 已完成双卡简短生成预检；本 48 条运行尚待用户上传后执行。
Tent、EATA、SAR、DeYO 及 proxy pool 仍是后续任务。
