# 大创项目：流式测试时自适应（OTTA）

本仓库用于三人团队共同维护项目资料、实验代码、配置与可复现实验结论。

## 仓库内容

- `项目总览.md`、`中期行动计划.md`、`项目看板.md`：项目目标、阶段安排和当前任务。
- `团队分工与协作.md`、`组会记录.md`：协作规则、负责人和过程记录。
- `文献资料/`：论文阅读总表、阅读模板与笔记。
- `方法设计/`、`实验方案/`、`实施计划/`、`阶段材料/`：研究和答辩材料。
- `src/`：实验实现；`tests/`：自动化检查；`configs/`：实验配置。
- `data/`：只保留小型固定样本与数据说明，不存放原始数据集。

## 环境与验证

本项目当前在 Windows、Python 3.11、RTX 4050 上验证。请始终使用项目虚拟环境，不要使用系统 Python（系统 Python 中的 PyTorch 可能是 CPU 版）。

```powershell
Set-Location -LiteralPath 'D:\大创\进度'
uv venv .venv --python 3.11
$projectPython = '.\.venv\Scripts\python.exe'
uv pip install --python $projectPython torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python $projectPython -r requirements.txt

& $projectPython -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
& $projectPython -m pytest -q
& $projectPython -m src.data.prepare_smoke_data --check --output data/smoke_32.jsonl --count 32
```

运行 No Adaptation smoke test：

```powershell
& $projectPython -m src.scripts.run_experiment --config configs/smoke.yaml --method no_adaptation
```

命令会在 `artifacts/runs/no_adaptation/42/<时间戳>/` 生成一份新的运行目录。请保留旧目录，不要覆盖历史结果；将新目录中的 `summary.json` 与 `environment.json` 登记到 `实验方案/实验记录.md`。

## 三人协作约定

1. 每项任务在 `项目看板.md` 或飞书任务中写明负责人、截止时间和验收产物。
2. 不直接在 `main` 分支开发；按任务新建分支，例如 `feature/baseline` 或 `docs/literature`。
3. 完成后提交 Pull Request，由另一位队友检查后合并。
4. 原始数据、模型权重和完整运行日志统一放共享云盘；仓库只提交数据说明、配置、代码和结果摘要。

详细规则见 [团队分工与协作.md](团队分工与协作.md)。
