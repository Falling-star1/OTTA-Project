# 历史归档

归档日期：2026-10-18。旧 OTTA 自研计划、旧环境与任务安排停止作为当前执行要求；本目录仅供回溯。归档不会将 GitHub 仓库设为只读。

| 归档 | 来源 | 内容 |
|---|---|---|
| `github-main-20261018/` | GitHub main：`807856c02eaaaf497134a69f8c26d677a24a3473` | 原主线所有 Git 跟踪文件，包含旧代码、配置、文献、计划与过程记录 |
| `fix-label-isolation-ec8dae8.zip` | 已有修复分支：`ec8dae8` | 修复分支完整工作树快照，保留未合入旧 main 的修改 |
| `handoffs-20261018/` | 本地两份旧交接文档 | 原文保留，其“单步待验收”等状态已过期 |
| `manifest.json` | 归档校验清单 | 主线逐文件 SHA256、修复分支 ZIP SHA256 与提交身份 |

远程归档标签：`archive/pre-reproduction-20261018` 指向旧 GitHub main；`archive/fix-label-isolation-20261018` 指向旧修复分支。不重写历史，不删除原分支。

回看旧版本无需切换当前工作区：

```bash
git show archive/pre-reproduction-20261018:README.md
git show archive/fix-label-isolation-20261018:src/runner.py
```

本次独立整理不会搬动原电脑的 `D:/大创/进度` 文件，未跟踪的模型、下载、外部阅读材料与机器工具仍留在原处。Git 标签只保护已提交版本，不能视为这些本地资料的完整备份。
