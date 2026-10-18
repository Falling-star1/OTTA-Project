# Reproduction Hub Implementation Plan

> For agentic workers: execute inline under the user's explicit request to complete the work; no separate approval checkpoint is required.

**Goal:** 将旧 GitHub 内容可验证地归档，把当前 TLM 复现进度整理为团队可协作的主线并发布。

**Architecture:** 独立工作树基于已同步 origin/main；原主线逐文件归档，旧修复分支完整 ZIP 留档。冻结实验包按原字节复制，当前文档作为唯一状态入口。

**Tech Stack:** Git、Markdown、Python 标准库、现有 unittest 与 GitHub REST API。

## Steps

- [x] 同步 origin，确认 main=807856c、旧修复分支=ec8dae8 和发布权限。
- [x] 建立独立工作树，复制 main 全部跟踪文件及修复分支归档，保存 SHA256 清单。
- [x] 写入项目首页、总览、看板、团队协作、复现协议、环境运行与实验记录。
- [x] 复制 stage1、Source、Tent 单步、offline 小样本和确定性全量包，保持哈希。
- [x] 运行 `python scripts/verify_repository.py`，确认归档、bundle 和链接完整性。
- [x] 使用已安装兼容库运行 `python scripts/run_cpu_tests.py --libs D:/大创/进度/.tools/tent-compat`，确认 39 项 CPU 用例。
- [x] 检查暂存区体积、敏感信息、变更范围与 Git 换行行为。
- [ ] 推送两份归档标签和 codex/reproduction-hub 分支，创建并核对 PR 后合并。
- [ ] 核对 GitHub main 提交与标签，报告更新链接、未完成实验和团队认领事项。
