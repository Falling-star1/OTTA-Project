# 三套数据与离线主指标资源准备

本包不含大模型，不安装 Python 包，不运行训练或模型推理。
服务器已有的数据已核验完整，脚本会复制选定的三套数据到 /data，并生成 16 条调试版本。
主代码放入项目的 local_repro/stage1；原仓库代码和原始数据文件保留。

Windows PowerShell 上传：

```powershell
scp -r "D:\大创\进度\offline_tta\tlm_stage1_bundle" chuwenjing@10.126.62.55:/data/chuwenjing/tta_offline/
```

服务器执行：

```bash
cd /data/chuwenjing/tta_offline/tlm_stage1_bundle
bash prepare_stage1.sh
```

成功标记为 `TLM_STAGE1_DATA=PASS`。脚本先检查传输哈希和源数据哈希，遇到内容不同的同名目标文件会停止。
随后验证 NLTK 离线分句和 ROUGE-Lsum 的基本运行。不会联网下载资源。
这一步不表示正式模型、GSM8K 答案提取或四种适应方法已经验证。

日后在服务器新终端进入项目，可执行：

```bash
source /home/chuwenjing/TTA_LLM/TLM/local_repro/stage1/activate_tlm.sh
```

这会激活独立环境，并将模型/数据缓存、NLTK 资源路径指向 /data；采用离线读取模式。

NLTK 英文资源来自官方 nltk/nltk_data 的固定提交，来源、归档 Git blob 校验值和传输 SHA256 见 nltk-manifest.json。
