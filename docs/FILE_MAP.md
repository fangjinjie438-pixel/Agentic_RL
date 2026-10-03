# 文件与目录导航

新接手顺序：`STATUS.md` → 本文件 → 根目录 `README.md` → `server_setup.md`。下表为项目约定；实施中以 STATUS 的验收记录为准。

| 路径 | 作用 |
|---|---|
| `AGENTS.md` | 给后续助手的项目约束和接手入口。 |
| `README.md` | 项目介绍、最短运行流程、验证边界。 |
| `pyproject.toml` | 项目安装、可选依赖、命令入口、测试设置。 |
| `upstream.lock.json` | Search-R1 来源及固定 commit。 |
| `src/procurement_rl/data.py` | 模拟目录与任务生成、切分及数据哈希。 |
| `src/procurement_rl/retrieval.py` | CPU BM25 和兼容 Search-R1 的检索 HTTP 服务。 |
| `src/procurement_rl/protocol.py` | 模型 search / evidence / answer 协议解析。 |
| `src/procurement_rl/reward.py` | 私有事实核验：约束、最低价和真实引用。 |
| `src/procurement_rl/training/` | 多轮 rollout、优势计算、actor 打分及 bundled veRL 接入。 |
| `src/procurement_rl/cli.py` | CPU 数据准备、导出和演示入口。 |
| `src/procurement_rl/evaluate.py` | 从真实模型生成轨迹并评测，写逐题结果。 |
| `src/procurement_rl/metrics.py` | 汇总、配对统计和预算估算。 |
| `src/procurement_rl/tracking.py` | 实验 manifest、版本、哈希、耗时与成本。 |
| `scripts/check_cpu.sh` | 一键最小 CPU 验收。 |
| `scripts/download_model.py` | 按固定 revision 下载模型，由用户执行。 |
| `scripts/check_smoke.py` | 核对真实训练与 checkpoint 重载证据。 |
| `docs/implementation.md` | 代码阅读路线、算法与上游边界。 |
| `docs/experiments.md` | 数据与模型准备、基线、三算法及重复实验命令。 |
| `docs/open_source_scenarios.md` | 新增开源业务场景选型：真实来源、交互环境、奖励、获取入口、许可及待实现的接入方案。 |
| `configs/` | 学校 1.5B 与 AutoDL 3B 的独立配置。 |
| `scripts/` | 固定上游下载、Linux 安装、预检、启动。 |
| `tests/` | 奖励、泄漏、数学、协议与配置等关键检查。 |
| `third_party/Search-R1/` | 固定版本上游源码；保留原始许可。 |
| `data/` | 生成数据、检索语料、训练 parquet；可重新生成。 |
| `outputs/` | 模型评测记录、演示记录和统计结果。 |
| `docs/research_log.md` | 阶段结论、来源、方案变化及验证结果。 |
| `docs/paper_notes.md` | 论文机制与本项目移植边界。 |
| `docs/server_setup.md` | Linux 学校与 AutoDL 上机步骤、环境诊断。 |

## 数据与训练流

任务生成 → 公开语料 / 私有核验事实分离 → 模型 search → BM25 返回带 ID 文档 → 模型继续生成 → 程序核验终局 → GRPO 或 IGPO 优势 → actor 更新。

IGPO 的 gold 答案打分只发生在训练评分阶段；gold 不传给检索器，不追加到真实 rollout。工具文本参与上下文，但不参与 policy loss。
