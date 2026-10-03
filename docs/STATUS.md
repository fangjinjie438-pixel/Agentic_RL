# 当前进度与接手说明

更新时间：2026-10-03；当前范围：repo 实现、README 和最小规模运行验收。用户明确要求下载数据/权重、GPU smoke 与完整实验由其自行执行。

## 结论

主线代码与操作文档已补齐，最小 CPU 流程已实际跑通。**没有运行 Qwen 模型评测或 GPU 训练，没有训练收益数据。** CPU tensor/fake-worker 测试不证明 FSDP/vLLM GPU 训练可用。

## 已实现

- 冻结模拟采购事实、整数金额、family-disjoint split；BM25 公共语料与私有 verifier 分离。
- search/evidence/answer 协议、真实曝光文档回执、严格约束/最优价/引用核验；拒绝私有语料字段与篡改证据。
- CPU prepare/export/demo，旧版 RLHFDataset parquet 契约，真实 HF 模型评测入口，family-clustered 配对统计。
- [Search-R1 bundled veRL] 多轮 rollout、原始 token 与 observation mask、更新前 actor gold teacher-forcing、GRPO / trajectory_ig / IGPO 三种优势。
- 独立 driver、seed 接线、三卡缩小批次补齐、完整验证尾批与精确训练迭代数；optimizer 前拒绝非有限梯度，记录参数变化样本、评分耗时、零优势组和显存指标。
- 学校 1.5B 单/三卡与 AutoDL ≥3B 单/双卡 profile；正式模型下限同时有静态检查和实际参数计数。
- 静态/数据/运行时预检；冻结数据与 parquet 逐行一致性、HTTP 检索 corpus hash、模型权重分片检查。
- 每次训练/评测的配置、源码/数据/权重哈希、版本、成本与失败 manifest；CPU 一键验收和真实 smoke 证据核验脚本。
- README、代码阅读路线、服务器安装/冒烟、模型数据准备/完整实验命令已完成。

## 实际验收（2026-10-03）

| 检查 | 结果与边界 |
|---|---|
| `.venv-cpu` pytest | 36 passed，6 个需要 torch/旧框架的测试 skipped。 |
| 旧训练环境 CPU 专项 | 12/12 passed，其中包含上述 6 项；`CUDA_VISIBLE_DEVICES=''`，真实 DataProto + fake workers，小型 CPU SGD 参数更新/非有限梯度拒绝。 |
| `bash scripts/check_cpu.sh` | 24/6/6 任务生成、108 SKU、107 公共文档、parquet 导出、6/6 脚本环境检查通过；不是语言模型成绩。 |
| 四个 profile | 静态检查及 pinned Hydra `--cfg job --resolve` 全通过；未初始化 Ray/GPU。 |
| Qwen 1.5B tokenizer 检查 | 对本地已生成 2000/200/400 数据检查：提示词最长 223 tokens，完整单篇文档含包装最长 614 tokens；26 篇超过旧 512 限额。已改 observation=1024、rolling context=6144，数据预检通过。 |
| 固定上游 | SHA `598e61bd1d36895726d28a8d06b3a15bed19f5d3` 核验通过，checkout 干净，LICENSE 保留。 |
| Linux 入口 | shell 文件 LF、bash 语法、Python 编译检查通过。 |

最小验收产物：`outputs/cpu-check-IWfUBy/`，含数据/导出 manifest、demo 逐题结果、pip freeze 与启动 dry-run。该记录中的 `kind=scripted_environment_check`。更早的 `outputs/cpu-demo-20261003/` 为共享索引修复前的调试产物，不作为当前验收。

## 环境与保留事项

- 顶层目录没有可用 Git 仓库元数据；实验记录使用逐文件 SHA256，顶层 `git_head=null`，不能冒充已提交版本。
- Windows 迁移产生 232 个上游 CRLF-only 变化。逐文件核对后恢复 pinned HEAD 原始字节，刷新迁移索引；没有修改上游逻辑。
- `.venv-cpu` 为项目独立 Python 3.10.12 环境。现有 `searchr1` 为 Python 3.9.25 / torch 2.4.0+cu121 / vLLM 0.6.3 / FlashAttention 2.6.3；`.venv-gpu` 只读继承此环境。候选要求 FlashAttention 2.7.0.post2，运行时预检会拒绝当前差异。全局 Python/CUDA 未修改。
- tokenizer/config snapshot `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` 在 `data/model-cache/`；没有下载完整 Qwen 权重。模型与数据准备之后由用户自行执行。
- 原上游 checkpoint 只保留其模型/tokenizer 语义，不实现 optimizer/RNG 完整恢复；SFT 与 AnySearch 扩展未纳入默认主线。

## 用户下一步

1. 从根目录 [README](../README.md) 入手，运行 `bash scripts/check_cpu.sh`；阅读 [实现路线](implementation.md)。
2. 按 [服务器说明](server_setup.md) 建立独立候选训练环境、准备 1.5B 权重、选择已分配 GPU，执行预检和学校 smoke。
3. 真实训练须验证非零信号、有限梯度、参数变化和 checkpoint 重载；满足证据后才可记录 GPU smoke 通过。
4. 按 [实验步骤](experiments.md) 下载固定 3B 模型并做统一基线、三算法和重复种子实验。显存、耗时、费用及收益均填实测值。
