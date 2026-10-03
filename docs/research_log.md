# 调研与实施记录

## 2026-10-03 · 新增开源业务场景调研

- **最新用户要求**：原合成采购实现和 README 保留；增加外部业务场景。用户进一步确认人工标注及部分合成都可以，有真实资料基础即可。
- **来源核查**：[WebShop](https://github.com/princeton-nlp/WebShop) 有真实商品与人工任务，但环境包含随机预算/价格；[ShopSimulator 论文](https://arxiv.org/html/2601.18225v1) 的商品和普通需求有真实/人工基础，画像和用户交互含合成成分。二者都符合当前标准，须如实记录来源。
- **框架边界**：[shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon) 当前使用 [Modern veRL] 0.8.0；可参考环境分工，不直接复制其 AgentLoop 到 [Search-R1 bundled veRL]。
- **设计建议，尚未选定**：[FinQA](https://github.com/czyssrs/FinQA) → [DocFinQA](https://huggingface.co/datasets/kensho/DocFinQA) 最贴近现有检索和短答案 IG；WebShop/ShopSimulator 更接近现成业务交互环境。金融路线需检查 FinQA 与 DocFinQA 的题目继承重叠；购物路线需处理 reward 接受多解与单参考目标 IG 的差异。
- **其他候选**：TAT-QA、ConvFinQA、BioASQ、PubMedQA、TechQA、BIRD；[AgentGym](https://github.com/WooooDyy/AgentGym) 提供 WebShop 和 BIRD/SQLGym 环境封装，可参考其环境服务接口。规模、来源、许可和获取入口见 [调研正文](open_source_scenarios.md)。
- **代码核对**：`protocol.py`、`reward.py`、`training/scoring.py` 等绑定报价 ID/采购事实；新数据不能直接通过改 `data_dir` 接入。接入计划已按实际文件列出。
- **本轮产物和边界**：新增调研文档，追加 README、导航、接手说明与状态；未改训练代码，未下载数据/权重，未跑模型或 GPU。外部场景适配与验证仍待实施。

## 2026-10-03 · 阶段 01：范围与用户选择

- **用户已确认**：面向 Agentic RL / Post-training 实习，电商采购模拟场景，4—6 周，人民币 2000 元租卡预算。
- **用户已确认**：学校按可独占 3 张 A6000 规划；第一版使用可控模拟目录，算法主线为回合级信用分配。
- **最新要求**：学校可用小模型，AutoDL 正式结果必须至少 3B。默认分别使用 Qwen2.5-1.5B-Instruct 与 Qwen2.5-3B-Instruct。
- **范围**：离线商品采购建议；规格、价格库存、供应商政策形成可验证任务。不声称真实企业部署或业务收益。

## 2026-10-03 · 阶段 02：参考项目核查

- **源码/文档已确认**：[shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon) 当前以 ShopSimulator + [Modern veRL] 0.8.0 为基础，未发现直接由 Search-R1 改造的证据。
- **文档已确认**：[历史评测](https://github.com/YYHDBL/shopping-grpo-longhorizon/blob/main/experiments/comparison.md) 的 SFT 60.5% → GRPO 62.0% 对应 200 题中的 3 题差异，无训练种子重复的稳定性证据，不作为本项目预期收益。
- **设计判断**：先冻结奖励、任务与评测，再做算法对照；用户允许多解的任务必须接受所有正确解。

## 2026-10-03 · 阶段 03：算法选择

- **论文已确认**：[IGPO v2](https://arxiv.org/html/2510.14967v2) 使用答案平均 log-prob 的回合增量，IG 与 outcome 分别归一化，再构造回合累计信号。v1 的概率差与合并归一化不能混称 v2。
- **实现选择**：初版批量 teacher-forcing 打分，复用训练 actor；不新增 critic。计入打分耗时，不套用论文优化 attention mask 后的开销数字。
- **消融选择**：outcome GRPO、同 IG 信号轨迹广播、回合 IGPO。后两组采用同样归一化，轨迹广播不再额外做 GRPO 标准化。
- **后续候选**：[AnySearch](https://arxiv.org/abs/2609.00813) 的预算控制；完整课程不纳入首版必需项。
- **局限**：正确答案置信度变化是检索价值的代理信号，不能直接解释为真实信息量或引用质量。

## 2026-10-03 · 阶段 04：环境和预算

- **版本规则**：以 [Search-R1 bundled veRL] 为实现基础，与 [Modern veRL] API 分开。
- **实际下载**：上游 Search-R1 固定到 `598e61bd1d36895726d28a8d06b3a15bed19f5d3`，目录 `third_party/Search-R1`。
- **源码已确认**：多轮分组使用 `rollout.n_agent`；某些 micro batch 参数在 worker 中除以 world size；必须保留工具 observation mask。
- **环境候选**：PyTorch 2.4.0、CUDA 12.1 wheel、vLLM 0.6.3。仍需在学校 Linux GPU 环境验证。
- **迁移选择**：学校 smoke 后，在云端从统一初始模型做正式对照。原始 checkpoint 不承诺保存完整 optimizer/RNG 状态。
- **预算**：1400 元训练与消融、300 元评测与重复、300 元余量。以实测 step 时间及下单整机价计算，不承诺时长。

## 2026-10-03 · 阶段 05：数据设计

- **设计选择**：有限候选目录，要求满足显式条件且含运费总价最低；主任务唯一最优，并列最优另作压力测试。
- **防捷径**：报价与文档 ID 独立随机分配；金标只用于训练评分；工具文档不得含答案标签。
- **交互选择**：根据商品、报价和政策信息的可见程度形成不同长度的检索路径；不奖励固定调用次数。
- **证据要求**：必须引用该轨迹实际获取、内容与冻结语料一致的文档，并覆盖任务所需事实。

## 当前验证边界

本日志区分调研事实、设计判断和实测结果。尚无学校或 AutoDL GPU 训练结果；不会将本地 CPU 测试或模拟策略成功率称为模型训练收益。后续检查结果在下方追加。

## 2026-10-03 · 阶段 06：学校接手与核心实现

- **实际文件核对**：Windows 交接时 README、server_setup、CLI、模型评测与训练 main/trainer 缺失；现已补齐。顶层没有可用 Git 元数据，记录源码文件 SHA256 而不编造 commit。
- **上游核验**：232 个文件仅 CRLF 与 pinned HEAD 不同；逐文件比对后恢复原始字节并刷新 Git 索引。bootstrap --check 已通过，没有上游逻辑补丁。
- **实现选择**：独立 ProcurementTrainer 复用旧 worker/resource pool/loss/save 接口。适配修复活跃 batch=1/world=3 补齐不足、旧训练循环少一次迭代、验证尾批遗漏及等权 batch 平均。
- **源码/机制核对**：IGPO v2 [公式 3–6](https://arxiv.org/html/2510.14967v2)；采用组内实际交互 IG 池与 outcome 分别标准化、gamma=1 后向累计。采购迁移增加固定 neutral answer prefix，只计算 offer ID；gold 只传入单独评分批次。未实现论文专用 attention mask 加速。
- **数据一致性修复**：训练 HTTP 和独立模型评测统一加载三个 split 的公共语料；使用同一 BM25 corpus hash。回执须对应真实 search query，文档内容必须与冻结事实和实际曝光一致。
- **实测 CPU token 预算**：Qwen tokenizer snapshot `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，完整数据 2000/200/400；最长 prompt=223、单篇完整工具文档=614 tokens，26 篇超过旧 512 限额。调整 max_obs_length=1024、rolling context=6144，预检检查完整文档，不截断后冒充完整证据。
- **诊断选择**：真实 optimizer 前检查梯度 norm 有限，采样参数变化提供更新证据；未变化样本不证明全模型不变。记录 PyTorch allocator 峰值，不冒充整机显存峰值。训练迭代和 PPO mini-batch optimizer step 分开计数。

## 2026-10-03 · 阶段 07：最小验收与用户执行边界

- **最新用户要求**：助手完成整个 repo 代码、README 与最小规模运行；数据/模型准备、GPU 冒烟至完整实验由用户自行执行。停止下载和 GPU 启动工作。
- **CPU 实测**：独立 CPU 环境 36 项 passed / 6 项因训练依赖 absent skipped；现有 Search-R1 Python 环境关闭 CUDA 后 12 项专项通过（包括这 6 项）。覆盖数学、真实 DataProto 接线、provenance、gold 隔离、driver 迭代/尾批及 CPU optimizer 的参数变化/非有限梯度拒绝。
- **最小集成实测**：`bash scripts/check_cpu.sh` 生成 24/6/6 任务、108 SKU、107 文档，导出 parquet，6/6 scripted checks 通过；产物 `outputs/cpu-check-IWfUBy/`。它是固定语法的公开文档解析程序，没有使用语言模型，不能作为模型成绩。
- **配置实测**：四个 profile 静态合法，均通过 pinned Hydra 配置合成；未启动 Ray worker 或 GPU。shell LF、语法、Python 编译、干净上游校验通过。
- **环境事实**：项目 CPU venv 独立；已有 searchr1 的 FlashAttention=2.6.3，候选为2.7.0.post2。ABI CPU 导入不等于 GPU kernel 验证。没有改全局 Python/CUDA，没有完整 Qwen 权重，没有模型/训练收益。
- **文档交付**：README 入门、implementation 代码阅读、server_setup 环境与冒烟、experiments 从冻结数据/模型到正式对照的命令。正式仍 ≥3B，学校1.5B。运行目录保存实际配置/版本/hash/cost/failure；未给租价时费用未知。
