# 论文与移植边界

检索日期：2026-10-03。以下是方法摘要和实施决策，不是论文实验结果复现。

| 方法 | 一手来源 | 采用与边界 |
|---|---|---|
| Search-R1 | [原仓](https://github.com/PeterGriffinJin/Search-R1)、[论文](https://arxiv.org/abs/2503.09516) | 基线使用其 bundled veRL；固定源码 SHA，不混用现代 AgentLoop API。 |
| IGPO v2 | [2026-03-24 修订](https://arxiv.org/html/2510.14967v2)、[作者代码](https://github.com/GuoqingWang1/IGPO) | 平均 token log-prob 增量、分别归一化、回合累计信号；采购任务 outcome 与原 QA 任务不同。朴素打分需要实际测量开销。 |
| AnySearch | [2026-09-01 论文](https://arxiv.org/abs/2609.00813)、[代码](https://github.com/xwsun01/AnySearch) | 后续扩展：同组同预算、正确性门控效率奖励。原实现基于 Slime；仅移植部分机制时称 inspired，不称完整复现。 |
| GiGPO | [论文](https://arxiv.org/abs/2505.10978)、[verl-agent](https://github.com/langfengQ/verl-agent) | 同环境状态分组，不把相同回合编号误当相同状态。暂不实现。 |
| AT²PO | [论文](https://arxiv.org/abs/2601.04767)、[代码](https://github.com/zzfoutofspace/ATPO) | 完整方法包含树展开和回合目标；单独采用 ATPO loss 不等于复现全部方法。暂不实现。 |
| MR-Search | [论文](https://arxiv.org/abs/2603.11327)、[代码](https://github.com/tengxiao1/MR-Search) | 跨 episode 反思训练增加生成与上下文；提示词要求反思不能等同方法复现。暂不实现。 |
| Query Recycling | [论文](https://arxiv.org/abs/2606.10709)、[作者仓库](https://github.com/cxcscmu/agentic_search_query_recycling) | 复用问题并重新 on-policy rollout；不是复用旧轨迹。额外采样需计费，首版不实施。 |

## 采购任务中的 IGPO

金标是唯一最优 offer ID，仅用于离线打分。初始状态与每次工具返回后 teacher-force 相同的短答案，不让最终生成的答案/引用进入打分前缀。只给实际生成 token 分配策略梯度。

引用正确性由独立规则奖励负责。ID 对错置信度变化不直接监督引用质量。商品资料全部明确为模拟数据；无法据此声称真实商城泛化。

## 需要保留的实验

相同模型、语料、任务奖励、组大小和采样条件下，比较 outcome GRPO、同信息增益信号的轨迹广播、回合 IGPO。优先给 GRPO 与 IGPO 补训练种子，报告成功率、引用、工具调用、生成 token、GPU 时间和打分耗时。
