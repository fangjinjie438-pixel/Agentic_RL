# 企业采购检索 Agent：Search-R1 与回合级信用分配

模拟企业采购任务：根据商品规格、报价库存和供应商政策，选择满足约束且含运费总价最低的报价，引用本条轨迹实际检索到的证据。金额按整数分计算，事实和数据版本冻结，商品家族不跨 train/val/test。

实现基于 **[Search-R1 bundled veRL]**，固定 SHA 见 [upstream.lock.json](upstream.lock.json)。上游源码与许可证保留，项目适配位于 `src/procurement_rl/training/`。不使用 [Modern veRL] AgentLoop API。

## 当前状态

核心环境、三种算法的训练接入、真实模型评测、配对统计和启动工具已实现，并通过 CPU 检查。**GPU 训练、模型收益与 checkpoint 重载尚未验收。** 详细结果与下一步见 [STATUS](docs/STATUS.md)，目录见 [FILE_MAP](docs/FILE_MAP.md)。

| 方法 | 训练信号 |
|---|---|
| `grpo` | 严格业务结果奖励，在同题采样组内标准化并广播 |
| `trajectory_ig` | IG 与结果分别标准化；把首回合累计优势广播到所有生成 token |
| `igpo` | 相同 IG 与结果信号，按回合累计并映射到相应生成 token |

IG 来自更新前 actor 对短 gold offer ID 的平均 log-prob 增量。gold 只进入私有打分批次，不进入真实 rollout 或工具返回；工具 token 的 policy loss mask 为零。采用 IGPO v2 的主要信号机制，分批 teacher forcing，未实现论文专用 attention mask 加速。方法边界见 [论文记录](docs/paper_notes.md)。

## 上手顺序

- [代码阅读路线与训练数据流](docs/implementation.md)：先理解环境，再跟进 rollout、评分与更新。
- [学校与 AutoDL 环境、GPU 冒烟](docs/server_setup.md)：安装、预检、运行和 checkpoint 重载。
- [数据、模型、完整对照实验步骤](docs/experiments.md)：由你执行的准备和实验命令。

代码最小验收一条命令：激活下方 CPU 环境后运行 `bash scripts/check_cpu.sh`。它不下载模型或运行 GPU。

## CPU 最短流程

在项目目录执行；现有数据和结果目录应复用或换新路径，不覆盖冻结数据：

```bash
python3 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -e '.[dev,data]'
python -m pytest -q
python scripts/bootstrap_search_r1.py --check
python -m procurement_rl prepare --output data/synthetic
python -m procurement_rl export --data-dir data/synthetic
python -m procurement_rl demo --data-dir data/synthetic --split test --output outputs/demo
python scripts/preflight.py --profile configs/school_single.yaml --config-only
python scripts/launch_train.py --profile configs/school_single.yaml --algorithm igpo --dry-run
```

默认生成 999 SKU、2000/200/400 条任务；train/val/test 的公共文档组成同一个冻结检索索引。私有事实只存于任务文件和 parquet 的 `reward_model.ground_truth`，不能作为工具语料。导出记录源文件和 parquet 哈希，预检逐行检查二者一致。

`demo` 是读取公开文档的确定性程序，结果标记为 `scripted_environment_check`。其成功率不能报告成语言模型性能。CPU 环境会跳过需要 torch/旧 DataProto 的测试；在训练环境中补跑：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -p 'test_training*.py' -v
```

## 学校 smoke 与正式实验

完整操作见 [服务器说明](docs/server_setup.md)。学校使用 **Qwen2.5-1.5B-Instruct**，AutoDL 正式训练和评测必须 **至少 3B**，默认 **Qwen2.5-3B-Instruct**。正式配置有名称检查和实际架构参数计数，不能通过改成本地路径绕过下限。

| Profile | 用途 | GPU | 训练迭代数 |
|---|---|---:|---:|
| `school_single.yaml` | 单卡 smoke | 1 × A6000 | 5 |
| `school_smoke.yaml` | 三卡 smoke | 3 × A6000 | 5 |
| `autodl_3b.yaml` | 正式实验候选 | 1 × 80GB | 200 |
| `autodl_3b_dual4090.yaml` | 正式双卡候选 | 2 × 24GB | 200 |

这些配置的可运行性、显存和速度仍需 GPU 实测。动态选择已分配且空闲的卡，不按旧 GPU 快照直接启动。

## 真实模型评测与比较

```bash
python -m procurement_rl.evaluate --data-dir data/synthetic --split test \
  --model /absolute/path/to/model --purpose smoke --device cuda:0 \
  --max-turns 3 --max-obs-tokens 1024 --output outputs/model-eval
python -m procurement_rl.metrics summarize outputs/model-eval/results.jsonl
python -m procurement_rl.metrics paired outputs/grpo-eval/results.jsonl outputs/igpo-eval/results.jsonl
```

正式评测使用 `--purpose formal`。所有对照统一初始化、数据、检索器、奖励、解码和预算；先测未训练模型，再测三个算法。配对工具检查题目、语料、verifier 与解码配置一致，以商品家族为单位 bootstrap。仍需检查 manifest 中的初始化来源，不能只根据 checkpoint 名称判断。

每次训练/评测保留配置、代码和数据哈希、依赖版本、起止时间、逐题/逐步日志及失败记录。`--hourly-cost-cny` 填真实整机小时价；未提供时成本记为未知。没有实测结果前不填写提升数字。

## 新增方向：基于开源业务资料的场景

选型与接入设计见 [开源业务场景调研](docs/open_source_scenarios.md)。筛选标准为真实商品、财报、文献、技术支持资料或数据库作为基础，允许人工标注和部分合成。

- **金融检索与计算**：FinQA 起步，DocFinQA 扩展；最容易延续当前 Search-R1 与 IGPO 主线。
- **购物交互**：WebShop 或中文 ShopSimulator；复用外部商品环境、动作与奖励。
- **其他候选**：TAT-QA、ConvFinQA、BioASQ、PubMedQA、TechQA、BIRD-SQL。

这些方向目前完成了来源核查与改造设计，**尚未接入训练入口**。上面的安装、CPU 验收与实验命令继续用于原模拟采购场景；新场景需要独立的数据导入、协议、奖励和 IG 目标适配。
