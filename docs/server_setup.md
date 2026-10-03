# 学校 Linux / AutoDL 操作说明

## 1. 环境与版本

先读 [STATUS](STATUS.md)。当前候选栈是 Python 3.9、PyTorch 2.4.0+cu121、vLLM 0.6.3、transformers 4.46.3、tensordict 0.5.0、FlashAttention 2.7.0.post2。完整候选约束见 `requirements/linux-candidate.txt`；成功运行后还需保存实际 `pip freeze`，候选列表不是完整依赖锁。

2026-10-03 学校只读检查发现 `/home/fangjinjie/miniconda3/envs/searchr1` 已有大部分依赖，但 FlashAttention 为 2.6.3。CPU ABI 导入通过，GPU kernel 未测。项目 `.venv-gpu` 只读继承该环境，尚未补齐候选版本，不能视为已通过预检。系统环境未修改。

新机器建议独立安装：

```bash
conda create --prefix ./environments/search-r1 python=3.9 -y
conda activate ./environments/search-r1
export FLASH_ATTN_WHEEL=/absolute/path/to/compatible/flash_attn.whl
bash scripts/setup_linux.sh
```

FlashAttention wheel 必须匹配 Python、torch、CUDA 和 C++ ABI；使用 [作者发布页](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.0.post2) 的构建。没有兼容 wheel 时，安装脚本只在显式设置 `BUILD_FLASH_ATTN=1` 且 `CUDA_HOME` 指向 CUDA 12.1 Toolkit 时编译。不要把 `nvidia-smi` 的 CUDA 版本当成 Toolkit，也不要修改系统 CUDA 链接。

## 2. 固定模型与数据

学校模型已下载分词器和配置，snapshot revision 为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；**该目录仅含分词器与配置，模型权重由你按以下命令准备**。有本地完整模型时直接使用其路径。

下载命令（不初始化 GPU）：

```bash
python scripts/download_model.py --profile configs/school_single.yaml \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

AutoDL 换用 `configs/autodl_3b.yaml` 并提供该 3B 模型的真实固定 revision，不能复用 1.5B 的 revision 或权重。不同算法均从同一 snapshot 初始化。

数据准备见 README。已有 `data/synthetic` 时直接预检，不重新生成覆盖：

```bash
python scripts/bootstrap_search_r1.py --check
python scripts/preflight.py --profile configs/school_single.yaml --data-only \
  --model /absolute/path/to/local/tokenizer-or-model
```

`--data-only` 在 CPU 上检查冻结文件哈希、parquet 逐行一致性、提示词和完整文档 token 长度。当前 Qwen 分词器测得最长提示词 223 tokens，最长单篇文档含工具包装 614 tokens。候选 observation 预算已从 512 调为 1024，rolling context 为 6144。检索结果只保留放得下的完整文档，被丢弃文档不记为已获得证据。

## 3. 启动检索器与预检

终端 A，激活任一可导入本项目的 Python 环境：

```bash
bash scripts/run_retriever.sh
```

工具服务加载三个 split 的公共语料，与模型评测一致；不加载 `tasks.jsonl`。默认仅监听 `127.0.0.1:8000`。客户端对本机地址禁用代理，避免学校代理配置把本机请求转发到外网。

终端 B，选择已分配的 GPU。例如单卡已分配 1 号卡时：

```bash
export CUDA_VISIBLE_DEVICES=1
python scripts/preflight.py --profile configs/school_single.yaml \
  --model /absolute/path/to/Qwen2.5-1.5B-Instruct --data-dir data/synthetic --check-retriever
```

预检验证环境版本、驱动、可见 GPU 数和空闲显存、CPU 内存、模型参数/权重分片/上下文、数据与 HTTP 检索 corpus hash。`--config-only` 只证明静态配置合法。

## 4. 学校真实训练

```bash
bash scripts/train.sh --profile configs/school_single.yaml --algorithm grpo \
  --model /absolute/path/to/Qwen2.5-1.5B-Instruct --run-name school-grpo-s42
bash scripts/train.sh --profile configs/school_single.yaml --algorithm trajectory_ig \
  --model /absolute/path/to/Qwen2.5-1.5B-Instruct --run-name school-broadcast-s42
bash scripts/train.sh --profile configs/school_single.yaml --algorithm igpo \
  --model /absolute/path/to/Qwen2.5-1.5B-Instruct --run-name school-igpo-s42
```

三卡用 `school_smoke.yaml` 和实际分配的三个 A6000 编号；它的 global micro batch 为 3，单卡配置为 1。不要混用 [Modern veRL] 参数。训练输出在 `outputs/<run-name>/`，输出目录不可覆盖。

验收需有真实证据：

- 发生 search → 工具返回 → 继续生成 → answer。
- 同题组 ID 正确，至少出现非零优势；工具/填充 token loss mask 为零。
- loss/KL/梯度有限，参数样本观察到实际改变。
- 真实保存 checkpoint，并用该 checkpoint 完成下述模型评测。
- 检查每步耗时、IG 评分开销、生成 token、零优势组比例、丢弃文档和预算耗尽率。

actor 记录本进程 PyTorch 分配/保留峰值，不等于整机 `nvidia-smi` 峰值；整机占用仍需单独采样。参数样本不变不能证明整个模型未更新；样本改变则证明至少部分参数发生更新。

全零结果奖励可能导致 GRPO 全零优势；不能把只有有限 loss 的运行当成学习验收。先评测格式和检索行为，必要时所有方法共用可回放的格式 SFT，再重新建立基线。

## 5. checkpoint 重载与 AutoDL

```bash
python -m procurement_rl.evaluate --data-dir data/synthetic --split val \
  --model outputs/school-grpo-s42/checkpoints/actor/global_step_5 \
  --purpose smoke --device cuda:0 --limit 12 --output outputs/school-grpo-reload
```

实际目录以保存日志为准。保留 pinned upstream 的模型/tokenizer checkpoint 语义，不承诺恢复 optimizer/RNG。云端对照从统一初始模型开始，不把学校训练产物当成正式初始化。

正式实验采用 ≥3B，统一数据/初始化/检索器/verifier，GRPO 和 IGPO 至少各两个种子，广播先一个种子。先测稳定步骤与保存评测总耗时，再根据真实整机租价安排 2000 元预算。训练与评测 `--hourly-cost-cny` 记录实际租价；学校无租卡费用时可明确传 0，未知则省略。

保存与重载完成后自动检查工程验收：

```bash
python scripts/check_smoke.py --run outputs/school-grpo-s42 --reload-eval outputs/school-grpo-reload
```

只有实际训练、非零优势、参数变化、工具链路和 checkpoint 重载证据满足要求才写入 `smoke_acceptance.json`。每次 rollout/update 迭代含多个 PPO mini batch，因此 5 次训练迭代不一定只有 5 次 optimizer step；日志另记 `actor/optimizer_steps`。
