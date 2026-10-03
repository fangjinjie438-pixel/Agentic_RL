# 从数据准备到正式对照

这些操作由你执行。代码最小运行与 GPU/模型实验分开验收，GPU 是否可运行以实际输出为准。

## A. 准备与最小运行

```bash
python3 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -e '.[dev,data]'
python scripts/bootstrap_search_r1.py
bash scripts/check_cpu.sh
```

bootstrap 仅在固定上游缺失时下载，已有 checkout 则核对 SHA 与干净状态。最小检查使用 24/6/6 条任务，跑 CPU 测试、真实 BM25、parquet 导出、脚本策略交互和静态启动配置；产物路径打印在末尾。它不会运行模型或训练 GPU。

## B. 数据集与模型

本项目使用自生成采购事实，不需要下载外部数据集。正式数据只生成一次，三个算法都使用同一目录：

```bash
python -m procurement_rl prepare --output data/procurement-v1 \
  --train-size 2000 --val-size 200 --test-size 400 --seed 42
python -m procurement_rl export --data-dir data/procurement-v1
```

`manifest.json` 冻结 6 个 JSONL 文件，`parquet_manifest.json` 记录导出哈希。不要手改文件后继续沿用旧版本名称。每个 split 的 `corpus.jsonl` 为公开检索语料，`tasks.jsonl` 包含私有验证事实；后者不能输入检索服务。

安装 Linux 训练环境后，按 `server_setup.md` 下载学校 1.5B。正式模型使用 Qwen2.5-3B-Instruct，先通过 Hugging Face 官方 API 取得 commit SHA 并保存，随后按 SHA 下载：

```bash
python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("Qwen/Qwen2.5-3B-Instruct").sha)'
python scripts/download_model.py --profile configs/autodl_3b.yaml --revision YOUR_40_CHARACTER_SHA
```

命令会输出实际 snapshot 路径。后续 `--model` 统一使用此绝对路径，不使用浮动 revision。正式训练预检和正式评测均会检查实际参数数 ≥3B。

## C. 学校冒烟

按 [服务器说明](server_setup.md) 选择单卡或三卡，启动 HTTP 检索、运行预检，依次 smoke 三种算法。使用 `--data-dir data/procurement-v1` 指向上述数据；检索器也设置同一路径：

```bash
DATA_DIR=data/procurement-v1 bash scripts/run_retriever.sh
```

先跑未训练 1.5B 的小规模真实模型评测（例如 val 12 题），再跑 5 次训练迭代、checkpoint 重载和 `scripts/check_smoke.py`。先解决非法动作、全零优势、检索和显存问题，再进入正式实验。若需共同格式 SFT，先固定可回放轨迹与共同初始化，并单独报告这个基线；本仓库未将 SFT 纳入默认训练流程。

## D. AutoDL 正式实验

下列变量必须填实际值；环境保持一致，三个方法每次从同一初始模型开始：

```bash
export CUDA_VISIBLE_DEVICES=0
MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct/snapshot
DATA_PATH=/absolute/path/to/data/procurement-v1
HOURLY_CNY=YOUR_ACTUAL_INSTANCE_HOURLY_PRICE
```

先启动同数据目录的检索器，并在另一个终端执行预检：

```bash
python scripts/preflight.py --profile configs/autodl_3b.yaml \
  --model "$MODEL_PATH" --data-dir "$DATA_PATH" --check-retriever
```

未训练模型基线使用固定 test 集、4 次交互上限和 greedy 解码：

```bash
python -m procurement_rl.evaluate --data-dir "$DATA_PATH" --split test \
  --model "$MODEL_PATH" --initialization "$MODEL_PATH" --purpose formal \
  --max-turns 4 --max-obs-tokens 1024 --max-context-tokens 6144 \
  --output outputs/base-3b-test --hourly-cost-cny "$HOURLY_CNY"
```

正式训练命令（先确认短规模试跑费用与稳定性；默认 200 次迭代）：

```bash
for seed in 42 43; do
  for method in grpo igpo; do
    bash scripts/train.sh --profile configs/autodl_3b.yaml --algorithm "$method" \
      --seed "$seed" --model "$MODEL_PATH" --data-dir "$DATA_PATH" \
      --run-name "3b-${method}-s${seed}" --hourly-cost-cny "$HOURLY_CNY"
  done
done
bash scripts/train.sh --profile configs/autodl_3b.yaml --algorithm trajectory_ig \
  --seed 42 --model "$MODEL_PATH" --data-dir "$DATA_PATH" \
  --run-name 3b-trajectory_ig-s42 --hourly-cost-cny "$HOURLY_CNY"
```

短规模成本预估可以用单独输出目录加 `--set upstream.trainer.total_training_steps=20`；这些预跑不复用为正式结果。不得把通过静态校验的双 4090 配置当成已通过显存验证。

对每个最终 checkpoint 使用与基线相同的评测参数，例如：

```bash
python -m procurement_rl.evaluate --data-dir "$DATA_PATH" --split test \
  --model outputs/3b-igpo-s42/checkpoints/actor/global_step_200 \
  --initialization "$MODEL_PATH" --purpose formal \
  --max-turns 4 --max-obs-tokens 1024 --max-context-tokens 6144 \
  --output outputs/3b-igpo-s42-test --hourly-cost-cny "$HOURLY_CNY"
python -m procurement_rl.metrics paired \
  outputs/3b-grpo-s42-test/results.jsonl outputs/3b-igpo-s42-test/results.jsonl
```

测试集只在实验方案确定后使用。用 val 选择超参数/格式修复，不根据 test 成绩调参。比较表同时填写每个种子的严格成功率、引用有效率、平均调用次数、token、训练/评分/评测时间和费用。家族 bootstrap 的区间不包含训练种子方差，需另外报告重复种子的结果。

## E. 结果与失败分析

- `manifest.json`：类型、状态、实际配置、源码/数据/权重哈希、版本、耗时、费用。
- `resolved_profile.json` / `invocation.json` / `pip-freeze.txt`：训练启动与环境记录。
- `metrics.jsonl`：按迭代记录 loss、KL、梯度、优势、评分耗时、参数变化样本。
- `trajectories/*.jsonl`：模型动作、真实曝光文档、私有评分诊断、各回合优势。
- `results.jsonl` / `summary.json`：真实模型逐题结果与汇总；`kind` 区别模型评测和脚本检查。

失败运行保留 manifest 和日志；重跑换新名称。原 checkpoint 没有完整 optimizer/RNG 恢复保证，中断后不要声称无缝续训。真实整机费用还应计入计费取整、空闲、初始化及存储；程序记录的 wall-time 费用只是估算。
