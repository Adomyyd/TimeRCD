# 实验

本目录提供两条独立实验路线，共用同一份400k单变量数据。

| 路线 | 做什么 | 完整命令 |
|---|---|---|
| original_400k | 原TimeRCD流程的400k复现：生成数据、恢复训练、对比检测结果 | [使用说明](original_400k/README.md) |
| ablation_v9 | P0/P1/P2消融、padding对照、多GPU训练、合成与真实数据统一评测 | [使用说明](ablation_v9/README.md) |

## 快速检查

在仓库根目录激活项目环境，并先安装项目及检查依赖：

```bash
python -m pip install . pytest -r experiments/ablation_v9/requirements-eval.txt
```

以下命令查看入口并运行合成回归，不启动正式训练，也不需要400k数据：

```bash
python experiments/original_400k/run_experiment.py --help
python experiments/ablation_v9/run_experiment.py --help
python experiments/ablation_v9/tests/run_checks.py
```

## 运行前准备

将已有数据和同名manifest放在`experiments/original_400k/data/`。v9直接使用其`configs/`中的配置；原流程续训还需要原run的检查点。数据、权重和检查点不随Git仓库分发。v9完整评测还需准备`datasets/TSB-AD-U/`；一次处理合成留出与官方单变量11类700条真实序列，完整命令见v9使用说明。

日志和模型分别保存到两条路线各自的`runs/`目录。`_archive/`保存维护历史，不参与实验执行。
