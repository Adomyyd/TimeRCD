# 实验工作区

> 迁移整理注（2026-11-10）：本页已统一路径表示与环境表述，实验数值和结论保留。旧清单中的文档摘要对应整理前版本；历史命令不作为当前用法，当前入口见两路线README。

两条实验路线各自管理代码、配置和产物，目录并列：

```text
experiments/
├── README.md
├── original_400k/
│   ├── run_experiment.py
│   ├── src/                 # 原流程的数据生成、训练和比较
│   ├── data/                # 400k 数据及 manifest，v9 只读共用
│   ├── runs/                # 原流程运行产物
│   └── migrations/          # 路径迁移记录
└── ablation_v9/
    ├── run_experiment.py
    ├── EXPERIMENT_SPEC.md
    ├── configs/
    ├── src/
    ├── tests/
    ├── audits/
    ├── registry/
    └── runs/
```

- 原流程入口：`experiments/original_400k/run_experiment.py`；使用方法见该目录 README。现有 400k 在续训，勿重复启动。
- v9 入口：`experiments/ablation_v9/run_experiment.py`；P0/P1/P2 共享实现，由 JSON 配置区分。P0 旧 run 已中断，详情见根目录实验记录。
- v9 工程规范：`experiments/ablation_v9/EXPERIMENT_SPEC.md`。
- 两条路线读取同一份 400k 数据，v9 的默认配置指向 `original_400k/data/`，不复制或重新生成数据。

原来散落在 experiments 根目录的脚本、数据、runs 和规范路径现为兼容符号链接，用于当前运行和历史快照；它们不是独立代码副本。迁移详情见 `experiments/original_400k/migrations/layout_20261110.json`。

开展工作前阅读 `AGENTS.md`、`研究问题和思路.md`、`方法设计.md` 和 `实验记录.md`。目录整理不改变实验协议，也不授权自动重启训练、生成数据或测评。
