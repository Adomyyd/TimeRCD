# 跨机器实验迁移

命令从仓库根目录执行。使用目标机器已有的项目环境，推荐 Python 3.10 和与驱动兼容的 PyTorch/CUDA。面向读者的安装和命令见[实验 README](README.md)。

| 资产 | 仓库相对位置 |
|---|---|
| 共用 400k 数据及同名 manifest，约 39 GB | `experiments/original_400k/data/` |
| 原流程 run、检查点及原日志 | `experiments/original_400k/runs/time_rcd_400k_20261108/` |
| v9 运行快照、轮末检查点及日志 | `experiments/ablation_v9/runs/` |
| 官方单变量、多变量权重 | `best_model/` |
| 正式评测数据 | `datasets/TSB-AD-U/`，完整 700 条、11 组 |

数据、权重和运行产物需单独传输。使用真实目录并按 manifest 核对文件身份。原始配置快照和日志保留历史位置及环境信息。v9 源配置中的数据路径相对于仓库根目录，直接使用已有 JSON。若改变数据位置，对各组统一修改源配置中的 `data.path` 和 `data.manifest_path`；为既有权重做评测时先核对匹配配置和检查点的数据身份。

迁移代码时保留 [三处审计源码依赖](ablation_v9/audits/README.md)。先运行[回归检查及 smoke](ablation_v9/README.md#准备与检查)。双卡测试跳过时需另行验证目标机器的并行能力。更改设备或训练 batch 后，按同一条件重做配对。

v9 可用 `--resume-from` 从含完整训练状态的轮末检查点恢复到新 run，按原配置、数据与并行条件恢复。历史缺 RNG 权重可评测，其训练轨迹无法精确续接。原流程恢复复用原 run，默认从 latest 继续，当前入口固定使用物理 GPU 0，旧续训缺完整 RNG 和早停状态。两条路线分别使用自己的入口和检查点。[v9 续训](ablation_v9/README.md#续训)、[原流程恢复](original_400k/README.md#恢复训练)
