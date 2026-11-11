# 跨机器实验迁移

受众：Agent。保存跨机器迁移约束、资产定位与配置核查步骤；面向用户的完整准备及执行命令集中在两路线README中，不要求用户阅读本文才能操作。

本文所有文件位置均相对于仓库根目录，命令也在该目录执行。使用目标机器已激活的项目环境，示例中的`python`指该环境的解释器。GPU按目标机器的可见编号选择，不绑定型号或主机名。

## 环境与资产

建议使用Python 3.10及与目标驱动兼容的PyTorch/CUDA；安装方法见根README与Tutorial。迁移后记录实际软件版本与GPU环境，不能将另一台机器上的测试通过视为本机验证完成。

Git代码仓库不包含训练数据、官方权重和完整检查点，须通过自己的文件传输方式单独准备：

| 资产 | 仓库相对位置 | 用途 |
|---|---|---|
| 400k数据及同名manifest | experiments/original_400k/data/ | 两路线只读共用；约39GB，避免重复复制 |
| 原流程续训检查点 | experiments/original_400k/runs/time_rcd_400k_20261108/checkpoints/ | 保留latest/best及其原始日志、配置 |
| v9现有检查点与日志 | experiments/ablation_v9/runs/ | 只迁移实际存在的run；旧P0目录本次未找到，不能假设可恢复；eval可加载完整模型，无resume |
| 官方单变量/多变量权重 | best_model/ | 官方推理或原流程对比 |
| 原流程外部检测数据 | datasets/TSB-AD-U/ | 仅compare需要，smoke不需要 |

历史符号链接和历史JSON中的位置只是原环境记录。新机器使用上表真实目录；不要依赖旧链接，也不要为迁移修改历史配置快照或重生成一份不同的400k数据。manifest中的旧位置字段不影响数据内容身份，应结合文件大小和已有校验值核对。大型文件校验按实际需要单独执行。

## v9配置与路径

直接使用`experiments/ablation_v9/configs/`中的六份JSON，不生成本机副本。源配置的数据和manifest字段使用仓库相对路径，数据准备在上表目录即可跨机器使用。

配置读取器将相对数据路径解释为**进程工作目录**下的位置，而非配置文件所在目录，所有命令必须从仓库根目录执行。数据确需另存时，对相关组统一修改`data.path`和`data.manifest_path`，保留实验超参一致；优先沿用约定目录。

每次运行会在独立run下保存实际配置快照。当前源配置的修订不会改写已有run，旧快照中的历史位置保留原文。

## 先检查，再运行

先安装项目与合成检查依赖，再运行无需真实训练数据的回归：

```bash
python -m pip install . pytest
python experiments/ablation_v9/tests/run_checks.py
python experiments/original_400k/run_experiment.py --help
python experiments/ablation_v9/run_experiment.py --help
```

只有一张可见CUDA卡时，真实双GPU测试会跳过；其余通过不代表已验证本机双卡。以下smoke会核对数据文件和manifest，但不加载完整400k pickle；run-id必须未使用过：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config experiments/ablation_v9/configs/p0_mask_x.json \
  --run-id p0_portable_seed072_smoke --mode smoke
```

确认检查通过后，由操作者按v9 README选择正式命令、独立run-id与相应源配置。双卡仍通过同一入口增加`--parallel data_parallel --gpu-ids 0 1`，同时设置`CUDA_VISIBLE_DEVICES=0,1`；batch_size是总batch。更换设备不保证dropout、浮点计算与训练轨迹逐位相同；对照组应固定设备、软件、并行方式和预算。

## 原流程恢复

原流程根据脚本位置定位仓库、数据和run目录，不依赖原挂载位置。将原run及检查点放回上表位置后，以下命令按既有默认latest恢复；它会开始正式训练，不能当作环境测试：

```bash
python experiments/original_400k/run_experiment.py --mode train
```

当前原流程入口固定使用本机物理GPU0，并会覆盖外部可见卡设置；此限制与GPU型号无关。恢复不包含完整RNG/早停历史，不能承诺逐位等价。v9没有恢复训练接口：复制权重后可按匹配配置执行eval，不能用原流程入口恢复v9模型。

## 发布与历史证据

共享文档使用相对路径；机器连接、解释器安装位置等由各自环境管理。原始日志、JSON、diff和XML是历史证据，可能带有当时的环境元数据，不能通过改写它们伪造为另一台机器的结果。原始产物保持原文；当前六份源配置的数据字段已改为仓库相对位置，其他设置保持不变。文档版审计报告及归档说明已做路径整理，旧文档摘要对应旧版本。

## 当前配置能力与资产核查

源JSON总batch32、length_bucket50；单卡README例子显式覆盖batch16，双卡保持32。迁移不自动调整batch/采样器，不将不同设置混成配对结果。mixed_precision字段目前不启用AMP；threshold_source不执行阈值拟合。bench会读完整数据并进行优化器更新，不能代替smoke。短程测速摘要、旧run缺失和最新训练启动事实见根实验记录第12节。

原流程compare额外依赖pandas、scipy、statsmodels、scikit-learn、matplotlib、seaborn和tqdm；仅pip install .不足以覆盖这些实验依赖。生成器依赖见dataset_gen/requirements.txt。原流程CLI自定义--resume要求本机规范化后的检查点位置；默认latest命令无需额外路径。不要把历史源脚本的路径限制迁移到v9的相对配置用法。
