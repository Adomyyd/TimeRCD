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
| v9历史检查点与日志 | experiments/ablation_v9/runs/ | eval可加载完整模型，当前入口不支持resume |
| 官方单变量/多变量权重 | best_model/ | 官方推理或原流程对比 |
| 原流程外部检测数据 | datasets/TSB-AD-U/ | 仅compare需要，smoke不需要 |

历史符号链接和历史JSON中的位置只是原环境记录。新机器使用上表真实目录；不要依赖旧链接，也不要为迁移修改历史配置快照或重生成一份不同的400k数据。manifest中的旧位置字段不影响数据内容身份，应结合文件大小和已有校验值核对。大型文件校验按实际需要单独执行。

## v9本机配置

仓库中的六份JSON保留既有实验设置与数据定位。下面从它们生成六份本机副本，**只替换两个数据定位字段**，不覆盖原文件，不改变实验超参。配置放在已被Git忽略的`outputs/`中；运行时入口会另存实际配置快照。

先把数据和manifest放到上表目录，再运行以下Linux shell示例。若修改存放位置，仅调整`data_dir`，并对六组保持一致：

```bash
python - <<'PY'
import json
from pathlib import Path

source_dir = Path("experiments/ablation_v9/configs")
output_dir = Path("outputs/portable_configs")
data_dir = Path("experiments/original_400k/data")
stem = "dataset_400000_samples_Nonelen_1.0ratio_univariate_attr_set"
data_path = data_dir / (stem + ".pkl")
manifest_path = data_dir / (stem + ".manifest.json")
assert data_path.is_file() and manifest_path.is_file(), "先准备数据与manifest"
sources = sorted(source_dir.glob("*.json"))
assert len(sources) == 6, "应为P0/P1/P2各两份padding配置"
output_dir.mkdir(parents=True, exist_ok=True)
for source in sources:
    config = json.loads(source.read_text(encoding="utf-8"))
    config["data"]["path"] = data_path.as_posix()
    config["data"]["manifest_path"] = manifest_path.as_posix()
    target = output_dir / source.name
    if target.exists():
        raise FileExistsError(f"已有本机配置，先核对：{target}")
    target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)
PY
```

当前配置读取器将相对数据路径解释为**进程工作目录**下的位置，不是配置文件所在目录，因此上述配置和后续命令均从仓库根目录使用。无需改变配置加载或训练代码。

## 先检查，再运行

无需真实训练数据的合成回归：

```bash
python experiments/ablation_v9/tests/run_checks.py
python experiments/original_400k/run_experiment.py --help
python experiments/ablation_v9/run_experiment.py --help
```

只有一张可见CUDA卡时，真实双GPU测试会跳过；其余通过不代表已验证本机双卡。以下smoke会核对数据文件和manifest，但不加载完整400k pickle；run-id必须未使用过：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/ablation_v9/run_experiment.py \
  --config outputs/portable_configs/p0_mask_x.json \
  --run-id p0_portable_seed072_smoke --mode smoke
```

确认检查通过后，由操作者按v9 README选择正式命令、独立run-id与相应本机配置。双卡仍通过同一入口增加`--parallel data_parallel --gpu-ids 0 1`，同时设置`CUDA_VISIBLE_DEVICES=0,1`；batch_size是总batch。更换设备不保证dropout、浮点计算与训练轨迹逐位相同；对照组应固定设备、软件、并行方式和预算。

## 原流程恢复

原流程根据脚本位置定位仓库、数据和run目录，不依赖原挂载位置。将原run及检查点放回上表位置后，以下命令按既有默认latest恢复；它会开始正式训练，不能当作环境测试：

```bash
python experiments/original_400k/run_experiment.py --mode train
```

当前原流程入口固定使用本机物理GPU0，并会覆盖外部可见卡设置；此限制与GPU型号无关。恢复不包含完整RNG/早停历史，不能承诺逐位等价。v9没有恢复训练接口：复制权重后可按匹配配置执行eval，不能用原流程入口恢复v9模型。

## 发布与历史证据

共享文档使用相对路径；机器连接、解释器安装位置等由各自环境管理。原始日志、JSON、diff和XML是历史证据，可能带有当时的环境元数据，不能通过改写它们伪造为另一台机器的结果。本次仅整理文档，未改写这些原始产物或六份源配置。文档版审计报告及归档说明已做路径整理，旧文档摘要对应旧版本。
