# 原流程400k复现

使用原TimeRCD模型和损失，在400k条单变量合成序列上训练，并与官方权重比较检测结果。与v9共用数据，训练入口相互独立。

所有命令从仓库根目录执行，使用已激活项目环境中的`python`。

## 查看入口

```bash
python experiments/original_400k/run_experiment.py --help
```

## 生成数据

仅在没有已有数据、确实需要生成新数据时执行；将生成约39GB文件，目标已存在时会拒绝覆盖：

```bash
python -m pip install -r dataset_gen/requirements.txt
python experiments/original_400k/run_experiment.py --mode generate
```

数据和manifest保存到`experiments/original_400k/data/`。迁移已有实验时直接复制原数据及manifest，避免重新生成不同的数据。

## 恢复训练

**当前`train`模式用于断点恢复。** 先将原run检查点放回`experiments/original_400k/runs/time_rcd_400k_20261108/checkpoints/`，其中需有`pretrain_checkpoint_latest.pth`：

```bash
python experiments/original_400k/run_experiment.py --mode train
```

默认batch为16、patch和遮挡块均为16、遮挡比例15%。当前入口固定使用物理GPU0，会覆盖外部`CUDA_VISIBLE_DEVICES`设置；缺少检查点时不能从头训练。恢复不包含完整随机数和早停历史。

## 对比检测结果

准备训练所得best检查点、`best_model/pretrain_checkpoint_best_uni.pth`，以及`datasets/TSB-AD-U/`中的外部评测数据后运行：

```bash
python -m pip install pandas scipy statsmodels scikit-learn matplotlib seaborn tqdm
python experiments/original_400k/run_experiment.py --mode compare
```

训练日志和检查点保存在`experiments/original_400k/runs/time_rcd_400k_20261108/`，对比指标保存在其`comparison/`目录。数据、权重和检查点需单独准备，不随Git下载。
