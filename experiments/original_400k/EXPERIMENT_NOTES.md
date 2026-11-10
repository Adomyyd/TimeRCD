# 原流程400k：Agent实验记忆

受众：Agent。保留原流程结构、历史状态、完整配置与续训限制；用户操作集中在本目录README。以下内容来自README精简前的工程记录，不是新的执行授权。

更新时间：2026-11-10。此路线保存原训练流程的本地400k复现，不代表完整复刻官方语料和预算。与ablation_v9并列；v9只读共用本目录data，不复制数据。

## 文件职责

| 路径 | 职责 |
|---|---|
| run_experiment.py | 唯一工作流入口：generate/train/compare |
| src/generate_400k.py | 生成400k数据及manifest，已有数据时拒绝覆盖 |
| src/train_400k.py | 原流程续训配置，调用仓库根training.py |
| src/compare_400k.py | 本地best与官方单变量权重的TSB-AD比较 |
| data/ | 已有单变量数据及manifest，只读共用 |
| runs/time_rcd_400k_20261108/ | 日志、配置快照、检查点；保留原内容 |
| audits/stop_20261110/ | 停止身份、日志解析和epoch11尖峰核查 |

旧migrations备份已归档至`experiments/_archive/20261110_cleanup/original_400k_migrations/`，不参与训练。一次性停止脚本位于同次归档one_off_scripts，不作常规工具执行。

## 当前运行状态与协议

最初从头训练，后从本run断点恢复。2026-11-10 22:12按用户要求停止PID408588，最后打印epoch12 batch2100/23750；epoch12未完成部分没有保存，不自动恢复。

原代码与官方padding分类行为保留。模型patch16、遮挡块16、mask_ratio0.15，batch16、lr5e-4、weight_decay1e-5、seed72、FP32、最多50轮、patience7、workers0、log_freq100、save_freq5、物理GPU0。模型/损失与参数实现仍读取根training.py；训练不是v9的P0。

epoch11平均train total155.5050，主要来自recon155.4366；classification0.0685，validation total0.0948。周期batch日志未捕获大幅异常，根因未明，不能直接归因于padding。完整事实见根实验记录及本目录审计。

## 命令参考（不会自动执行）

下列命令从仓库根目录执行，并使用已激活项目环境中的`python`。数据与检查点需按[迁移说明](../MIGRATION.md)单独复制；不依赖旧兼容链接。

只查看入口帮助：

```bash
python experiments/original_400k/run_experiment.py --help
```

以后用户决定恢复同一run时，将检查点放回既有run目录，使用默认latest定位：

```bash
python \
  experiments/original_400k/run_experiment.py \
  --mode train
```

当前train模式默认从latest恢复，缺失会报错；它不是新的“从头训练”接口。恢复模型/优化器并从下一epoch开始，但不恢复完整RNG和早停历史，不保证逐位等价。按现有日志/保存逻辑latest为epoch11、best为本次续训epoch10（val0.0678），本次未重新加载权重核内部元数据。入口固定GPU0，外部设置CUDA_VISIBLE_DEVICES会被原脚本覆盖。

generate、compare不会与训练自动串联，均需明确任务授权再执行。历史数据/manifest/runs链接保留，新命令使用仓库相对路径。自定义恢复位置时，优先将检查点复制到默认目录；入口仍有既有路径校验，本文示例使用无需额外参数的默认恢复方式。
