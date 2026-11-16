# P1 epoch24及后续诊断训练（2026-11-16）

用户明确授权重新训练epoch24及其之后。原事故训练保留best(epoch22)、epoch23/24/27；此前诊断run已由用户确认删除，汇总与删除证据见上级run_cleanup_execution_20261116.json。

新run：p1_epoch24_continue_20261116_210441；tmux：p1_epoch24_continue。从原P1 epoch23模型/Adam/完整RNG恢复，best22/stale1不重置，总上限50（epoch0–49）、patience7。双卡DataParallel/总batch32/official/seed72/length_bucket50/patch16-mask16-ratio0.15/FP32/lr5e-4及任务权重保持。未加梯度裁剪/新归一化/学习率修正。

每批轻量诊断，详细间隔100，滚动250，最多6个有限升级事件；非有限现场后停止。有限异常保存后继续。新run权重保留best、最近两轮和首次事件前驱；源run不受该策略影响。完整检测评测未授权。恢复不保证历史CUDA轨迹逐位一致。

以下为统一入口命令，从仓库根目录、项目环境执行；已启动后不可直接重复执行同一run-id：

```bash
python -u experiments/ablation_v9/run_experiment.py --config experiments/ablation_v9/runs/p1_mask_normal_official_dp_20261115_015137/config.json --resume-from experiments/ablation_v9/runs/p1_mask_normal_official_dp_20261115_015137/checkpoints/epoch_0023.pth --run-id p1_epoch24_continue_20261116_210441 --epochs 50 --log-freq 100 --mode train --diagnostics --diagnostic-detailed --diagnostic-detail-interval 100 --diagnostic-start-epoch 24 --diagnostic-loss-threshold 1 --diagnostic-grad-threshold 1000 --diagnostic-snapshot-interval 250 --diagnostic-max-events 6 --checkpoint-keep-last 2
```

启动/首批状态见replay_continue_launch_20261116.json与对应run日志，启动不等于训练完成。


启动核验（2026-11-16T21:10:35+08:00）：统一入口已完成229.3秒数据加载，恢复next_epoch24/best22/stale1，实际执行epoch24至batch302/11875。batch100总loss0.1304（观测0.0305、正常参考0.0411、分类0.0589）；检查时303条batch记录、303条更新记录，阈值触发0条、更新后非有限状态记录0条。batch0/100为detailed，batch1为light，初始rolling快照存在；源代码摘要与启动时一致。当前为运行中，后续沿用50轮总上限和patience7，不承诺一定运行到epoch49或重现历史事故。详细核验见replay_continue_launch_20261116.json。
