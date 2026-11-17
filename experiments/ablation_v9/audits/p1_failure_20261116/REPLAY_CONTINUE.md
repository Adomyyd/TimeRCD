# P1 epoch24及后续诊断训练（2026-11-16）

> 当前状态（2026-11-17）：用户要求停止，11:03:36已退出；最后epoch32/b11450，完整轮至31。后续不得自动重启。异常分析见ANALYSIS_20261117.md；下文启动命令仅为历史记录，不能复用已存在run-id。

用户误中断上一轮续训后，明确要求删除本次记录并重新开始。旧run p1_epoch24_continue_20261116_210441及对应控制台、启动记录已清理；仅保留最小删除状态，避免把已删除产物当作可用。

当前run：p1_epoch24_continue_20261116_211434；tmux：p1_epoch24_continue。从原P1 p1_mask_normal_official_dp_20261115_015137/checkpoints/epoch_0023.pth恢复完整模型/AdamW/RNG，next_epoch24、best22、stale1；总上限50（epoch0–49），patience7。双卡DataParallel、总batch32、official、seed72、分桶50、patch16/遮挡16/比例0.15、FP32、lr5e-4保持。

逐batch轻量诊断、详细间隔100、滚动250、最多6有限升级事件；有限异常保存后继续，非有限保存后停止。新run仅保留best/最近两轮/首次事件前驱。未启动检测评测，不保证历史CUDA轨迹逐位一致。

以下命令从仓库根目录及项目环境执行；运行已启动时不可重复执行：

```bash
python -u experiments/ablation_v9/run_experiment.py --config experiments/ablation_v9/runs/p1_mask_normal_official_dp_20261115_015137/config.json --resume-from experiments/ablation_v9/runs/p1_mask_normal_official_dp_20261115_015137/checkpoints/epoch_0023.pth --run-id p1_epoch24_continue_20261116_211434 --epochs 50 --log-freq 100 --mode train --diagnostics --diagnostic-detailed --diagnostic-detail-interval 100 --diagnostic-start-epoch 24 --diagnostic-loss-threshold 1 --diagnostic-grad-threshold 1000 --diagnostic-snapshot-interval 250 --diagnostic-max-events 6 --checkpoint-keep-last 2
```

启动阶段需加载400k数据，实际恢复/首批核验见replay_continue_launch_20261116.json及run日志。查看tmux后使用Ctrl+b再按d离开会话；Ctrl+c会中断训练。

启动核验（2026-11-16T13:19:10.196810+00:00）：实际恢复epoch23模型/Adam/RNG，next_epoch24、best22、stale1。已完成epoch24至batch115/11875，最新总loss=0.21202469；116条batch诊断/116条更新诊断，阈值触发0条，更新后非有限0条。batch0/100详细、其余批轻量（滚动批次例外），rolling现场存在。训练继续运行，原50轮总上限/patience7保留；源配置和代码摘要核验通过。该记录只代表启动核查时点，后续以日志为准。
