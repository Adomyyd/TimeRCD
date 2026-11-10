# 400k 停止与 epoch 11 日志核查（2026-11-10）

用户明确要求停止400k训练并指出epoch11的loss达100+。先核实PID 408588命令为time-rcd/bin/python3.10 experiments/train_400k.py --resume .../pretrain_checkpoint_latest.pth，确认单卡400k续训身份后，在2026-11-10 22:12:18（Asia/Shanghai）仅向此PID发送SIGTERM。进程随后不存在，未使用SIGKILL。后续nvidia-smi核实GPU0为6MiB、0%利用率，GPU1为270MiB、0%。未启动或恢复任何训练。

停止时最后已打印epoch12 batch2100/23750。已有训练日志和检查点保留，未强制保存epoch12未完成状态。按日志及原save_checkpoint实现，latest对应已完成epoch11，best对应本次续训epoch10（val0.0678）；本次未加载或改写检查点验证内部元数据。

| 日志epoch（从0起） | 平均训练Total | Recon | Anomaly | 验证Total |
|---|---:|---:|---:|---:|
| 10 | 0.0609 | 0.0423 | 0.0186 | 0.0678 |
| 11 | 155.5050 | 155.4366 | 0.0685 | 0.0948 |

100+为epoch11全轮平均，主要贡献是重构MSE。该轮每100batch打印一次，共打印的238个batch中最大total=0.5330（batch400，recon0.4677、anomaly0.0653）；大幅异常不在已打印batch中。代码按每batch累加再除batch数，因此存在未被周期日志捕获的大loss贡献；日志无法确定其batch、数量、输入样本、预测或梯度。

这不是持续整轮所有batch都保持100+的证据；epoch12最后已打印batch2100的total0.1345、recon0.1075、anomaly0.0269。分类项本身没有出现同等量级增长，不能直接将尖峰归因于分类padding。分类梯度可能间接影响共享编码器，因此也不能据此排除其间接作用。原流程也出现尖峰，不能仅凭P0历史尖峰断定v9实现或mask大小有误。

进一步根因定位需要逐batch异常监测，并在触发时保存epoch/batch/样本索引、归一化统计、输入目标与预测幅值、分项loss和梯度范数；本次仅核查日志，未增加诊断代码、加载大数据或重跑。

证据：stop_and_log_evidence.json保存进程身份、信号、停止后状态、检查点目录状态和两份日志解析；resume_console.log:1952、train.log:12080为epoch11汇总。stop_and_inspect.py是本次PID限定停止/日志审计脚本，不是训练入口。
