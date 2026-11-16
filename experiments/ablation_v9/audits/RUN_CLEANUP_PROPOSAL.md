# runs与检查点清理提案（2026-11-16）

状态：用户已确认，已执行；见run_cleanup_execution_20261116.json。路径以下均相对于experiments/ablation_v9/runs。进程盘点未发现活动训练，执行前需再次核查。

| 目录/范围 | 建议 | 预计释放GiB |
|---|---|---:|
| p0_official_spike_diag_continue_20261114 | 删除整个目录 | 0.429 |
| p0_official_spike_diag_r3_20261114 | 删除整个目录 | 1.302 |
| p1_epoch24_replay_20261116_153436 | 删除整个目录 | 2.225 |
| p1_mask_normal_official_dp_20261115_015137 | 删除25份中间epoch权重 | 10.724 |

合计约 **14.68 GiB**。

新P1保留4份：best.pth（epoch22，验证最优）、epoch_0023.pth（主要事故前驱，含模型/Adam/完整RNG）、epoch_0024.pth（事故后，支持前后对比）、epoch_0027.pth（最后完整轮次）。epoch_0022.pth与best模型重复，拟删；epoch28未完成，没有轮末权重。仅保存epoch23仍不保证精确重演历史CUDA轨迹。

两个P0不再删：official保留best(epoch31)、epoch38、epoch0和16；ignore保留best(epoch26)、epoch33、epoch1和30。它们正是此前保留的最佳/最后/尖峰前驱。两个已完成eval目录原样保留。旧完整输入P1目录无权重、仅少量日志，暂保留。

三个整run删除会同时移除逐batch诊断、滚动现场、日志和配置快照；审计已有摘要保留，无法再使用这些run里的状态续训。原新P1的best/epoch23不受影响。

逐文件路径/大小/mtime/inode及保留列表见run_cleanup_proposal_20261116.json；执行时重新核查进程、文件一致性，并记录删除账本，更新registry和相关Agent文档。

CPU载荷核验已完成：best与epoch22模型逐张量一致；拟保留四份均含模型、Adam、Python/NumPy/Torch CPU/CUDA RNG，证据见run_cleanup_proposal_checkpoint_checks_20261116.json。epoch5、16存在曾恢复的暂时波动；本提案仅保留epoch24持续退化边界，早期权重仍待用户确认后才会删除。

执行结果：删除三整个诊断run与25份P1中间权重，释放约14.68GiB。原P0及评测资产、P1的best/23/24/27全部保留，逐文件摘要核验一致。
