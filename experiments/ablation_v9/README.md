# v9 P0/P1/P2 消融实验

本目录实现根目录研究文档中登记的最小消融协议。相对原始 TimeRCD 的逐项代码差异见 `IMPLEMENTATION_DIFF.md`。先阅读：

- `/diskA/YYD/TimeRCD/AGENTS.md`
- `/diskA/YYD/TimeRCD/研究问题和思路.md`
- `/diskA/YYD/TimeRCD/方法设计.md`
- `/diskA/YYD/TimeRCD/实验记录.md`
- `/diskA/YYD/TimeRCD/experiments/EXPERIMENT_SPEC.md`

三个配置对应：

- `p0_mask_x.json`：原始输入遮蔽，选中位置重构观测 `X`；
- `p1_full_x.json`：完整输入，选中位置重构观测 `X`；
- `p2_full_x0.json`：完整输入，选中位置重构配对正常值 `X^0`。

统一入口：

```bash
python /diskA/YYD/TimeRCD/experiments/ablation_v9/run_experiment.py \
  --config /diskA/YYD/TimeRCD/experiments/ablation_v9/configs/p2_full_x0.json \
  --run-id p2_full_x0_seed072_audit \
  --mode audit
```

`audit` 和 `smoke` 只做配置、结构和小批量检查；`train`、`eval` 和 `all` 会读取 400k 数据快照并可能消耗大量时间和内存，未经明确安排不要执行。


静态检查：

```bash
python -m pytest /diskA/YYD/TimeRCD/experiments/ablation_v9/tests
```

该测试只检查配置、采样、padding 标签过滤和零样本损失边界，不读取 400k 数据。
