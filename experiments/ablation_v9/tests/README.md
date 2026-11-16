# 回归测试

从仓库根目录、项目环境执行；测试使用临时合成数据：

```bash
python experiments/ablation_v9/tests/run_checks.py
```

需要双卡或 CUDA 的检查在条件不满足时跳过；完整模型检查会占用较多显存，避免与正式训练同时执行。

| 文件 | 作用 |
|---|---|
| test_config_contract.py、test_patch_parity.py | 当前配置与官方 patch/遮挡对齐 |
| test_padding_contract.py、test_parallel_contract.py | 两种 padding、全 batch 损失、多卡与检查点 |
| test_p1_normal_reference.py | 新 P1 的监督、梯度、初始化和激活重算 |
| test_bucket_sampler.py、test_bench_contract.py | 分桶与测速入口行为 |
| test_benchmark_eval.py | 官方推理/指标、双域评测、进程隔离与 NaN |
| test_diagnostics.py、test_diagnostic_detail.py | 观测不改变更新/RNG、分频采集与事件现场 |
| test_resume.py | 完整状态恢复和权重保留策略 |

`conftest.py` 设置导入路径；`run_checks.py` 仅启动此测试套件。

测试依赖的三处历史源码见 `../audits/README.md`，迁移时须一并保留。`test_loss_contract.py` 和 `test_sampling_contract.py` 已于2026-11-16合并去重，覆盖映射见 `../audits/cleanup_20261116.json`。退役 P1/P2 名称的拒绝测试仍有用，用于防止旧配置被误用。
