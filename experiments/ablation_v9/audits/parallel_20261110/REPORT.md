# P0/P1/P2 可选多 GPU 验证

> 迁移整理注（2026-11-10）：本页已统一路径表示与环境表述，实验数值和结论保留。旧清单中的文档摘要对应整理前版本；历史命令不作为当前用法，当前入口见两路线README。

日期：2026-11-10。实现：单机、单进程 torch.nn.DataParallel；默认单卡。

## 设计与文件

新增 src/parallel.py 管理校验/设备/封装/解封装，新增 tests/test_parallel_contract.py 和 tests/run_checks.py。入口增加 --parallel 与 --gpu-ids，配置增加 runtime.parallel/gpu_ids，训练访问属性和保存权重时解封装。具体逐行差异见 changes.diff，12 个发布文件哈希及读回一致性见 source_manifest.json。

全 batch 标准化/padding/采样后才分卡，预测汇总后按原有效点分母计算损失。总 batch_size=16、数据顺序、MSE/CE、学习率及更新步数语义保持一致。模型/数据/采样/损失/评估算法文件未修改；make_batch_inputs AST 与修改前一致；三个配置除新增默认 none/null 外其余值逐项一致。检查点保存未封装 state_dict，兼容单卡/多卡 eval。

## 实际验证

- 本地 CPU：37 passed, 3 skipped；跳过的是双 CUDA GPU 测试。
- 实验服务器，time-rcd 环境 Python 3.10.22：最终 40 passed in 8.58s，无跳过。报告 pytest_remote.xml。
- 覆盖 P0/P1/P2 真正双 GPU 小模型的输出、全局 loss/梯度与单 GPU 对照（关闭 dropout 比较并使用浮点容差）、优化器更新、尾 batch 只有一条样本的反向、开启 dropout 的 smoke；合成小数据上的训练/验证/权重保存加载；配置错误/旧配置兼容/CLI 覆盖；默认规模模型的 CPU smoke 和原有 patch 回归。
- 首次远程有一项测试构造 CPU DataParallel 时自动选中 CUDA 设备而失败，修复测试中的 CPU fallback 构造后全量通过。生产设备选择逻辑没有该问题。
- 入口 --help 在远程通过。12 个发布文件 SFTP 读回 SHA 对应内容完全相同，语法检查通过。

命令：

```bash
python experiments/ablation_v9/tests/run_checks.py --junitxml experiments/ablation_v9/audits/parallel_20261110/pytest_remote.xml
```

## 边界

未启动 400k 正式训练或耗时评估；没有测试正式 batch=16 的双卡峰值显存/吞吐。真实双卡检查使用缩小模型和合成变长数据，原默认规模模型做 CPU smoke。单卡/多卡 dropout 随机流和浮点归约顺序不同，不能保证训练轨迹逐位相同。当前不增加恢复训练接口。

执行前 GPU0 为既有任务使用，GPU1 空闲；仅做短小合成验证，未停止/重启既有训练。

## 执行格式更正

此前某次远程命令格式与连接工具规则不匹配，不能概括为无法运行Python；后续已在项目环境成功完成测试，未修改连接配置。当前可复核命令见上文，本机连接细节不作为跨机器要求。

> 当前性说明（2026-11-11）：本文保留该次核查的历史事实，旧batch、顺序、配置准备及资产路径不能直接作为当前协议。当前代码语义见experiments/ablation_v9/RUNTIME_NOTES.md；旧P0和测速run缺失、当前训练启动及文档纠正见根实验记录第12节。原始JSON/diff/XML未改写。
