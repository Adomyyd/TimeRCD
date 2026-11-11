# 当前代码与全文档核对（2026-11-11）

受众：Agent。用户要求按最新代码更新所有文档，重点排查已不存在的功能、入口与资产。当前范围仅文档；未修改代码/六配置/registry/数据/权重/run，未启动或停止训练、生成数据或真实测速。

## 核查范围与处理原则

遍历仓库Markdown、RST和文本清单，检查根四份Agent文档、各README、Tutorial、两路线运行/结构/差异/迁移说明、审计及归档。代码基于两统一入口、v9全部src、六JSON、测试、原流程src、公开推理API及生成器交叉核对。依赖清单保持代码对应版本；time_rcd.egg-info下文本是生成元数据，不作为手写文档改写。论文、官方源码快照及原始JSON/diff/XML不改写。

README保留直接运行命令和必要前提；Agent文档保留研究计划、历史失败、配置/调用链、实际状态与证据边界。归档及历史审计加当前性说明，不把旧实验条件重写成当前结果。

## 主要纠正

| 原表述或问题 | 按当前代码修订 |
|---|---|
| v9训练固定shuffle=False | 默认length_bucket，每轮重排池和池内；sequential才固定split后顺序；验证不分桶 |
| 当前默认batch16或不区分来源 | 六JSON总batch32、bucket_multiplier50；单卡命令统一--batch-size 16，双卡P0两模式均32；组间不能混用 |
| 入口列表缺测速/覆盖 | 补bench、--max-batches与--batch-size；明确max-batches不限制train |
| 每个run都有所有产物 | 按模式说明audit/smoke/train/bench/eval/report产物，数据准备后才写train.log |
| 未测默认规模吞吐/显存 | 已有短程摘要；全轮未验证，摘要外推和设备条件不视为通用结论 |
| 六组均尚未开始 | 已核实P0 official双卡run/config/status及进程；快照不能代表持续进度 |
| 旧P0和四个bench run仍可读取 | 本次未找到旧P0目录；bench仅存汇总，历史数值保留但不承诺资产仍可用 |
| 配置字段等于功能 | mixed_precision未启用AMP；threshold_source不拟合阈值；schema_version无具体版本校验；MODES漏bench但实际CLI支持 |
| 生成器旧导入/输出/长度说明 | 正确依赖、平级导入及工作目录、1000–10000、num_workers；400k使用统一入口，无data_generate/samples输出约定 |
| Tutorial称权重不会写Hub缓存 | 区分from_pretrained缓存与from_project本地查找；明确predict参数及不重叠窗口 |
| stop脚本与padding before仍在原目录 | 改为现有_archive映射；历史引用加日期边界 |

P3–P5、B/Z、偏离头、基元、少样本、外部VUS、密度干预、尖峰触发和v9 resume/权重初始化仍未实现，不删去其研究计划，也不写成现有功能。

## 最新状态证据与限制

初次只读快照中run为p0_official_dp_20261111_003608，start为2026-11-11本地00:36，配置为双卡DataParallel、batch32、length_bucket50、seed72、16/16/0.15；对应进程PID821206。核查时train.log未出现，不推断已完成batch/epoch；后续状态必须重新查日志/进程。

audits/bench_20261111/bench_summary.json由他次操作保存，声明四个原run已移除。单卡b16/双卡b16/双卡b32分别记录0.1612/0.1496/0.2756秒每批，单卡b32 OOM。已知计时/尾批/预热/多卡峰值及长度池外推问题仍未修复；没有重新测速，不能追认摘要中的整轮小时或2.8倍为完整实测。旧P0缺失原因与其他备份位置未知，本次未清理任何用户资产。

## 验证

文档部署前比较原文哈希，避免覆盖并发修改；部署后回读校验。对当前说明中的Python代码块做AST语法检查，对bash块做bash -n，对两个入口命令按实际argparse定义解析；六JSON及源数据路径核对，配对仅padding字段不同。当前文档本地链接/入口/源码引用按目录清单复核；未来run/checkpoint与历史来源标注为条件或历史引用。所有共享Markdown扫描机器绑定名称和本机规范路径。

精确检查计数、命令解析、引用与保护校验见verification.json；文档改动摘要见changes.diff和source_manifest.json。此处验证不包含正式训练、模型数值回归或完整数据读取；旧测试结果保留其日期范围。
