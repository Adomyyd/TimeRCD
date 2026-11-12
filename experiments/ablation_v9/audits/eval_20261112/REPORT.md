# 统一检测评测实现核查（2026-11-12）

受众：Agent。实现已完成；本次未执行全量合成/真实检测评测，不报告检测增益。所有检查仅CPU，正在训练的P0 ignore未被操作；训练代码和六训练JSON未改。

## 官方来源

固定官方仓库thu-sail-lab/Time-RCD@d458549f6962029a5c5241c43a5baa1ec5eccbb2，是轻量推理包替换前最后保留main.py的版本。当前官方main为轻量包，原基准脚本已移除。来源链接：https://github.com/thu-sail-lab/Time-RCD/tree/d458549f6962029a5c5241c43a5baa1ec5eccbb2 。

sources保留官方main、HP_list、TimeRCD tester、dataset、slidingWindows、metrics/basic_metrics七份原始文件，SHA256见verification.json；其中数份被tests/test_benchmark_eval.py读取，是回归依赖。当前本项目main.py、HP_list.py、utils/dataset.py、utils/slidingWindows.py逐字节相同，affiliation全部六文件也与固定官方版本哈希一致。models/TimeRCD.py仅包导入路径不同。根metrics/basic_metrics有两处历史修正：F1_T串行避免嵌套线程池、串行传入同一n_splits，本次未改它们；800分位点的16指标固定样例与原始官方并行路径数值对照通过，未以此声称任意输入逐位等价。

## 实现与对齐

新增configs/eval_official.json、src/benchmark_eval.py、requirements-eval.txt、tests/test_benchmark_eval.py；唯一执行入口仍为run_experiment.py。test_config_contract.py改为只枚举六训练JSON，避免误把独立评测协议当训练配置。eval/all调用新检测评测；evaluate.py及train.py保留训练验证、区域误差和valid_total选best。eval检查checkpoint所存实验/model/input/loss/split/数据路径与传入配置一致，防止P组或padding混用；权重CPU加载、严格state_dict恢复，再独立设置评测seed2024/CPU线程1。

真实清单：官方单变量排除表；11类700条，IOPS17、MGAB9、NAB28、NEK9、Power1、SED3、Stock20、TODS15、UCR228、WSD111、YAHOO259。清单缺失/意外数量拒绝继续，不无声跳过。先dropna再读文件名训练边界，测试后缀输入，ACF周期来自完整清理后的序列。

合成使用与训练相同95/5 seed42的留出索引，完整原始X逐条输入，不读取X0作检测输入；它已用于选best，不称独立测试集。两侧复用TimeRCDDataset，整个测试序列每通道numpy总体std标准化，5000点非重叠窗口，短序列窗口缩短，尾部重复末值且mask=false，展开后截回真实长度。P0评测不做训练遮蔽；仅执行encoder及anomaly_head。概率softmax(mean_channel_logits)[...,1]与logit=l1-l0分别计算16项官方指标，不拼接序列、不降采样、不改变事件边界。

默认fast、F1_T 800分位点/chunk25、VUS opt/250阈值、worker4/4；保留官方mean+3std传pred的形式及各指标oracle搜索行为。官方内部部分异常返0予以披露，不在外层伪造替代值；缺字段/非有限数报失败。默认推理batch64，与训练batch独立；参数变化记录official_default_parameters=false。

## 产物与完整性

新eval run的eval_config.json/evaluation_manifest.json记录生效参数、checkpoint epoch/哈希、源码/依赖版本、逐CSV哈希、留出索引及哈希。metrics/per_series.csv逐行写入；group_mean_metrics.csv、domain_summary.csv和eval.json区分序列宏平均与组等权宏平均，合成与真实分开。官方main只输出逐序列CSV，本项目新增的汇总并非官方脚本已有行为。scores/*.npz默认保存，eval.log记录阶段。失败保留已完成行/分数并抛出，入口标failed，无成功eval.json；不支持评测resume。

默认全量20000条合成+700条真实，逐序列两类分数，计算量远大于训练验证。--eval-max-synthetic/--eval-max-real只截稳定顺序前N条，full_benchmark=false；完整pickle仍会读取。未为本次验证启动任何完整或子集生产评测。

## 验证

最终CPU回归：87 passed、6 skipped，9.25秒；跳过项均需GPU。新增11项覆盖官方短/整/尾窗推理逐元素相同、全部16指标与固定官方源码数值相同（绝对容差1e-12，主要指标非0）、官方选择/HP、完整清单/缺失、dropna/边界、配置错配、留出split、统一入口双域、组汇总与中途失败保留。流水线测试用小型临时合成文件及mock指标验证编排，指标数值另用真实官方函数对照；不冒充400k或真实基准测试。

17项既有src/六训练配置哈希未变；实际P0 official的best.pth仅CPU加载并严格匹配37056387参数，checkpoint epoch31，无推理。README七段bash语法与入口help通过，真实700条清单核对通过。详细布尔项及环境版本见verification.json。没有加载400k pickle、评测训练GPU或改写任何既有run/registry；测试入口产物位于临时测试目录。P0 ignore快照与配对配置检查见根实验记录第14节。
