# P0 loss尖峰代码分析（2026-11-15）

受众：Agent。此轮只读取代码和日志并记录分析，没有修改训练算法、配置或正在运行的进程。原始统计见code_analysis_20261115.json；快照时间为2026-11-15 00:18北京时间。

## 已确认事实

- r3已完成epoch0/1，续跑已实际从epoch2开始。epoch1共11875批，mean total=0.1481352032/reconstruction=0.0660134509/classification=0.0821217523；max total=0.4462524652；max gradient L2=29.7629814；无阈值触发或非有限值。
- 快照时epoch2完成5141批（最后batch5140），mean total=0.1278611471；max total=0.3028956652；无阈值触发或非有限值。仅为部分轮次快照。
- 旧official epoch1轮均值total=2908.18585937/reconstruction=2908.0352/classification=0.1507，epoch17为1.52080747/1.3985/0.1223。
- 旧ignore epoch31轮均值total=6.35217318/reconstruction=6.2829/classification=0.0692。epoch2 batch100已打印reconstruction=8.3409/classification=0.1257。
- 旧日志每100批采样，不能排除未打印batch的分类尖峰；不能说分类项完全不受影响。official epoch2平均classification升至0.2132。
- 当前r3 epoch0 batch4 classification=1.9881066，大于重建项1.2224997；初期有限波动不是数千量级事故，也直接反驳CE一定小于1的说法。

## 代码事实与推导

1. src/data.py:79-82对batch全部有效观测X逐通道统一标准化，s为样本标准差，分母s+1e-4；padding在标准化后加入。P0目标为同一标准化X。
2. 对单变量有限精度正常的batch，令有效点N，选中点K，则全体标准化目标能量sum(x²)=(N-1)s²/(s+eps)²<=N-1；选中目标均方<=N/K。当前epoch1实际最大N/K=7.3743549（规则floor和尾patch使其不恰等于1/0.15）。
3. 所以零预测的P0 masked MSE约束在N/K以内；个别极值可以存在，但单靠目标幅值/小标准差不能解释数千的batch平均MSE。若loss实现和数值计算正常，巨大MSE需要预测端的大误差。可用RMS(pred)>=sqrt(MSE)-RMS(target)进一步约束。
4. 上述上界针对P0/P1的X目标；P2的X0使用X的统计，不能直接套用。不能由此排除输入形态触发模型不稳定，也不能当作历史运行逐batch数值有限性的完整证明。
5. src/losses.py:15-21在mask选中位置做square().mean()；非空分母是实际元素数，没有除近零浮点数；空mask有保护。classification_ce使用PyTorch cross_entropy(logits,labels)，没有先softmax再CE。
6. time_rcd/_core/ts_encoder_bi_bias.py:197-204为pre-RMSNorm残差层；MLP:178包含GELU(gate(x))*up(x)乘法门。各层归一化的是分支输入，不约束残差和幅值；最后一层输出直接经projection_layer到H，没有final RMSNorm。
7. src/models.py:25-33的重建头为256→1024→1024→1，GELU/Dropout，末端Linear，无归一化/输出幅值约束；分类头为256→128→2。结构提供潜在放大路径，不代表已证明某一层引发事故，也不是仅凭缺final norm就认定实现错误。
8. MSE对预测导数为2*(prediction-target)/K；CE对logit导数为(p-onehot)/C，单点分量有界，但CE本身及对参数的梯度没有通用上界。自信错判时CE也可爆炸。两头共享骨干，重建异常可损害分类。
9. src/train.py:140和233-239使用AdamW固定lr=5e-4，loss权重1/1，无warmup/调度/梯度裁剪，FP32。AdamW不是普通SGD，大梯度不意味着参数步长线性同倍放大；仍没有保证输出稳定的机制。学习率/优化器状态只能列候选因素。
10. mask只屏蔽key/value，masked query仍参与计算并预测是预期补全行为；当前约15%遮挡不会使整条序列所有有效key被屏蔽。padding开关仅直接改变CE；ignore仍有尖峰，因此不是必要原因。

## 当前解释及验证边界

优先假设：特定模型/AdamW状态与batch输入、mask、dropout相遇，使骨干投影或重建头瞬时放大，MSE及其梯度放大扰动，随后更新进一步影响骨干；优化器可能恢复。损失突增不等于已经捕获梯度爆炸，旧运行缺逐步梯度与现场。

当前未复现不能否定历史事故；r3在epoch0已偏离旧轨迹，是同配置重跑。尚不能定位是骨干、projection还是重建头首先失稳，不能将CUDA非确定性、padding、小标准差或固定学习率写成已确认根因。

当前H全位置绝对值曾到845.88而选中MSE仍正常，说明全位置H/预测max不能直接替代masked位置证据。首次事件需结合mask、逐样本误差、预测/目标与前序batch；局部重放时再查各层激活、分支梯度与Adam更新量，区分前向放大和更新诱发。保持当前训练捕获现场，后续稳定性改动须独立单因素对照，不混入本次诊断。


## 官方重建链路复核（2026-11-15）

按用户要求直接读取patch_20261110/sources中的官方固定版本源码：historical training.py@ebf4f056b571f7ea90602ef666ae56f7d717ebfe，SHA256=311bfc8eece430c99a8fc4fac0bcc24b877ec9193f8d3292214611887a238b23；library@b031773e323bc0e6c9dcd8c9119b800ec4d70f8c，SHA256=f05b70d9daaf2f5d4e2867cb2d6daa0b235d5d3941bbc0862eff71614b349028。与既有source_manifest一致。本次尝试联网重取两个固定版本均超时；证据来自此前下载保存的源码，不声称检查了最新远端main。

官方training.py:240-247按batch观测统计标准化X及X0；:101-109重建头末端Linear，无预测标准化/裁剪；:148-156输出仅view后对选中位置直接F.mse_loss；:414-423默认FP32分支sum两项loss后backward/optimizer.step，无clip_grad、有限大loss跳步、回滚；:639-643 AdamW，:783 lr5e-4，:788 mixed_precision=False。可选AMP分支有GradScaler，不是有限大误差保护，默认不启用。库快照:91-98也是直接MSE。根training.py对应模型构造、重建loss、collate_fn和train_epoch四段AST与官方快照完全相同；v9在本次讨论的标准化/输出/损失/FP32更新保护语义上保持该行为，不代表整个训练流程相同。

用户担心大误差反传影响已学权重有代码依据：两项loss联合更新共享骨干，默认不会因有限尖峰阻止step。历史后续损失恶化与此相容，但不能仅凭loss证明永久损坏或具体更新幅度，需现场Adam状态和参数差分。官方一致不等于稳定性设计充分。本次无训练改动；详细逐行证据见official_reconstruction_check_20261115.json，一次性脚本已自删。
