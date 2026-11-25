# Social-Transmotion 作者代码的可运行范围

本研究将 Social-Transmotion 作者的 JRDB 实现在 PINNS 上适配为单目标、5 帧观测、30 步确定性预测。Kaggle 版本 373628213 在 CPU 上完成四地留出训练，耗时 4,612.0 秒。输入条件为仅地面轨迹、地面轨迹加二维框，均从种子 42 开始训练。作者原始 9 帧输入另用合成样本检查接口。[作者仓库固定版本 7695ba3](https://github.com/vita-epfl/social-transmotion/tree/7695ba335f7cba7f41660d0edc66a57cd52da08a) · [ICLR 论文 §4.1–4.2](https://proceedings.iclr.cc/paper_files/paper/2024/file/a627810151be4d13f907ac898ff7e948-Paper-Conference.pdf)

## 原版协议、发布物与许可

- JTA 和 JRDB 均按 2.5 Hz 观察 9 帧、预测 12 步。首末观察相距 3.2 s，预测到 4.8 s。论文把该模型列在确定性预测的比较中。作者模型对场景中的主行人输出一条二维轨迹，评分函数对该轨迹的 12 个欧氏误差取均值，最后一步为 FDE。模型不产生六候选、模态概率或可供校准的置信度。[论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/a627810151be4d13f907ac898ff7e948-Paper-Conference.pdf) · [JRDB 输出](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/model_jrdb.py#L241-L253) · [作者评价器](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/evaluate_jrdb.py#L23-L71)
- JRDB 配置使用轨迹和二维框两种 token，推理可选择轨迹，或轨迹加框。模型文件内虽声明姿态编码层，实际 JRDB 前向只接入这两种 token。JTA 原版可接受轨迹、二维框、二维/三维骨架、三维骨架加三维框、或全输入。JTA 配置使用 22 关节；直接传入 PINNS 的 17 个 COCO 关节会改变语义。官方 release 没有提供针对 COCO-17 的同一姿态接口。[作者模态说明](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/README.md#L35-L58) · [JTA 模型](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/model_jta.py)
- 官方 release 的 ZIP 为 305,870,808 字节。本轮仅读取 65,318 字节目录及尾记录。目录列出 JTA 的 train/val/test 文件 206/10/12，JRDB 为 10/1/2。这些是预处理 NDJSON **文件数**，不能等同于原始数据集全部录像或评测场景数。另有 JRDB 与 JTA 的原生 checkpoint，各为 37,280,495 与 37,712,843 字节，成员路径与 CRC 见同目录 release_inventory.json。原版 loader 读取 checkpoint 内 config 和 model state；本轮只验证了两份权重在受检目录中的身份。模型库要求 PyTorch 1.13.1+cu116；模型级 CPU 冒烟使用 2.10.0。[发布页](https://github.com/vita-epfl/social-transmotion/releases/tag/ckpt_data) · [作者依赖](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/requirements.txt) · [原版权重加载](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/evaluate_jrdb.py#L88-L110)
- 仓库随附 GNU Affero General Public License version 3（AGPLv3，2007-11-19）。README 对预处理数据许可指向原数据论文。release 中未发现独立的权重授权说明。仓库源码许可、数据使用条款、权重授权应分别记录。[许可证原文](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/LICENSE) · [数据许可指引](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/README.md#L24-L30)

## 时长适配与评分

- JRDB 的 YAML 设置了 9/12。模型内部独立地把输入框长度写为 9，把总长度写为 21，用于时间编码、embedding、切片和 reshape。评价器也固定 reshape 为 12 步。驱动核验作者模型文件的 SHA 后在临时模块里替换相应数值为 5/35，并把局部 transformer 序列长由 30 设为 40（35 个轨迹 token 和 5 个观测框 token）。其余模型层沿用原版。原版时间 embedding 和修改后的形状不同，所以原生 checkpoint 无法以严格匹配直接加载。默认复制末次观测作为未来查询初始化，不用任何未来位置。[时间编码](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/model_jrdb.py#L46-L105) · [模型与查询](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/model_jrdb.py#L138-L252)
- 实测原版在两个合成人物（批次大小 2）上返回 2×12×1×2 未来坐标，适配版在单个合成主行人上返回 2×30×1×2。以经校验的官方 PINNS 元数据/冻结框重建了全部 1,482 个实际窗口，7,410 个观测框的宽高均为正。适配器只接受五个地面坐标和对应的历史框。它可将框 token 置零以使用纯轨迹，或把项目缓存的中心点宽高转换为左上角宽高，按作者 JRDB 预处理减末帧并乘 0.25。这里只核验了输入构造；原库未提供能证明两个数据源像素尺度或框检测分布等价的标定。[作者归一化](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/dataset_jrdb.py#L24-L59) · [原轨迹/框顺序](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/utils/trajnetplusplustools/reader.py#L63-L95)
- 若开展同队列比较，需在另外三个地点重新训练上述适配版，预先锁定优化器、种子与最终/验证选取规则。PINNS 的 5×30 为 10 Hz、实际观察跨度 0.4 s、未来终点 3.0 s。报告作者适配版的**单路径 ADE/FDE**，项目预测器另列**概率最高候选的 ADE/FDE**，二者都在预测时交一条路径。项目六候选中按未来挑选的 minADE/minFDE 只描述覆盖范围。单路径没有模态分类的 NLL、Brier 或选对率。单人设置排除了人际交互，代表作者网络在相同目标可见输入下的一种适配，不能代表完整原版社交方法在其原生任务的表现。
- 作者 JRDB 训练配置为 seed 0、100 轮、batch 16、Adam、初始学习率 1e−4，原脚本会记录验证最佳和末轮。论文写 MSE，发布代码中名为 MSE_LOSS 的函数对二维位置差取欧氏范数，再对时间、批次平均并乘 100；该函数未使用传入的 mask。项目的全未来完整队列可直接适用其目标，方法部分应以实际执行的损失为准。JRDB 的前向把两种随机特征屏蔽比例均设为 0。[作者训练配置](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/configs/jrdb_2dbox.yaml) · [损失实际实现](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/utils/metrics.py#L4-L14) · [原训练调用](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/train_jrdb.py#L51-L61)

## JRDB 划分与因果输入

作者派生测试成员为 packard-poster-session-2019-03-20_1 和 tressider-2019-03-16_0。JRDB 官方工具将两者列入原版 train；同日同地点的其他录像也出现在作者训练成员里。论文作者划分、官方挑战和独立新地点的含义不同。本轮只读取成员名称。JRDB 的移动机器人参照系仍需结合里程计核验，不能假定派生二维坐标已在静态场地世界系。[JRDB 官方划分](https://github.com/JRDB-dataset/jrdb_toolkit/blob/main/traj_eval/convert_dataset_to_KITTI_tracking.py#L26-L85) · [作者发布](https://github.com/vita-epfl/social-transmotion/releases/tag/ckpt_data)

作者 reader 要求邻人前 9 帧齐全，随后通过整段的最小相对距离筛选 6 m 内邻人，包含未来 12 帧；一个只在未来靠近的人可以进入当前观测的社交输入。直接复用该 loader 无法保证严格在线的邻人选择。此次单目标输入完全不选择邻人。若以后提供社交版本，须从五帧内的当前/已见人员构造候选，不以邻人未来完整性或未来距离筛人，单列这种协议变化。[作者场景和邻人筛选](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/utils/data.py#L7-L21) · [距离与完整性函数](https://github.com/vita-epfl/social-transmotion/blob/7695ba335f7cba7f41660d0edc66a57cd52da08a/utils/data.py#L66-L90)

## PINNS 四地点结果

四折合计 1,482 个完整窗口。美国、意大利、日本和泰国分别贡献 816/14、84/9、432/11、150/13 个窗口/片段。各折只用其余三地训练 28 轮，输出一条地面轨迹。先在每地计算窗等权 ADE 和 3 秒末步 FDE，再对四地等权。轨迹加框为 **0.6522 / 1.3244 m**，仅轨迹为 **0.5534 / 1.1239 m**，末两点匀速为 **0.4839 / 1.1046 m**。[计分前协议](PROTOCOL.md) · [Kaggle 原始聚合](scores.json)

预设主比较是轨迹加框减匀速。ADE 增加 0.1683 m（95% 配对区间 0.1238 至 0.2260），FDE 增加 0.2199 m（0.1219 至 0.3615）。仅轨迹减匀速的 ADE、FDE 分别为 +0.0695 m（0.0443 至 0.0943）、+0.0194 m（−0.0399 至 0.1103）。轨迹加框减仅轨迹的两项差分别为 +0.0988 m（0.0611 至 0.1463）、+0.2005 m（0.1294 至 0.2848）。美国片段中，轨迹加框的 ADE 为 0.9018 m，低于匀速的 0.9477 m；另外三地均高于匀速。区间来自固定四地点内的 20,000 次整片配对重采样，只涵盖这八次拟合和这些已用于开发的地点。[逐地与配对结果](scores.json)

## 复核步骤

从仓库根目录执行。使用 PINNS 的原始标签、对应标定与归档成员清单。第一条命令重建逐窗历史及评分目标。第二条读取私有 Kaggle 原始返包，并将复算聚合写入指定位置。

```bash
python -m scripts.audit_pinns_matched_histories /path/to/PINNS \
  --output /tmp/matched_summary.json --manifest /tmp/matched_histories.npz
python -m reviews.social_transmotion_adaptation.verify_return \
  /path/to/social_transmotion_pinns.zip /path/to/PINNS \
  /tmp/matched_histories.npz /tmp/social_transmotion_verification.json
```

复核脚本对照协议、8 个检查点的训练记录、留出键和两份独立重建的观测及未来坐标，从归档的确定性预测重算了三种方法共 4,446 行 ADE/FDE。另按原种子重建每个逐地和四地点等权区间。逐窗误差最大绝对差为 1.78×10⁻¹⁵ m，各级汇总及区间最大绝对差为 1.70×10⁻¹⁵ m。这次复核使用归档预测，未执行模型前向；原标签、逐人预测和权重保留在私有环境。[复核脚本](verify_return.py) · [独立聚合与来源摘要](independent_verification.json)

作者发布包的目录与许可记录见 [release_inventory.json](release_inventory.json)。若要另做作者模型的接口检查，可将固定作者版本的 `model_jrdb.py` 路径传给 `check_protocol.py --source`，并用 `--output` 指向临时结果。
