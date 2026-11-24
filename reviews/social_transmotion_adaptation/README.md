# Social-Transmotion 作者代码的可运行范围

作者 JRDB 模型在官方 9 帧输入形状下已用合成序列运行。PINNS 的 5 帧输入、30 步输出也已完成软件验证，需修改模型内部时间长度。后者可称“基于 Social-Transmotion 作者 JRDB 代码的 PINNS 单人适配”。两次均为随机初始化的 CPU 前向。尚未验证真实预训练权重的推理、训练完整性或任何预测分数。[作者仓库固定版本 7695ba3](https://github.com/vita-epfl/social-transmotion/tree/7695ba335f7cba7f41660d0edc66a57cd52da08a) · [ICLR 论文 §4.1–4.2](https://proceedings.iclr.cc/paper_files/paper/2024/file/a627810151be4d13f907ac898ff7e948-Paper-Conference.pdf)

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

## 复核步骤

运行 inspect_release.py 可复查 ZIP 目录。官方仓库的原始工作副本保存在 source/，未修改。令 PYTHONPATH 包含本项目及已有依赖，运行以下命令可复查合成前向和真实窗口输入。程序不提供训练或真实预测入口。

    python check_protocol.py --metadata ../pinns_metadata --pose ../loso_artifacts/pose.npz
