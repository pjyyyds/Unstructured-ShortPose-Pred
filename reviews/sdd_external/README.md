# Stanford Drone Dataset 标注频率核验

开题报告把 SDD 列为非结构化混行场景的备选数据集。我们检查其 8 个场景、60 段录像的行人标注能否直接构成 10 Hz 的实测短历史。窄化条件只要求目标在画面内（`lost=0`）且该帧的框没有自动插值（`generated=0`），允许 `occluded=1`。按原视频 30 fps，每隔三帧取样，三个可能的起始相位分别检查。此次核验没有读取预测分数，也不依赖像素到米的比例。[斯坦福项目与数据许可](https://cvgl.stanford.edu/projects/uav_data/) · [OpenTraj 标注字段定义](https://github.com/crowdbotp/OpenTraj/blob/97ec1d5e1b579f26febeaa9083705952deac338a/datasets/SDD/README.md)

画面内有 4,094,713 条行人标注，其中 4,019,895 条带自动插值标记，74,818 条没有该标记。对同一个人，任一相位均找不到连续 5 个未插值的 10 Hz 位置。最长连续序列为 3 个位置。因此直接使用这些发布框，不能构成本课题要求的五帧地面运动输入及其连续三秒真值。原录像仍可用于重新检测、跟踪和独立标注。此次没有以 SDD 计入姿态或轨迹迁移成绩。[各片逐相位计数及来源摘要](annotation_support.json)

项目组官方完整压缩包端点在此次检查时连接超时。我们改取两个固定版本的公开分发库，验证两边全部 60 份标注 Git 对象摘要及长度相同，并核实实际解包文件的内容。原始标注和视频按来源的 CC BY-NC-SA 3.0 许可分发。原数据署名 Alexandre Robicquet、Amir Sadeghian、Alexandre Alahi、Silvio Savarese，ECCV 2016。[数据项目](https://cvgl.stanford.edu/projects/uav_data/) · [镜像一](https://github.com/crowdbotp/OpenTraj/tree/97ec1d5e1b579f26febeaa9083705952deac338a/datasets/SDD) · [镜像二](https://github.com/flclain/StanfordDroneDataset/tree/9179f3c0dd9ac0bd67a6223703d04a7577a9fb18)

从仓库根目录运行即可复算。运行时需要能够读取两个固定版本的 GitHub 文件清单。

```sh
curl -fL https://codeload.github.com/flclain/StanfordDroneDataset/tar.gz/9179f3c0dd9ac0bd67a6223703d04a7577a9fb18 -o sdd_annotations.tar.gz
python3 reviews/sdd_external/audit_annotation_support.py sdd_annotations.tar.gz sdd_annotation_support.json
```
