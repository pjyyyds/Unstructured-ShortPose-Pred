# JRDB 训练片段的坐标核验

Social-Transmotion 发布的 10 个 JRDB 训练片段都有同名的 Human Scene Transformer 轮式里程计记录。逐片比较首帧与后续各帧，记录到的最大水平位移为 0.000542 m，转角最大的一段为 0.003825°。这些记录中的机器人几乎静止，用它们判断派生轨迹是否已经补偿自运动，辨识力有限。[逐片结果](train_split_odometry.json) · [数据发布包](https://github.com/vita-epfl/social-transmotion/releases/tag/ckpt_data)

在训练片段 `stlc-111-2019-04-19_0`，111 人的 8,304 个位置分布在 290 个时刻，轨迹原始帧号每六帧递增。290 个时刻都对上了里程计帧号。按 HST 的变换，将轨迹坐标视作机器人系下的米制坐标并转换到首帧坐标系，水平位置改变量中位数为 0.000210 m，最大为 0.001140 m。核验尺度和 Social-Transmotion 导出时采用的变换仍需要开发片段的原始 3D 框中心或作者的导出说明。[帧对齐及输入摘要](training_coordinate_check.json) · [HST 坐标转换](https://github.com/google-research/human-scene-transformer/blob/main/human_scene_transformer/data/utils.py#L153-L179) · [JRDB 数据任务](https://jrdb.erc.monash.edu/benchmark/)

## 室外训练片段的全景投影和姿态匹配

JRDB 作者将 Huang Lane（`huang-lane-2019-02-12_0`）标为室外静止机位。Social-Transmotion（ST）训练文件包含 70 名行人的 6,430 个唯一（原帧号、行人 ID）记录。JRDB 原生 RGB 为 15 Hz，ST 每 6 帧取一次，对应 2.5 Hz。此片覆盖原帧 0 至 720，共 121 个采样时刻。[JRDB 论文图 2(c)](https://arxiv.org/html/1910.11792#S3.F2) · [JRDB 训练划分](https://github.com/JRDB-dataset/jrdb_toolkit/blob/4fbf7d6eba3255746000eb8c15f707af69561c5d/traj_eval/convert_dataset_to_KITTI_tracking.py#L31-L60) · [HST 训练划分](https://github.com/google-research/human-scene-transformer/blob/f60bc7fd63487f7135f70b7ed4a1f25a1f32c5bd/human_scene_transformer/config/jrdb/dataset_params.gin)

按官方坐标轴，把 ST 的 `(x,y)` 投影到 3,760 像素宽的圆柱全景，比较投影方位与同帧本人二维框的水平中心，拼接边界处取较短的圆周距离。中位偏差为 8.53 px（0.817°），95% 分位为 27.35 px（2.619°）。4,839/6,430（75.26%）落在本人框的水平范围内。同帧错配到下一行人 ID 后，中位偏差为 208.05 px。结果支持该片段的轴向与配对。整套平面坐标乘以任一正数都会得到相同的投影方位，独立验证距离尺度仍需原始 3D 中心或已知尺度的参照。[JRDB 传感器说明，第 3 页](https://download.cs.stanford.edu/downloads/jrdb/Sensor_setup_JRDB.pdf#page=3) · [固定版本的官方投影](https://github.com/JRDB-dataset/jrdb_toolkit/blob/4fbf7d6eba3255746000eb8c15f707af69561c5d/visualisation/visualise.py#L556-L591)

HST 作者发布的该片段有 3,320 个人帧姿态记录。按原帧和行人 ID 与全部 ST 样本匹配，得到 554/6,430（8.62%），覆盖 40 人。554 份匹配记录均有 33 个有限的三维关节坐标，关节记录只含人员 ID 和 99 个坐标值。该格式不提供二维像素关节或 RTMPose 置信度。JRDB-Pose 另有官方人工 17 点二维标注，可见性取 0、1、2。按其通常采用的 6,500 px² 面积线，ST 框中有 1,582 个达到该面积。实际人工标注覆盖率需要原标注确认。原论文公开了单张带框和点云的合成图，取得同名原始连续帧和人工标注需通过 JRDB 官方下载。[HST 关键点读取](https://github.com/google-research/human-scene-transformer/blob/f60bc7fd63487f7135f70b7ed4a1f25a1f32c5bd/human_scene_transformer/data/utils.py#L73-L86) · [JRDB-Pose 标注说明](https://jrdb.erc.monash.edu/dataset/pose) · [Huang Lane 论文单图](https://arxiv.org/html/1910.11792v4/figs/jrdb_viz_14600.png) · [JRDB 原始下载](https://jrdb.erc.monash.edu/#downloads)

室外运动对照片段 Memorial Court（`memorial-court-2019-03-16_0`）在 JRDB 与 HST 中均属训练。其实测 1,091 帧轮式里程计相对首帧的最大水平位移为 62.692 m，最大三维转角为 59.995°。间隔 42 原帧（名义 2.8 s）的机器人水平位移中位数为 2.614 m。ST 训练包缺少该片段的行人轨迹，尚无法用它核实 ST 导出时对运动机位实施的变换。[JRDB 论文图 2(d)](https://arxiv.org/html/1910.11792#S3.F2) · [HST 里程计发布说明](https://github.com/google-research/human-scene-transformer/tree/f60bc7fd63487f7135f70b7ed4a1f25a1f32c5bd/human_scene_transformer/data#get-the-robot-odometry) · [汇总结果、来源版本和输入 SHA-256](outdoor_projection.json)

## 复核

在本目录运行以下命令可重算汇总。脚本从作者公开 ZIP 获取指定训练成员并校验 CRC。室外核验还校验三个成员的固定 SHA-256。环境需有 Python 3.9 以上及 NumPy。

```sh
python3 screen_training_odometry.py
python3 check_training_coordinates.py
python3 check_outdoor_projection.py
```

若已从 JRDB 取得授权的训练集原始 3D 标注，可在第二条命令后增加 `--labels-3d /path/to/stlc-111-2019-04-19_0.json`，对齐原始帧号及行人 ID，比对派生坐标与原始框中心。[HST 对原始数据的说明](https://github.com/google-research/human-scene-transformer/tree/main/human_scene_transformer/data)
