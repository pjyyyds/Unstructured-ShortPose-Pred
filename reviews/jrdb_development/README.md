# JRDB 训练片段的坐标核验

Social-Transmotion 发布的 10 个 JRDB 训练片段都有同名的 Human Scene Transformer 轮式里程计记录。逐片比较首帧与后续各帧，记录到的最大水平位移为 0.000542 m，转角最大的一段为 0.003825°。这些记录中的机器人几乎静止，用它们判断派生轨迹是否已经补偿自运动，辨识力有限。[逐片结果](train_split_odometry.json) · [数据发布包](https://github.com/vita-epfl/social-transmotion/releases/tag/ckpt_data)

在训练片段 `stlc-111-2019-04-19_0`，111 人的 8,304 个位置分布在 290 个时刻，轨迹原始帧号每六帧递增。290 个时刻都对上了里程计帧号。按 HST 的变换，将轨迹坐标视作机器人系下的米制坐标并转换到首帧坐标系，水平位置改变量中位数为 0.000210 m，最大为 0.001140 m。核验尺度和 Social-Transmotion 导出时采用的变换仍需要开发片段的原始 3D 框中心或作者的导出说明。[帧对齐及输入摘要](training_coordinate_check.json) · [HST 坐标转换](https://github.com/google-research/human-scene-transformer/blob/main/human_scene_transformer/data/utils.py#L153-L179) · [JRDB 数据任务](https://jrdb.erc.monash.edu/benchmark/)

在本目录运行以下命令可重算汇总。脚本从作者公开 ZIP 获取匹配的训练成员，校验 CRC。环境需有 Python 3.9 以上及 NumPy。

```sh
python3 screen_training_odometry.py
python3 check_training_coordinates.py
```

若已从 JRDB 取得授权的训练集原始 3D 标注，可在第二条命令后增加 `--labels-3d /path/to/stlc-111-2019-04-19_0.json`，对齐原始帧号及行人 ID，比对派生坐标与原始框中心。[HST 对原始数据的说明](https://github.com/google-research/human-scene-transformer/tree/main/human_scene_transformer/data)
