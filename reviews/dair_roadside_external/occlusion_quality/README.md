# 遮挡等级与五帧 RTMPose 分数

54 个北京路侧窗口提供 270 个不同的行人帧、210 张不同的相机图像。原等级 0、1、2 分别有 172、96、2 帧，17 关节分数均值为 0.49955、0.35401、0.39193。以原特征构建的 0.20 门槛，三组分别有 2846/2924、1291/1632、34/34 个有效关节。所有帧均保留，整人全零输出为零帧。这些分数衡量检测器的信心，缺少人工关节坐标，无法换算为定位误差。

有 13 窗同时含无遮挡帧和遮挡帧。先在窗内比较同人两组帧，再对窗等权，遮挡减无遮挡的平均差为 −0.04930，中位差为 −0.03555；10 窗下降，3 窗上升，涉及 3 处路口、6 段录像、10 名片内行人。其中一处路口的 7 窗均值为 +0.00137。重遮挡的两帧来自同一人同一窗。[冻结口径](PROTOCOL.md) · [原归档汇总](occlusion_quality.json) · [独立复核](verification.json)

将原 Kaggle 输出 ZIP 与其来源的 DAIR-V2X-Seq 路侧元数据目录作为入口参数。该目录包含 data_info.json 与 label/camera/。主程序读取两个原始关节像素坐标及一个置信度的存档格式，参考投影未进入统计。第二个程序只需要 Python 标准库。两份程序不运行模型。

```sh
d=reviews/dair_roadside_external/occlusion_quality
python "$d/check_occlusion.py" --test
python "$d/check_occlusion.py" --archive RETURN.zip --metadata METADATA --protocol "$d/PROTOCOL.md" --out /tmp/dair_quality.json
python -I -S "$d/verify_second.py" --archive RETURN.zip --metadata METADATA --protocol "$d/PROTOCOL.md" --reference /tmp/dair_quality.json --out /tmp/dair_quality_verified.json
```
