# PINNS 图像地图与轨迹的地面坐标

每一条源 Car 记录同时给出鸟瞰图车辆中心 (u,v) 和地面中心 (world_x,world_y)。在四个地点、47 段录像的 47,555 条记录中，地面两轴随图像 u、v 同向增加。22 套标定中，项目读取轨迹的额外纵向翻转标志均为假。按整个宽高换算复现所有存储坐标，最大差小于 0.000000723 m。该数值描述同一发布行内两个字段的一致性。[完整来源、逐片与逐地点结果](vehicle_axis/car_bev_scale.json)

## 图像轮廓的坐标修复

初版程序从鸟瞰图取轮廓时先反转 v 方向，再作米制换算。对同一批车辆中心使用这个公式，中位和 P95 差分别为 11.187 和 23.863 m。仅用宽高减一、保留正确方向时则为 0.03464 和 0.04751 m。这里的比较利用有配对真值字段的车辆中心，不能把差值直接解释为某条路缘的位置误差。

对已登记的因果地图，26 段有合格的训练期参考，21 段无合格参考。重建同一批 182 个来源帧、260 条鸟瞰轮廓，原 7,283 个窗口的旧图像地图张量逐元素相同。只改图像坐标到轨迹坐标的转换，再按同一半径和上限选线，有图窗口由 4,898 个增至 4,925 个；2,913 窗的入选轮廓集合改变。静态图像轮廓、从行人轨迹构建的占用凸包和已在地面坐标中的邻居分别有独立来源。含图像轮廓的旧地图消融以及与它共同使用可用标志的模型，需在新坐标的同一窗上重训复测。四地留出的 M0/M1 不读图像地图。[独立重建](vehicle_axis/causal_map_support.json) · [重评协议](../pinns_map_axis_comparison/PROTOCOL.md)

## 三段美国录像的行人图像点

另一个问题来自三个以源帧 0 开始的美国片段。对 24,304 条 10 Hz 人物记录，原图点通过发布四角投影与同一行地面坐标的中位差为 0.863 m。在这三片与一个正常对照片，原图和作者生成的鸟瞰图有直接的图像对应。用第 0 帧的 SIFT 对应拟合后，比较第 60、150 帧；三个异常片中，以发布四角预变换后的双向匹配中位差均不足 0.81 个鸟瞰像素。进一步用图像配准修正后，三个片的同一行行人标注之间仍有 0.723、0.698 和 1.165 m 的中位差。[预设图像配准方法](PROTOCOL.md) · [逐片配准与行人核对](video_alignment.json)

用其他两个片的像素/地面标注估计单应再应用到留出整片，中位残差降至 0.09375 m，P95 为 0.71364 m。把同人地面帧前后移动一个原始帧，对原四角的中位差影响小于 0.003 m。行人数据字段含明显的空间相关差异；作者未发布这三片单独的导出记录。上面的单应使用同地点其他片的地面标注，只用于核对，不用于预测或部署。在原论文中，作者区分了原图的行人框和鸟瞰图的车辆框。[原字段诊断](label_projection/scores.json) · [作者的标注说明](https://arxiv.org/html/2605.25947v1#S3.SS3)

## 复现

原始 PINNS 授权目录应包含 labels/ 和每个采集条件的 calib.json 及原视角视频。BEV_ROOT 包含相同目录布局下的鸟瞰视频。旧侧表含已冻结的 maps.npz 与 manifest.json。合成测试或历史训练需要对应研究登记的 Git 提交；跨版本目录中新增文件会改变全源码 SHA，不能用于复原早期签名。图像复核使用 OpenCV 和 NumPy。

```sh
study=reviews/pinns_projection_sources
python "$study/check_camera_views.py" METADATA BEV_ROOT /tmp/video_alignment.json
python "$study/vehicle_axis/check_bev_scale.py" METADATA /tmp/car_bev_scale.json --repo . --bev-root BEV_ROOT
python "$study/label_projection/diagnose_projection.py" METADATA /tmp/label_projection.json --reference results/pinns_projection_audit.json
python "$study/plot_axis.py" "$study/vehicle_axis/car_bev_scale.json" /tmp/fig9.png
```
