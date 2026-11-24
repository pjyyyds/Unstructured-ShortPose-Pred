# JRDB 真实姿态与地面轨迹时刻核对

已归档的 Hewlett-Packard Intersection、Huang Lane、Tressider 三个室外训练片中，能证实满足 PINNS 原协议的窗数是 **0**。核对对象是 ST 作者的三份训练 NDJSON 和 HST 作者的三份三维人体关键点、三份轮式里程计。这九个成员逐个通过先前登记的 SHA-256。对其中 7,089／6,430／2,251 个有效地面行人帧逐个筛查，原帧号均为6的倍数。有当前地面中心的人在恰好45个原始帧之后（名义3.0s）具有下一条地面中心的计数，三片均为零。已核对的 ST 原作者归档目录列这三片各一份 shift_0 训练导出。原始样本时钟的毫秒级偏差无法从这些表恢复。[ST 作者发布](https://github.com/vita-epfl/social-transmotion/releases/tag/ckpt_data) · [HST 原作者数据说明](https://github.com/google-research/human-scene-transformer/tree/f60bc7fd63487f7135f70b7ed4a1f25a1f32c5bd/human_scene_transformer/data)

## 原生二维姿态与传感器时钟

JRDB-Pose 的来源是15 fps 全景影像，人工隔一帧标一次关节（7.5 fps），中间帧为线性补值。其17点定义包含头、眼、颈、髋等；可见性 0／1／2 区分不可见、虽遮挡可推断、可见。该标记与 RTMPose 检测置信度的含义不同。论文说明片内 track ID 与已有二维、三维人框一致。官方 JSON 示例含 image_id、track_id、category_id、keypoints；人体骨架筛选 category_id=2。公开示例没有示范单关节的插值标记；如获得原件，须从 images 的源文件名与作者可核验的标注相位确定人工帧，无依据时保留“来源未核实”。[官方 JRDB-Pose 格式与人工标注频率](https://jrdb.erc.monash.edu/dataset/pose) · [作者论文](https://openaccess.thecvf.com/content/CVPR2023/html/Vendrow_JRDB-Pose_A_Large-Scale_Dataset_for_Multi-Person_Pose_Estimation_and_Tracking_CVPR_2023_paper.html)

在名义15 Hz相机时间轴上，0.1秒等于1.5个帧间隔。即便拥有全部 RGB，也不能按整数帧每一步均取0.1秒；五个互不重复的真实图像可以覆盖0.4秒，但中间时间间隔将变化。7.5 Hz 手工姿态的连续五帧至少覆盖8/15秒，约0.533秒。原始 JRDB 的二维与三维框也含7.5 Hz手工与补到15 Hz的条目，布局文档在框的 attributes 中定义 interpolated。该文档把原始地面3D框定义在上部 Velodyne 坐标系，并提供 frames_img.json 和 frames_pc.json 的时戳字段；五个相机硬件同步，激光与相机软件近似同步。里程计 pcd 帧名的存在可证索引覆盖，不能替代两个时钟的实测差。[JRDB 原始数据说明](https://jrdb.erc.monash.edu/dataset/) · [作者的文件与字段文档](https://drive.google.com/file/d/1DyfU_A0x5OzdDGUWPPPWuW-epupIyvT7/view) · [原数据论文](https://arxiv.org/abs/1910.11792)

## 实际可得的轻量对照

将 t0 前后严格分开，取 t0−6、−5、−3、−2、0 这五个真正存在的 HST 源帧与同号里程计。相邻名义时间依次为1/15、2/15、1/15、2/15秒。要求该人 t0 同时有 ST 地面中心，留下 753／386／54 个起点。要求五个姿态时刻都有同人地面记录则剩0窗，因为 ST 只每6个源帧有一个中心。HST 这33点是由模型估计的三维骨架，原始人记录只有键和99个数值；这些点的变换与全局人体中心不同，未提供原始图像、像素置信度或激光测得的行人中心。[HST 坐标处理源码](https://github.com/google-research/human-scene-transformer/blob/f60bc7fd63487f7135f70b7ed4a1f25a1f32c5bd/human_scene_transformer/data/utils.py)

放宽至两个地面历史点、五份真实来源三维骨架及七个2.5 Hz地面预测目标（到名义2.8s），三片可以构造723／372／47个候选起点。按人贪心保留不重叠的完整历史到未来区间，留下 **123／68／6，共197窗**。另一个到3.2s的方案为115／63／5，共183窗。197窗的对照协议、完整与缺失分母分别在 PROTOCOL.md、input_support.json。已有138窗取样先划固定块再检查两个姿态；这里先检查五个来源姿态再按人贪心去重，两个分母不能直接比较。

## 授权与原始输入

官方总下载入口要求登录，注册表要求与机构一致的机构邮箱。本次未注册或代签协议。对固定版本官方开发工具的完整453条 Git 树路径作了检查，没有这三个场景的原始二维姿态、原始影像、三维中心或两个时间戳文件。若获得持有者许可并希望做时间近似的真实图像实验，关键资料为：同一训练片的原相机图像及帧名／相机纳秒时间戳，源点云帧时间戳、带插值位与匹配 ID 的原始3D人框，和相应标定／里程计。图像可用于按相同期号运行冻结的二维姿态网络。须先检查跨钟配准和正确坐标变换，区分实测、手工与插值，并固定准入容差。当前原始RGB和原始中心缺失；即使取得原件，采样频率差异仍须明确处理。三段与三个地点的录像均已用于开发。[官方下载页](https://jrdb.erc.monash.edu/#downloads) · [注册入口](https://jrdb.erc.monash.edu/register/) · [固定版本官方工具](https://github.com/JRDB-dataset/jrdb_toolkit/tree/4fbf7d6eba3255746000eb8c15f707af69561c5d)

## 本地覆盖与 Kaggle 比较

```sh
python check_intersection.py AUTHOR_TRAINING_MEMBERS intersection.json
python run_five_source_poses.py AUTHOR_TRAINING_MEMBERS --check-only
python test_five_source.py
```

kaggle_cell.py 是启用 Internet 的私有 Kaggle CPU 单元。它从两个研究团队的公开派生归档读取九个指定训练成员，或者接受指纹相同、已挂载且保持目录结构的九个文件。通过身份与合成测试后才会用另外两片训练并预测留出的第三片。输出 /kaggle/working/jrdb_five_source_poses.zip，保存实际输入、12组回归系数和折索引、五种路径（含匀速）、误差以及按窗和按人的聚合。只公开聚合。严格的 PINNS 五帧二维姿态、均匀10 Hz、三秒地面测试仍无观测结果。

仓库根目录运行 `python reviews/jrdb_five_pose/build_cell.py` 可重建完整独立 Kaggle 单元 `reviews/jrdb_five_pose/kaggle_cell.py`。脚本将本目录的协议和运行文件与上一项 JRDB 实验定义的四个源文件合并。该单元 SHA-256 为 `76d0a2e40955f119d6bfda8ed7318b1c128c8417f169cef68da10546febeeaa3`。作为 Kaggle 私有 CPU Notebook 的代码单元执行即可。
