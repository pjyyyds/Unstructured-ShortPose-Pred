# DUT 校园共享空间核查

Yang 等将校园分为无信号斑马线和环岛旁人车自由移动的共享空间。按作者文件名选全 11 个 roundabout 片段，另 17 个 intersection 片段未纳入。这里能支持作者定义的共享空间场景；作者没有给路缘或车道线逐像素标签，不能逐窗断言没有路面标线。[作者数据说明](https://github.com/dongfang-steven-yang/vci-dataset-dut/tree/80b8c746833664cd1e5244fccad79c7f2a7cbe31) [研究论文](https://dongfang-steven-yang.github.io/files/2019-IV19.pdf)

作者通过 Google Drive 提供可直接下载的稳定化视频，仓库提供轨迹和逐片比例。核对的是该链接公开的官方 RAR 中的 11 个共享空间视频，选择性解出 59,150,836 个压缩字节并验证单成员 CRC 和 SHA-256。只在私有目录处理，公开材料不附视频、图像、逐人轨迹或裁剪。作者邀请研究使用并引用论文，固定仓库没有 LICENSE 文件，不能据此承诺源数据可再分发。[作者下载入口](https://drive.google.com/file/d/1J0PW4NoL2mi_eaN_8_SgkWkc0rk7MJZC/view)

作者从悬停无人机俯视录像中人工初始化跟踪点，再用 CSR-DCF 追踪。源原始 CSV 里的 x、y 为裁切后 ROI 的像素坐标；以作者逐片提供的像素/米比例换成米。原文件与作者转换代码一致。尺度换算近似将人头、车顶与地面视为一平面。公开的 Kalman 平滑版本未用于建立历史与真值。轨迹文件提供人或车的 ID、帧号和点坐标，未提供逐人检测框或人工骨架。源码逐帧读稳定化录像，第一张图编号为 1；发布的 ROI 影片已经裁切，读取轨迹像素时应使用该 ROI 坐标。[作者转换脚本](https://github.com/dongfang-steven-yang/vci-dataset-dut/blob/80b8c746833664cd1e5244fccad79c7f2a7cbe31/scripts/filter_trajectories.py) [作者绘制脚本](https://github.com/dongfang-steven-yang/vci-dataset-dut/blob/80b8c746833664cd1e5244fccad79c7f2a7cbe31/legacy/scripts/vis_and_demo.py)

实读 11 个 MP4 得到 5,069 张原始 ROI 画面，全部按 1/23.976 秒展示，流时基为 1/23976，PTS 从零起每帧加 1000。作者未发布独立的相机曝光时标。按预登记取最近画面而不插值，得到 3,066 窗、879 个片内行人，其中第 3 片占 1,876 窗。头五帧实际跨 0.417084 秒，未来末点 3.003003 秒。历史各段或三秒末点均无满足 1 毫秒容差的窗口。图像模型需要另外验证检测到的单人框及真实图像姿态，不能仅凭跟踪点确认人体姿态可读。
