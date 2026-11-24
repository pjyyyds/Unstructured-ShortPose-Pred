# DUT 校园共享空间复现

本目录分析作者标为共享空间的全部 11 个 `roundabout_01` 至 `roundabout_11` 视频。原始发布还含 17 段人行横道视频。行人坐标由人工初始化、CSR-DCF 逐帧追踪得到。本研究使用未滤波坐标，并按作者给出的逐片像素/米比例换成米。作者把俯视的人头跟踪平面近似为地面。源文件没有逐人框、人体关节真值或独立曝光时间戳。[作者仓库](https://github.com/dongfang-steven-yang/vci-dataset-dut/tree/80b8c746833664cd1e5244fccad79c7f2a7cbe31) [作者论文 §IV](https://dongfang-steven-yang.github.io/files/2019-IV19.pdf)

## 从空数据目录开始

把本复现目录的脚本和冻结协议复制到自己的私有工作目录，随后在该目录执行以下命令。Python 需要 NumPy、OpenCV、PyAV 和 rarfile。视频解包使用系统的 `bsdtar`，macOS 自带，Ubuntu 可安装 `libarchive-tools`。可复用已有环境，也可在私有工作目录创建环境。

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy opencv-python-headless av rarfile
python stage_sources.py
python stage_shared_videos.py
python census.py
python stage_registration.py
python register_roi.py
python flow_registration.py
python spatial_coverage.py
```

1. `stage_sources.py` 首次运行会下载固定 Git 树，并核对全部 349 个条目的路径、类型、对象哈希和大小。作者提交为 `80b8c746833664cd1e5244fccad79c7f2a7cbe31`，树对象为 `b4bbe87e9b714cf9d2c571ab45ea4da6862049d1`，条目摘要为 `7b8792f59e66df28a2e99933816fe2ae0ca73fad805cad459313db07432add38`。检查通过后生成 `author_tree.json`，继续读取两类未滤波轨迹、比例、ROI 文件及官方 RAR 的成员目录。GitHub 和作者 Google Drive 都需要联网。[固定树目录](https://api.github.com/repos/dongfang-steven-yang/vci-dataset-dut/git/trees/b4bbe87e9b714cf9d2c571ab45ea4da6862049d1?recursive=1)
2. `stage_shared_videos.py` 只提取登记的 11 个视频成员，合计 59,150,836 压缩字节。原始 RAR 大小 104,243,676 字节。程序校验成员的大小与 CRC。视频位于 `private_source/vci-raw-dut/`，CSV 和比例位于 `private_source/data/`，裁剪参数位于 `private_source/legacy/clips/rois/`。[作者原始下载](https://drive.google.com/file/d/1J0PW4NoL2mi_eaN_8_SgkWkc0rk7MJZC/view)
3. `census.py` 读取真实 MP4 展示时间、每片比例及源坐标，产生公开覆盖汇总 `coverage.json` 和私有逐窗文件 `private_cohort.json`。3,066 个键的摘要为 `02c44cf55e5ca8690dc1fe9ac9c076af44dc89f11629b8ccc9e725e49e4580ec`。视频流均为 23.976 fps。最近 10 Hz 网格采样得到的五张历史画面实际跨 0.417084 秒，最后一个未来样本约为 3.003003 秒。历史间隔或末点满足 1 毫秒容差的样本数均为零。
4. `stage_registration.py` 自动把 11 张作者背景、11 份包含原始像素点的 legacy 行人 CSV、作者发布的 04 号完整稳定化录像、带标记 ROI 录像及比较所需文件写入 `private_reference/`。`register_roi.py` 验证背景/ROI 配准和 04 号原视频裁剪、叠画及帧号，输出 `registration.json`。`flow_registration.py` 用当前源 CSV 与 legacy 原始列逐键比对，并检验每片 20 对相邻图像的局部光流，输出 `flow_registration.json`。两个脚本各自读取对应预登记文本。
5. `spatial_coverage.py` 只使用已登记窗口的键及决策帧或更早的其他行人位置，输出 `spatial_coverage.json`。5 米方格的原点取各自 ROI 左上角；不同视频的格子尚未配准到同一个地理坐标系。

## 图像配准和公开边界

发布的视频已经裁成 ROI，第一张画面对应轨迹帧号 1。图像定位直接使用未滤波的 ROI 像素点。04 号的清洁视频与 Git 跟踪文件逐字节一致。将作者全幅稳定化视频按其记录的左上角裁切，42 个可与前后帧同时比较的固定样本中，同编号图像的差异均最小。所有 11 片的原点光流均比两个预设错位对照更接近源轨迹位移。02 号的公开背景为 628×568，而影片为 574×598，该背景不能作为已配准地图库使用。论文摘要所述 1920×1080、仓库概述所述 4K 均不能代替本次实际解码的逐片 ROI 尺寸。源标尺属于俯视平面近似，姿态检测还受像素分辨率和多人重叠影响。[作者裁剪与叠画程序](https://github.com/dongfang-steven-yang/vci-dataset-dut/blob/80b8c746833664cd1e5244fccad79c7f2a7cbe31/legacy/scripts/vis_and_demo.py) [作者公制转换程序](https://github.com/dongfang-steven-yang/vci-dataset-dut/blob/80b8c746833664cd1e5244fccad79c7f2a7cbe31/scripts/filter_trajectories.py)

原始视频、截图、压缩包、逐人轨迹、逐窗预测及缓存留在私有工作目录。作者提供公开下载并请研究使用者引用原论文，固定仓库没有附数据再分发许可证。代码和汇总可单独交付。Kaggle 的 M0 和融合单元需要本项目已核验的私有 PINNS 权重；检测单元从原作者来源直接获取图像和预训练模型。预测只在 Kaggle 执行。
