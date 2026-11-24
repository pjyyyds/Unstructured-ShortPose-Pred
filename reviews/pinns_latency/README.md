# 同一决策时刻的多行人前向耗时

Kaggle 解压交付包后，将 `pinns_sync_latency_private.bin` 单独上传为私有 Dataset，给新私有 Notebook 添加该输入，导入 `PINNS_sync_latency.ipynb`。在设置中选择 NVIDIA GPU，再运行全部单元。运行仅需本地挂载的输入以及自带 PyTorch CUDA 和 NumPy 的环境，Internet 可关闭。结果在 `/kaggle/working/pinns_sync_latency_results.zip`。其中 `timing.json` 是实际计时，`checked.json` 仅校验输入与权重。当前交付包经本地软件核验，尚无新设备实测结果。

## 相同的设备实测命令

在 RTX 4090 工作站或 Jetson Orin NX，使用与该设备相容的 CUDA PyTorch 2.x、NumPy 和 Python 3.9+ 环境。先将本包中的私有输入展开到空目录。

```sh
python -m zipfile -e pinns_sync_latency_private.bin latency_input
python latency_input/measure_latency.py --repo latency_input/model --fixture-dir latency_input/sync --device cuda:0 --output timing.json
```

`--check-only` 可在 CPU 上验证输入哈希、人群身份、图/邻居掩码与训练权重，跳过全部前向。默认顶层使用原始 PyTorch 模型，保留运动编码和轨迹解码的两个已脚本化循环。另行运行 `--backend torchscript --output traced.json` 可以测整网按形状追踪后的表现；每个时刻和每种形状的全部输出均与原模型比对。诊断性子集例如 `--agent-counts 1,4,8,12` 只覆盖所指定的 N。保留默认全范围用于开题的 N≤12 检查。输出路径须为新文件。

公开代码测试可从包含两个 `.py` 文件的目录运行。设 `SHORTPOSE_REPO` 为完整源码检出目录，或上文展开后的 `latency_input/model`。测试中的模型仅接收合成人工特征。

```sh
SHORTPOSE_REPO=/path/to/model python -m unittest -v test_measure_latency
```

## 输入与计时口径

- 每个时刻有 12 名不同的行人。美国、日本各一个实景时刻，三档历史窗长使用相同的 24 人次和末帧。每个 N 取该时刻按原 ID 升序的前 N 人，模型的邻居仍包含其周围实际可见的人。两组 12 人分别使用 100 和 124 条有向邻边，两帧均无入选车辆。这是两个真实车道场景中的行人负载，对车辆密集或更复杂现场的代表性有限。
- T=5/10/20 是 10 Hz 的真实观测，名义跨度 0.4/0.9/1.9 s。使用源标注、冻结的姿态历史、每片首帧提取的局部图像轮廓代理，两个时刻均有 12/12 行人带有效轮廓点。每人最多 12 邻居，原标签最近五个时刻构成边特征。候选窗口沿用以完整未来标注入组的前一实验证据；送入模型的特征仅截止决策时刻。个体输出包含 6 条 30 步轨迹、候选概率和动作概率。
- 全部负载共 72 格，即 3 个 T × 12 个 N × 2 个时刻。只用同一份训练后的原始 PINNS Full T5、seed 42 权重，471,442 个参数。T=10/20 用于量化改变输入长度后的计算负载。输出保存每格含义、实际邻边数、轮廓数、前向输出摘要、设备身份、可用的驱动/功耗信息及运行时间。
- FP32，关闭 TF32 与 cuDNN benchmark。先在设备准备全部特征，使用 eval 与 inference_mode；每格预热 25 次。之后独立记录 100 次主机同步单次调用，再记录 100 次 CUDA event 单次调用。主要量为同步主机墙钟耗时，每个时刻分别存原始分布、均值、中位数、P95（线性分位数）、最小值及最大值。设备 event 提供当前流上的时间跨度。两类计时都可能受主机调度和其他负载影响，重复次数也没有增加独立场景数。36 种形状用固定种子打乱，两时刻在各形状内打乱。采集与 GPU 传输、视频解码、检测、跟踪、地图/姿态特征提取、输出回传/选模均在计时范围之外。
- 开题 §1.3.1(3) 的指定条件是单张 RTX 4090 的同帧多目标 N≤12 <10 ms、Jetson Orin NX <25 ms；另有参数与算量指标。开题没有给出耗时分位数。本入口在实际硬件匹配时附每格墙钟样本低于目标的比例以及中位数、P95是否低于目标。其他 GPU 的实测只能代表所识别设备。NX 识别同时要求 ARM64、所选 GPU 名为 Orin 系、及设备树中的 NX 型号或 P3767-0000/0001 模块；容器未暴露板卡信息时，硬件目标字段留空。使用嵌入式平台时请保留电源和散热条件，缺少工具时功耗、温度或频率可能无法获取。
- 现有 V2 记录用 Tesla T4 上固定形状合成批次和新初始化的模型。T5、N12 的中位数 14.413 ms、P95 15.110 ms。本次更换为冻结权重、两个有来源的实景同时刻，结果另存。正式报告中的 <10 和 <25 ms 仍对应各自硬件。板卡身份参考 [NVIDIA Jetson Linux Quick Start](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/IN/QuickStart.html)，CUDA event 计时定义见 [PyTorch Event](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html)。

## 输入身份

协议 SHA-256 为 `b3f33a5b8a4216fb4240d4b3412145315a510d0bd4eaad21cce542de7137f52c`；权重为 `5069c5be520f2308dd190c01cd68427ba8bdb321d405fc963f5436e911d30689`。私有 .bin 内含三份实景张量、输入协议、计时脚本和所需的已冻结原预测器源码与权重。源模型文件取自项目提交 `2a2ae0d488e5726b5172796357452965e1ad86f0`，该提交所用前向源码和冻结权重逐文件校验。归档不含摄像视频、原标注表或预测真值。逐人的真实观测张量仅用于授权的私有运行。可公开的内容是计时代码、测试和统计报告。

## 从两份原归档核对同刻输入

若持有原授权材料，在一个输入目录中保留 `shortpose_heldout.zip`、`matched_durations_gpu.zip`、`loso_rtmpose_m.onnx`、`pinns_metadata/` 和 `geometry_audit/bev_sources/`。将私有包里的 `sync/` 放在其 `latency_audit/sync/` 下。以下命令会对照原视频首帧、原标注框、坐标、关键点、真实同时刻的成员及其邻接关系，仅在输出中保留汇总。此来源复核使用 Python 3.11+、NumPy 和 OpenCV。

```sh
python reviews/pinns_latency/verify_fixture.py INPUT_DIRECTORY . /tmp/verified_fixture.json
```

[独立复算与各地点分母](verified_fixture.json) · [历史 T4 的正式报告复核](ARCHIVED_TIMINGS.md)
