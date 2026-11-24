# 修正鸟瞰地图后的同刻多行人时延输入

美国、日本路口各取一个真实决策时刻，每处12人，历史为5、10、20帧。旧输入使用的图像边缘地图在鸟瞰纵坐标和像素分母上与源轨迹约定不同。本版从同片首帧重新提取完整场景轮廓，将像素按原图的 `u/W`、`v/H` 换算到标签地面坐标，根据每人的位置重新选择20米范围内的折线。三个时长的身份、运动历史、姿态、质量、类型和邻居等12个非地图字段均与父版本逐字节相同，地图可用标志也相同。每场仍有10条可供选择的图像轮廓、12/12人带有局部折线；美国有100条入选行人邻边、日本有124条，两个时刻均无入选车辆。图像边缘缺少人工道路语义标注。[输入来源与前后支持](sync_protocol.json)

24个人的局部图均发生变化。美国12人的折线槽总数由66增至102，38个成员新入选、2个移出。日本由108变为102，4个新入选、10个移出。同一份 Full T5 seed 42 权重用于72个负载（2个时刻 × 3档历史 × N=1…12）。当前记录覆盖输入和程序验收，尚无本版本实景前向耗时。[程序、模型与私有输入的 SHA-256](SOURCE_LOCK.json) · [先前使用的输入](../synchronized_sources.json) · [两种姿态图执行的配对协议](../../pinns_pose_compute/paired/PROTOCOL.md)

## 直接使用已授权的修正输入

在仓库根目录运行。以下名称为路径示例；`PRIVATE_INPUT_DIR` 表示存有 `pinns_sync_latency_source_label_private.bin` 的私有目录，可用相对路径替换。展开路径 `latency_input` 和结果文件使用新名称。

```sh
python -m zipfile -e PRIVATE_INPUT_DIR/pinns_sync_latency_source_label_private.bin latency_input
python reviews/pinns_latency/corrected_map/measure_latency.py --repo latency_input/model --fixture-dir latency_input/sync --check-only --output checked_source_label.json
python reviews/pinns_latency/corrected_map/measure_latency.py --repo latency_input/model --fixture-dir latency_input/sync --device cuda:0 --output timing_source_label.json
python reviews/pinns_latency/corrected_map/paired_pose_compute.py --input-root PRIVATE_INPUT_DIR --transform src/models/pose_inference.py --output paired_source_label.zip
```

第一项校验可在 CPU 上严格载入权重和真实输入，跳过所有前向。正式测量需要真实 NVIDIA CUDA 设备及与当前设备相容的 PyTorch 2.x、NumPy。原预测器保留其内部两个脚本化循环；每格预热25次，分别计100次同步墙钟和100次 CUDA event。配对程序测同批原始/优化路径，以100个平衡次序的配对差为每格原始测量，主比较固定在20帧、12人，先取每场配对中位数，再两场等权。Kaggle 的导入、返回文件和共享输入用法见[私有 Notebook 使用说明](使用说明.md)。RTX 4090、Orin NX 的目标仅在对应实机上判断。

## 从原材料复现

准备脚本需要原始私有时延输入 `pinns_sync_latency_private.bin`、作者标签与标定的根目录，以及存有官方鸟瞰视频的根目录。两处源视频须保留 `videos/<条件>/videos/bev_<索引>.mp4` 结构；文件身份见上面的输入协议。根目录参数 `.` 表示该仓库。复核归档使用 NumPy、OpenCV 与 nbformat；模型校验另需 PyTorch。逐字节重现用归档记录的 NumPy 和 OpenCV 版本，输入协议同时记录准备代码和坐标换算代码的 SHA。

```sh
python reviews/pinns_latency/corrected_map/prepare_corrected_sync.py PRIVATE_INPUT_DIR/pinns_sync_latency_private.bin METADATA_ROOT BEV_ROOT . rebuilt_sync
python reviews/pinns_latency/corrected_map/make_corrected_kit.py ORIGINAL_LATENCY_KIT.zip ORIGINAL_PAIRED_NOTEBOOK.ipynb rebuilt_sync generated_sources corrected_private_kit.zip
python reviews/pinns_latency/corrected_map/qa_corrected_kit.py corrected_private_kit.zip ORIGINAL_LATENCY_KIT.zip . METADATA_ROOT BEV_ROOT
```

`ORIGINAL_LATENCY_KIT.zip` 和 `ORIGINAL_PAIRED_NOTEBOOK.ipynb` 分别是原同刻时延交付包和原姿态图配对单元，两者的 SHA 在 `SOURCE_LOCK.json` 的 `freeze` 中。构建器复用两项原始程序，对运行所需的输入哈希和标识作固定替换。重建后的私有交付包 SHA-256 为 `94169d04af4b21f18caa783d447b9ec9f7d17fb85754378fd5b9198cad9ee3d9`。它包含两份 notebook 和一份共用私有输入。本目录保留程序、协议及聚合来源记录。含逐人张量的私有输入和配对输出仅用于授权环境中的复核。[准备真实地图](prepare_corrected_sync.py) · [打包入口](make_corrected_kit.py) · [双单元与72份地图核验](qa_corrected_kit.py)

只使用公开检出即可在仓库根目录核对程序来源和协议关系。

```sh
python reviews/pinns_latency/corrected_map/test_public_lock.py
```
