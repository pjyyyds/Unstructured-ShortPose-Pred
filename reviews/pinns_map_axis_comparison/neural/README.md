# 校正后的因果影像轮廓与 M2a 预测

两个模型只在同一局部影像折线的 X/Y 通道上有差异。对照保留全部逐点有效位、线条类型、顺序与来源可用位，但坐标置零。实验臂使用校正后的真实坐标。二者调用同一仓库 M2a 训练器；该入口记录训练器实际生成的初始权重摘要、每批抽到的窗口索引摘要和七次验证 ADE，并在发布配对结果前核对两臂初始状态及训练抽样序列。[预注册](PROTOCOL.md)

网络既有的训练增强会给有效点的 X/Y 分别加标准差 0.06 m 的高斯噪声。两臂共享这一规则。对照臂在增强前、验证和测试时均为精确零坐标，训练时其坐标为零均值扰动。[地图编码实现](../../../src/models/env_and_gatv2.py)

核验了作者资料的 7,283 个窗口、完整历史和未来真值的原始指纹，以及每个输入折线及来源标志。497 个测试窗来自 21 片，461 窗有影像轮廓，36 窗无图；233 窗在恰好 3 秒处有真值。此处公开的是输入支持。[输入核验](input_check.json)

## 复现

在已安装项目依赖的仓库根目录运行。第一条命令只用合成输入进行小规模测试，包括调用原训练器、确认相同初始化和抽样、恢复保存的两臂产物及测试首选轨迹计分。第二条可用作者标签、各条件的标定与已冻结的私有影像地图独立核对窗口，无需姿态提取或真实模型前向。

```bash
python -m reviews.pinns_map_axis_comparison.neural.test_neural
python -m reviews.pinns_map_axis_comparison.neural.check_inputs \
  /path/to/PINNS /path/to/source_label/maps.npz /path/to/private/input_check.json
```

Kaggle CUDA 短版入口只运行两臂。将下列调用置于同一个 Python 进程，传入已经完成原始图像姿态提取、经三元键匹配的全体窗口。应在启动该进程前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。可在首次训练前调用 `check_samples(aligned)` 验证地图、窗口输入和分区。正式训练另行记录每个划分的姿态覆盖与全部训练张量指纹。同一个窗的姿态可以全部缺失。若训练、验证或测试整组都没有有效关节，先核对原始视频和提取缓存。成对结果在 `/kaggle/working/pinns_neural_image_coordinates/pinns_neural_image_geometry_pair.zip`。

```python
from src.data.pinns_causal_input import attach_causal_image
from reviews.pinns_map_axis_comparison.neural.run_neural import run

aligned = attach_causal_image(pinns_samples, corrected_maps_path)
aggregate = run(aligned, "/kaggle/working/pinns_neural_pair")
print(aggregate["primary_test_paired_ade"])
```

最终文件是该输出目录中的 `pinns_neural_image_geometry_pair.zip`。可公开的 `aggregate.json` 包含首选 ADE 配对差的区间、两种 FDE 的分子与有效分母、分站及分片汇总。`test_rows.npz` 含私有测试键、历史与未来掩码及实测轨迹，两份臂文件各自包含最佳验证权重、六条测试轨迹、全部候选概率及训练记录。整个 ZIP 应作为私有产物处理。

入口在每臂完成后校验并原子保存。用同一输入、代码、配置和工作机重跑时，完整的臂直接复用；未完成的臂从原种子重启。归档校验失败会保留原文件并重建。工作机或输入身份不同的部分运行分开存放。
