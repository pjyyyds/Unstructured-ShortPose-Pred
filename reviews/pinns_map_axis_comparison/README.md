# 校正 BEV 轮廓坐标后的配对实验

同一批 7,283 个 PINNS 开发窗口沿用原划分和训练来源，只改变 BEV 轮廓转换到行人轨迹坐标的方式。两套地图的窗口身份、训练期参考帧、行人占用凸包张量和掩码逐项相同。测试 497 窗中，有可用影像轮廓的从 450 增至 461，11 窗新增，36 窗两版均缺少影像轮廓。测试窗中 446 个有独立的占用凸包。训练 5,686 窗的影像有无标志全都相同，具体影像几何仍随坐标修复而改变。[分区输入支持](input_support.json) · [原始坐标核验](../pinns_projection_sources/vehicle_axis/causal_map_support.json)

登记的主要量是在所有 497 个测试窗上，修复版的“加入最近影像线几何”减“仅运动和可用标志”的 ADE，单位米。另报与旧坐标同一个差值的差中之差。两者均按地点内整片重采样。独立拟合的岭模型会从有图与无图训练窗学习参数，缺图时实际执行六个连续几何量归零，并保留缺图标志。[登记方案](PROTOCOL.md) · [原坐标已发表结果](../../results/pinns_causal_geometry.json)

## 复算

在仓库根目录运行。只跑合成端到端测试需要 Python 3 和 NumPy。它检查缺图归零、修复前后独立占用槽、30 个时点的训练掩码、两种 FDE 分母和逐片误差求和。

```bash
review=reviews/pinns_map_axis_comparison
original=scripts/evaluate_pinns_causal_geometry.py
python "$review/test_comparison.py" "$original"
```

重算已登记队列须使用私有输入。已持有配对包 `pinns_map_axis_inputs.bin` 时，用下列命令核对所有窗口和来源。该命令只产生历史特征校验包。请把路径替换为自己的工作目录。

```bash
private_dir=/path/to/private
python "$review/compare_axes.py" \
  --input "$private_dir/pinns_map_axis_inputs.bin" --lock "$review/input_lock.json" \
  --original-evaluator "$original" --protocol "$review/PROTOCOL.md" \
  --output "$private_dir/checked_inputs.zip"
```

也可从原因果地图输入包、原 Kaggle 逐窗回放包、核准的新地图和作者元数据重建同一个私有挂件。新地图可用作者标签、标定和原生 BEV 视频，按以下命令形成。已验证的生成环境是 OpenCV 5.1.0、NumPy 2.5.5。其他解码器生成的边缘可能不同，程序会核对原始字节及张量指纹。旧输入包和原返回包是复核原 7,283 窗逐步预测的必要材料，仓库公开记录的是聚合。[原料与数组指纹](input_lock.json) · [PINNS 作者数据](https://github.com/Songan-Lab/PINNS)

```bash
python -m scripts.build_pinns_causal_maps \
  "$private_dir/PINNS" "$private_dir/PINNS" "$private_dir/source_label" \
  --bev-convention source_label
python "$review/prepare_private.py" \
  --legacy-input "$private_dir/pinns_causal_geometry_inputs.zip" \
  --legacy-return "$private_dir/pinns_causal_geometry_results.zip" \
  --corrected "$private_dir/source_label" --metadata "$private_dir/PINNS" \
  --evaluator "$original" --output "$private_dir/pinns_map_axis_inputs.bin" \
  --lock "$private_dir/rebuilt_lock.json"
cmp "$review/input_lock.json" "$private_dir/rebuilt_lock.json"
```

真实拟合与预测只在 Kaggle CPU 上运行。可直接导入随私有交付附上的 Notebook，也可在同时挂载代码和同一份已核验输入的 Kaggle 会话中，从仓库根目录运行以下命令。结果包中的 `aggregate.json` 提供分站、分片及配对统计，`private_replay.npz` 含拟合权重及逐窗轨迹，适合在私有目录留存。本文列出的 450 和 461 为已核验输入覆盖量。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python "$review/compare_axes.py" \
  --input "$private_dir/pinns_map_axis_inputs.bin" --lock "$review/input_lock.json" \
  --original-evaluator "$original" --protocol "$review/PROTOCOL.md" --score \
  --output /kaggle/working/pinns_map_axis_comparison_results.zip
```
