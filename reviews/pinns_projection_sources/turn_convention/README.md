# PINNS 转向标签的坐标约定

原程序用带符号的航向角变化判定转向，正值记作 `left`，负值记作 `right`。本核验将 47 片、22 套标定、1,482 个完整窗的 51,870 个行人位置与原标签逐一对齐。每套标定均声明原生 BEV 横轴为 `x`、纵轴为 `-y`。在每个窗口以末观测点为原点，将其相对轨迹的每个 Y 值取相反数，对应改用 BEV 的向右、向上约定。在两套坐标下运行同一个冻结事件函数。[逐地点与合并计数](turn_convention.json) · [源坐标核验](../README.md)

原坐标五类依次为起步 49、停止 53、正向转向 214、负向转向 208、保持 975。反射后两类转向变为 208 和 214。422 个转向窗逐窗互换，另外 1,060 窗的整个事件向量保持一致，起步、停止、保持的标签各自均无差异。共有 28 窗先后达到过两个方向的持续阈值。原坐标下 16 窗先触发正向，12 窗先触发负向。冻结代码取较早事件，同刻并列时优先正向；本队列同刻并列为零。两个方向的首次持续触发时点在反射后也逐一对应。

这些计数说明已有结果的方向名称依赖坐标约定。人体解剖左右、路面物理朝向和摄像头有无镜像均需要独立证据。若仅将真值与预测同时换名，各转向专属指标应同步交换，类别等权宏平均不变；在反射输入上重新运行预测器属于另一项实验。

## 复算

在仓库根目录运行，已持有原窗口缓存时只需第二条 Python 命令。生成缓存需要仓库的数据提取依赖、作者标签和各采集条件的 `calib.json`。直接审查缓存只用 NumPy、Python 标准库和仓库内的冻结事件函数。

```bash
private_dir=/path/to/private
python -m scripts.audit_pinns_matched_histories "$private_dir/PINNS" \
  --output "$private_dir/history_support.json" \
  --manifest "$private_dir/pinns_matched_histories.npz"
python reviews/pinns_projection_sources/turn_convention/audit_turn_convention.py \
  --repo . --metadata "$private_dir/PINNS" \
  --histories "$private_dir/pinns_matched_histories.npz" \
  --out "$private_dir/turn_convention.json"
```

程序核对原事件源码 SHA-256、锚点顺序、实际用于分析的七个数组及原标签中的每个坐标。不同 ZIP 封装只要七个数组按字段名、类型、形状和值的指纹一致即可复算，输出另行记录输入文件的原始字节摘要。聚合只含规模、首次触发关系及分站总量。作者数据及输出格式见 [PINNS](https://github.com/Songan-Lab/PINNS)。
