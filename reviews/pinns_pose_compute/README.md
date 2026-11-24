# 冻结模型的姿态图推理优化

给已载入参数的模型调用一次 `prepare_pose_inference`。返回值拥有独立存储，适合在部署或配对评测前准备。默认训练和旧权重的原始入口维持归档实现。

```python
import torch
from src.models.pose_inference import prepare_pose_inference
from src.models.predictor import UnstructuredShortPosePredictor

original = UnstructuredShortPosePredictor("Ours_Full").eval()
weights = torch.load("checkpoints/PINNS_T5_Ours_Full_seed42.pt",
                     map_location="cpu", weights_only=True)
original.load_state_dict(weights, strict=True)
optimized = prepare_pose_inference(original)
with torch.inference_mode():
    outputs = optimized(batch)  # 与原模型相同的特征字典
```

邻接矩阵 A 在关节轴上聚合，投影 W 沿通道轴扩张，二者中间没有非线性。以 (AH)W 代替 A(HW) 后，三层的邻接乘法分别使用 6、32、64 个输入通道。以一次乘加计两次运算，每个目标、每个历史时刻节省 2 × 17² × (26 + 32 + 64) 次。浮点重排可引入舍入差异。

同批 12 人时，5、10、20 帧分别节省 4,230,960、8,461,920、16,923,840 次计算。20 帧下，只计矩阵乘和卷积为 781,306,176 次；另计逐元素浮点加减乘为 790,639,788 次。两者都未覆盖完整前向的全部算术。在扩展口径下，相对原执行顺序减少 2.10%。[预先登记的比较规则](PROTOCOL.md) · [独立逐算子清单和遗漏项](independent/REVIEW.md)

同一份实际训练权重的 161 项参数及缓冲状态逐张量相同。CPU FP32 下，三档历史和四种合成输入共比较 144 项行人输出。六条轨迹坐标最大差小于 0.000001 m，候选概率与动作概率最大差均小于 0.0000002，首选索引变化为零。该比较覆盖部分关节缺失、全缺姿态和空环境等情形。两处实景同时刻、目标 GPU 的排序和耗时需用相同输入另测。[逐输出复算](compute_scores.json) · [实景输入与设备范围](../pinns_latency/README.md)

在仓库根目录复算合成结果可运行下列命令，输出写入指定路径。

```sh
python reviews/pinns_pose_compute/check_compute.py /tmp/pose_compute.json
python reviews/pinns_pose_compute/independent/check_compute.py . /tmp/independent_pose_counts.json
```

真实同刻配对入口固定比较美国和日本两个相同时刻。在每种负载交错进行 100 对同步计时，主要比较 20 帧、12 人条件下两个现场等权的场内配对中位差。[实景数值与耗时配对协议](paired/PROTOCOL.md)
