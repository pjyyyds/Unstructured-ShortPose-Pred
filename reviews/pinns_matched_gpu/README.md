# PINNS 同窗长四地 GPU 融合对照

Kaggle 版本 373591607 含 24 份权重。每个地点轮流留出，另三地用于训练；同一批 623 个窗分别取 5、10、20 帧历史，随后均有连续 30 步真值。两个融合臂都用种子 42 训练 28 轮。M1b 是隐式门控，M1-concat 将运动和姿态表示按通道拼接，再以可学习的无偏置投影降回 128 维。

四地等权的首选轨迹 ADE 在 5、10、20 帧时，M1b 分别为 0.6069、0.5772、0.5625 m，M1-concat 为 0.5928、0.6210、0.6019 m。两臂的 ADE 差依次为 +0.0141、−0.0438、−0.0393 m，其 95% 区间分别为 [−0.0132, +0.0474]、[−0.0881, +0.0081]、[−0.0826, +0.0178] m。FDE 的三个区间同样跨零。各地分别对整段录像有放回抽样，并保留片内全部窗口，四地等权，种子 2026，重复 20,000 次。区间只覆盖这些固定模型和四个开发地点内的片段变异。[逐地结果、融合组与基线组的全部配对差](gpu_fusion_summary.json)

M0 和显式门控 M1 来自另一保存版本 373585040。与新融合臂比较时，训练划分及测试窗严格对齐，但两项作业的物理工作机身份没有共同记录。跨作业差值只能在此条件下解释。M0/M1 包只保存首选轨迹误差，此处复核它的窗口和聚合。该组此前的权重回放见[原运动与姿态报告](gpu_motion_pose_summary.json)。

## 从原始归档复算

依赖 Python、NumPy 和能运行 `torch.load(weights_only=True)` 的 PyTorch。将下面四个环境变量设为私有输入路径。脚本逐项验证 SHA-256，在 CPU 上读取权重张量及保存的六条候选，不调用模型前向。运行时无需网络或 Git 历史。

- `FUSION_ZIP` 指向 373591607 的原始 `matched_fusion.zip`。
- `BASE_ZIP` 指向 373585040 的原始 `matched_durations.zip`。程序直接从中读取 `matched_observations.npz`。
- `HISTORIES_NPZ` 指向冻结的 `pinns_matched_histories.npz`，SHA-256 为 `caa8f0959f28f2d69a6f16cc9df115ab6222d0b6eb3297edd575615fef81b94c`。仓库中 `scripts/audit_pinns_matched_histories.py` 的 `--manifest` 选项可生成该队列明细。
- `PINNS_METADATA` 指向包含 `labels/` 的 PINNS 原始标注根目录。

在当前仓库根目录运行：

```sh
python reviews/pinns_matched_gpu/verify_saved_fusion.py \
  --fusion "$FUSION_ZIP" --base "$BASE_ZIP" \
  --histories "$HISTORIES_NPZ" --metadata "$PINNS_METADATA" \
  --output reproduced.json
```

从其他目录调用时，使用脚本的绝对路径。程序默认从自身所在仓库检查六个训练脚本的字节摘要；可用 `--source-tree` 显式指定冻结提交 `a2eeaf01568d74f7a01a0f74a077632e33e41b47` 对应的源码树。复算遍历 47 个片段的原标签，检查 623 × 50 个地面位置和每折的训练、测试键，按六条候选原有概率选取轨迹。全部 3,738 组预测的首选 ADE/FDE 与保存误差最大相差 0.000000754 m。公开 JSON 仅含指纹和聚合计数、误差、区间。
