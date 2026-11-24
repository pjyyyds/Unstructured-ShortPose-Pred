# 晨间审查总览报告：短时姿态观测下的非结构化道路行人动作轨迹预测

> **项目仓库**：`https://github.com/pjyyyds/Unstructured-ShortPose-Pred`
> **云端算力**：Kaggle 双卡 `NVIDIA Tesla T4 x2`（零本机负载，累计完成 V1 + V2 两轮共 **484 组** 5 随机种子独立训练与评测）

---

## 1. 核心交付物清单与目录导航

- **完整学术论文初稿**：[`paper/ShortPose_Unstructured_Pedestrian_Paper.md`](../paper/ShortPose_Unstructured_Pedestrian_Paper.md)（含中英文摘要、方法公式推导、V1$\to$V2 演进剖析、配对假设检验、5 幅 300-DPI 论文插图与 24 篇标准文献）
- **5 幅 300-DPI 高清出版级图表**：[`docs/figures/`](../docs/figures/)（`fig1` 多窗三桶矩阵、`fig2` V1$\to$V2 演进与单调门控响应、`fig3` 无标线几何与梯度阻断、`fig4` 多窗外部 SOTA 与 `JTA` 泛化、`fig5` `Tesla T4` 实时延迟与敏感性）
- **双版本全量实证数据与检验报表**：
  - V2 正式版结果：[`results/EMPIRICAL_BENCHMARK_REPORT.md`](../results/EMPIRICAL_BENCHMARK_REPORT.md)、[`results/master_experiment_report.json`](../results/master_experiment_report.json)
  - V1 诊断基线归档：[`results/v1_initial_diagnosis/EMPIRICAL_BENCHMARK_REPORT.md`](../results/v1_initial_diagnosis/EMPIRICAL_BENCHMARK_REPORT.md)
  - 代表性模型权重：[`checkpoints/`](../checkpoints/)（`PINNS_T5_{M0, M1, M2, M3, Ours_Full}_seed42.pt`）
- **模块化源码与 13 项数学不变量单测**：[`src/`](../src/)、[`tests/test_invariants.py`](../tests/test_invariants.py)（`13/13 OK`）
- **持久化记忆文档**：[`CONTEXT_MEMORY.md`](../CONTEXT_MEMORY.md)

---

## 2. 代码深度审查与自主排雷记录（共根治 9 项底层隐患）

1. **`PINNS` 官方标注足底接地锚点重大纠偏**（`pinns_extractor.py`）：抽帧核验发现官方标注 `(ox, oy)` 实为用于单应地面投影的足底中心 `(cx, y_bottom)` 而非几何框中心；修正裁剪框中心为 `(ox, oy - 0.5*h)` 后，避免头肩转向先兆被截断，使四国监控视频 `RTMPose-m` 17 关节检出率达 `100%`、置信度翻倍（`0.20~0.29 -> 0.47~0.58`）。
2. **短视频分层无泄漏切分与 6.0s 物理隔离带**（`splits.py`, `dataset_builder.py`）：针对 `PINNS` 47 段 `20~35s` 短片段，按 4 处跨国路口时间序 `70%/15%/15%` 划分独立片段并叠加首段 `6.0s` 隔离带，静态路缘/驻车折线严格仅由训练片段提取冻结，区间扫描验证跨集时间重叠严格为零（`train=5686, val=1100, test=497`）。
3. **跨 $\pm\pi$ 分支切割伪转向消除**（`actions.py`）：改为仅在连续运动段（速率 $\ge 0.50\text{ m/s}$）局部解缠绕，确保 $0^\circ \sim 360^\circ$ 任意旋转严格不变。
4. **Turing `sm_75` 下 `cuDNN 9.27` 融合算子容错与缓存**（`pinns_extractor.py`）：启用 `ORT_ENABLE_BASIC` 避开不支持 `sm_75` 的 `FusedConv` 引擎，5.4 分钟完成全量 47 段视频 GPU 骨骼提取并持久化 `167 MB` 特征缓存。
5. **V1 $\to$ V2 四项机制瓶颈自主诊断与根治**：
   - *质量向量淹没与分母饱和死区* $\to$ 对称 32D 质量瓶颈 + $\sqrt{q_1 q_2}(0.4+0.6q_4)$ 包络 + 凸生物力学权重，使门控从平坦 `0.27` 转化为严格单调递减（`0.135 -> 0.116 -> 0.078`）；
   - *静态路口坐标死记硬背与旁观者强制注意力求和为 1* $\to$ 自车恒等主干保护 + 训练期 `6 cm` 折线微扰 + **可学习空汇点 `Null-Sink GATv2`**，将 `M2`/`M3`/`Ours_Full` 误差整体削减 **$-14.0\% \sim -14.8\%$**（同时天然根除孤立行人 `0 * NaN` 反向梯度）；
   - *截断动作概率经 `LayerNorm` 抖动放大* $\to$ 平滑 `Tanh` 零均值确定度映射 + 距离软排序模态损失；
   - *Python 循环核函数启动开销* $\to$ 静态门控预投影 + TorchScript C++ 融合循环，`Tesla T4` 12 人并发延迟降至 **`14.49 ms`**（峰值显存仅 `31.5 MB`，参数量 `0.471 M < 2.5 M`，算力 `0.409~0.798 GFLOPs < 0.80 GFLOPs`）。

---

## 3. 对照开题报告第 5.3 节客观判读准则的核心结论

- **研究内容一（明确成立，兼具情形 A/B 强显著性）**：等权直拼 `M1a` 在 `PINNS`（`+1.50%`, Holm $p=0.0209$）与 `JTA`（`+48.62%`, Holm $p=2.99\times 10^{-126}$）均显著退化；显式质量门控 `M1` 彻底消除重度遮挡负迁移（`PINNS` 重度桶 $p=0.589$；`JTA` 重度噪声较 `M1a` 大降 **`-34.36%`**, Holm $p=8.62\times 10^{-24}$，达 MPSES），且在 `PINNS` 全三窗长（`0.4783 / 0.4660 / 0.4704 m`）全面超越 `MCENET`、`HST` 与 `Social-Transmotion`。
- **研究内容二（条件有效 + 固定机位边界诚实界定）**：局部折线 `M2a` 在状态转换子集取得全部 17 组模型最优（`0.7078 m`），联合模型 `M2` 显著优于仅图模型 `M2b`（`-4.01%`, Holm $p=0.00958$），且全系统在场景更丰富的 `JTA` 上超越 `HST` 与 `Social-Transmotion`；同时如实指出在仅含 4 处固定单应视角的 `PINNS` 全局均值上，个体融合 `M1` 因不受早晚高峰主导流向偏移影响而保持最低全局误差。
- **研究内容三（梯度阻断核心假设双轮一致证实）**：`M3`（`Stop-Grad`）在 V1 与 V2 两轮实验中动作宏平均 `Macro-F1` 均稳定高出端到端 `M3b` 约 **`6.0 ~ 6.4` 个百分点**，并在 V2 中同步改善轨迹精度与避障越界率。
