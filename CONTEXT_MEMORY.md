# 项目与对话持久化上下文记忆 (CONTEXT_MEMORY)

> [!NOTE]
> 本文档由 `context-memory-manager` 滚动维护，记录《短时姿态观测下的非结构化道路行人动作轨迹预测》核心规格、代码排雷记录、实证数据与算法演进决策。
> 更新时间：2026-11-24 16:27

---

## 1. 核心目标与硬性技术边界 (Core Objectives & Boundaries)

- **课题定位**：面向无车道线、人车混行、近场遮挡的非结构化道路，仅利用 $t \le t_0$ 的地面历史与 2D 骨骼（$T \in \{5, 10, 20\}, \Delta t = 0.1\text{ s}$），预测未来 $T_f = 30$（$3.0\text{ s}$）的 $K=6$ 条地面候选轨迹 $\hat{\mathbf{Y}}$、模态概率 $\hat{\boldsymbol{\pi}} \in \Delta^5$ 及五类可观测动作概率 $\hat{\mathbf{p}}_{\text{act}} \in [0,1]^5$（绝不预测未来人体骨骼）。
- **硬指标红线**：主干参数量 $< 2.5\text{ M}$（V2 完整版 `Ours_Full` 实测仅 **`0.471 M` / `471,442`**）、$N=12$ 单次前向 $< 0.80\text{ GFLOPs}$（实测 $T=5$ 为 **`0.409 GFLOPs`**，$T=10$ 为 **`0.539 GFLOPs`**，$T=20$ 为 **`0.798 GFLOPs`**）；严禁跨帧线性插值；同机位切分强制 $\ge 6.0\text{ s}$ 物理时间隔离带。
- **真实数据切分实证底座（100% 锁定）**：
  - `PINNS`（4 国 47 段 10.0 Hz 高清监控）：共 `7,283` 条无泄漏滑窗样本（`train=5,686, val=1,100, test=497`），测试集三桶均衡（`Clear=111, Partial=252, Heavy=134`），五类动作全覆盖（有效动作 `277` 条：`Start=43, Stop=29, Turn-Left=25, Turn-Right=10, Maintain=178`）。
  - `JTA`（原生 2.5 Hz 跨集遮挡/噪声验证）：共 `3,604` 条滑窗样本（`train=1,844, val=800, test=960`）。

---

## 2. 本地代码排雷与数学不变量审计记录 (Code Audit & Bug Fixes)

遵循“慢即是快”原则，在本地 `tests/test_invariants.py`（12 项单测 `12/12 OK`）中提前捕获并根治 6 处隐患：
1. **跨 $\pm\pi$ 分支切割的伪转向消除**（`src/data/actions.py`）：废弃全局 `unwrap(arctan2)`（经过静止帧 `(0,0)` 会产生 $\pm 180^\circ$ 假跳变），改为仅在平滑速度 $\ge 0.50\text{ m/s}$ 的连续运动段内局部解缠绕，实测 $0^\circ \sim 359^\circ$（含 $179.5^\circ, 180^\circ, 180.5^\circ$）旋转严格不变。
2. **遮挡关节零空间传染与全缺测硬回退**（`pose_stgcn.py`, `quality_gate.py`）：ST-GCN 采用有效邻域动态度归一化并在 `LayerNorm` 外乘有效指示位；$q_1=0$ 时 `M1a/M1b/M1/M1p` 的 $g \equiv 0, \mathbf{h}_p \equiv \mathbf{0}$，比特级退化至纯运动表征；且 `M1b` 与 `M1`、`M2b_norad` 与 `M2b`、`M3b` 与 `M3` 参数量逐个整数精确相等。
3. **IEEE-754 `0 * NaN` 反向梯度污染根治**（`env_and_gatv2.py`）：设计 `_safe_masked_softmax`，在孤立行人（$|\mathcal{N}_i|=0$）或有图空旷广场（$m_{\text{map}}=1, 0$ 折线）样本上彻底消除全 `-inf` Softmax 的反向 `NaN` 梯度。
4. **邻近缺测速度保护**（`scene_graph.py`）：当 $m_{\text{spd},ij}=0$ 时强制置零 $\Delta\mathbf{v}_{ij}$ 与径向逼近速度 $v_{\text{rad}}$，防止单帧初现目标被误算为静止障碍。
5. **解码静态上下文预投影与死参数清除**（`action_and_decoder.py`, `env_and_gatv2.py`）：将 Eq. (9) 中 30 步恒定的 112 维静态输入门提至循环外单次矩阵乘（数学严格等价 `GRUCell(114,128)`，输入门提速 $16.7\times$），结合深度可分离时间卷积并移除 `GATv2` 第 2 层未使用的死分支 `nbr_ffn`，确保 100% 参数均有真实梯度且全窗长满足 `< 0.80 GFLOPs`。
6. **`PINNS` 数据集足底接地锚点重大审计修正**（`pinns_extractor.py`）：真机抽帧实测发现 `PINNS` 官方 `README` 标注的 `(ox, oy)` 并非几何框中心 `(cx, cy)`，而是用于单应矩阵地面投影的**足底接地中心 `(u_foot, v_bottom)`**。修正裁剪中心为 `(ox, oy - 0.5 * h)` 后，避免了头肩先兆区域被截断，使美/意/泰多国监控视频的 `RTMPose-m` 有效关节检出率跃升至 `100%`、平均置信度翻倍（`0.20~0.29 -> 0.47~0.58`）。
7. **压缩包 `__MACOSX` 字典序抢占与 `JTA` `540p` 尺度校准**（`jta_builder.py`, `cloud_experiment_runner.py`）：过滤 `Social-Transmotion` 归档中的 `__MACOSX` 资源分支，并将 `JTA` 分辨率尺度对齐至发布的 `960x540` 坐标系。
8. **`PINNS` 短视频分层无泄漏切分与 Turing 双卡推理加速**（`splits.py`, `dataset_builder.py`, `pinns_extractor.py`，`13/13` 单测通过）：针对 `PINNS` 47 段 `20~35s` 短片段内插双隔离带导致末端 `test` 被挤空的实测发现，升级为按 4 处跨国路口的时间序片段比例（`70%/15%/15%`）划分独立视频段 + 跨子集首段起始 `6.0s`（60 帧）时间隔离带 + 仅由 `train` 片段冻结生成静态折线；同时针对 `cuDNN 9.27` 在 `sm_75` 上拒绝 `FusedConv` 的问题，构建 `onnx2torch` 双卡 `Tesla T4` 并行推理（含 `<5e-2` 精度自动对账与 `ORT_ENABLE_BASIC` 兜底）及 `extracted_samples_v1.pt` 缓存。

---

## 3. 首轮 V1 实证诊断发现与 V2 数学演进（已归档 `results/v1_initial_diagnosis/`）

1. **V1 已证实的核心科学结论**：
   - `M1` 在 `PINNS T=5` 显著优于等权拼接 `M1a`（`0.4896 vs 0.4950`, Holm $p=0.0374$），在 `JTA` 重度噪声桶相比 `M1a` 降低误差 **`-11.62%`**（`1.5176 vs 1.7171`, Holm $p=1.49\times 10^{-7}$，达 MPSES），且优于全部三项外部基线（`MCENET=0.5234`, `HST=0.5265`, `Social-Transmotion=0.5152`）；
   - `M3`（`Stop-Grad`）将动作 `Macro-F1` 从端到端 `M3b` 的 `30.85%` 提升至 **`37.29%`**，`M3p`（`ACMC`）显著优于 `M3`（`0.5771 vs 0.5926`, Holm $p=8.61\times 10^{-5}$）。
2. **V1 暴露的四项机制瓶颈与 V2 闭环改进（提交 `c5344f1`）**：
   - **质量瓶颈能量均衡与去死区 `BPD-Gate`**：将 4 维 $\mathbf{q}$ 升维至 32 维与 32 维运动-姿态投影对等融合，引入可观测性包络 $\sqrt{q_1 q_2}(0.4+0.6q_4)$，并用生物力学凸权重 `(0.45, 0.30, 0.25)` 替代 $\max(1,\sum g_r)$ 死区分母；
   - **自车恒等主干保护 + 空汇点（Null-Sink）`GATv2`**：引入可学习空汇点逻辑值使远距无冲突旁观者注意力自然衰减至零，配合训练期 `6 cm` 折线微扰消除固定路口坐标死记硬背（根治 V1 中 `minADE_6=0.33m` 不变但 `NLL` 升至 `1.72` 的模态重排序退化）；
   - **去 `LayerNorm` 动作条件 + 距离软排序模态损失 + TorchScript C++ 融合循环**：消除截断动作概率微抖动放大，引入 $\text{softmax}(-\text{ADE}_k/0.15)$ 软排序监督与 C++ GRU 核函数融合以压缩 `Tesla T4` 推理延迟。

---

## 4. 核心文件索引与当前阶段 (Key Files & Active Stage)

- **核心源码与测试**：`src/data/`、`src/models/`、`src/losses/losses.py`、`src/metrics/metrics_and_stats.py`、`tests/test_invariants.py`（`13/13` 通过）。
- **独立 GitHub 仓库**：`https://github.com/pjyyyds/Unstructured-ShortPose-Pred`（已备份 V1 全量结果与 V2 源码，提交 `c5344f1`）。
- **云端执行通道**：Kaggle V2 任务 `https://www.kaggle.com/code/pengjy25/unstructured-shortpose-exp-v1cc0fd051f7`（`scriptVersionId=373403739`，已挂载 `extracted_samples_v1.pt` 秒级跳过视频提取，正在 `GPU T4 x2` 双卡运行全部 242 组实验）。
