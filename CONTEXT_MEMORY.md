# 项目与对话持久化上下文记忆 (CONTEXT_MEMORY)

> [!NOTE]
> 本文档由 `context-memory-manager` 滚动维护，记录《短时姿态观测下的非结构化道路行人动作轨迹预测》核心规格、代码排雷记录、实证数据与算法演进决策。
> 更新时间：2026-11-23 23:09

---

## 1. 核心目标与硬性技术边界 (Core Objectives & Boundaries)

- **课题定位**：面向无车道线、人车混行、近场遮挡的非结构化道路，仅利用 $t \le t_0$ 的地面历史与 2D 骨骼（$T \in \{5, 10, 20\}, \Delta t = 0.1\text{ s}$），预测未来 $T_f = 30$（$3.0\text{ s}$）的 $K=6$ 条地面候选轨迹 $\hat{\mathbf{Y}}$、模态概率 $\hat{\boldsymbol{\pi}} \in \Delta^5$ 及五类可观测动作概率 $\hat{\mathbf{p}}_{\text{act}} \in [0,1]^5$（绝不预测未来人体骨骼）。
- **硬指标红线**：主干参数量 $< 2.5\text{ M}$（当前完整版 `Ours_Full` 实测仅 **`0.495 M`**）、$N=12$ 单次前向 $< 0.80\text{ GFLOPs}$（实测 $T=5$ 为 **`0.410 GFLOPs`**，$T=20$ 为 **`0.799 GFLOPs`**）；严禁跨帧线性插值；同机位切分强制 $\ge 6.0\text{ s}$ 物理时间隔离带。
- **云端资源状态**：Kaggle (`pengjy25`, Phone Verified, `30h/30h` 免费 GPU 满额) 与 Google Colab (`Google One`) 已在内置浏览器完成核验；`PINNS`（`973 MB` Google Drive 直链）与 `Social-Transmotion/JTA`（`306 MB` GitHub Release）实测免登 HTTP 200 直达。

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

---

## 3. 论文级算法升级假设（待云端实证检验）

除严格实现开题报告 `M0`–`M3` 及外部基线 `MCENET`/`HST`/`Social-Transmotion` 外，针对审稿高频质疑预埋两项微增量（共 `< 700` 参数）物理改进：
- **`M1p` (`BPD-Gate`, 生物力学部位解耦质量门控)**：针对车顶露头等“下肢遮挡、头肩清晰”高频工况，分头肩导向、躯干、下肢步态三组独立评估质量并以 $\max(1, \sum g_r)$ 凸归一聚合，避免全局标量门控因下肢遮挡一刀切丢弃头肩转向先兆。
- **`M3p` (`ACMC`, 动作运动学一致性模态校准)**：提取 $K=6$ 候选轨迹的 6 维可微物理摘要（初末速度差、横向转向正弦、曲率比）与预测动作概率做兼容度校准（保持 `stop_grad` 零梯度泄漏），提升短窗状态转换样本的首选模态 `ADE_1` 命中率。

---

## 4. 核心文件索引与当前阶段 (Key Files & Active Stage)

- **核心源码**：`src/data/`（特征/动作/折线/防泄漏切分）、`src/models/`（`M0–M3`、`M1p/M3p`、`Ext-1..3`）、`src/losses/losses.py`、`src/metrics/metrics_and_stats.py`、`tests/test_invariants.py`（12/12 通过）。
- **独立 GitHub 仓库**：`https://github.com/pjyyyds/Unstructured-ShortPose-Pred`（SSH 别名 `github-shortpose`，已开启自动里程碑推送）。
- **云端执行通道**：Kaggle Notebook `https://www.kaggle.com/code/pengjy25/unstructured-shortpose-exp-v1`（通过 GitHub 外层直连导入，绕开跨域编辑子帧；`Version #2` 正在 `GPU T4 x2` 双卡后台 Commit 运行 Suites A–E，本机保持零负载）。
