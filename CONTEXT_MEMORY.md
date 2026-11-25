# 项目与对话上下文记忆 (CONTEXT_MEMORY)

> [!NOTE]
> 本文档由 `context-memory-manager` 技能自动维护，持久化记录清华大学自动化系硕士学位论文《短时姿态观测下的非结构化道路行人动作与轨迹联合预测》（彭嘉钰，导师姚丹亚教授）的核心进展、验证指标、架构决策与隐私边界（控制在 ~1,500 tokens）。

---

## 1. 核心目标与项目背景 (Objective & Background)
- **项目定位**: 面向无标线混合交通场景，在严格因果短时历史（主设定 10 Hz 五帧 $0.4\text{ s}$，扩展 $T \in \{5, 10, 20\}$）与可变质量二维姿态下，联合预测未来 $3.0\text{ s}$（$T_f=30$）$K=6$ 多模态地面轨迹及五类可观测动作（`Start, Stop, Turn-Left, Turn-Right, Keep`）。
- **当前阶段目标**: 已完成跨机位正交奇偶等变模型升级、外部多视角实拍泛化与 Tesla T4 实景计时闭环，正等待最后一项修正正向鸟瞰地图的五种子完整系统套件（`374033219`）跑完归档。

---

## 2. 用户偏好与全局约束 (User Constraints & Preferences)
- **算力与隐私红线**: 所有训练、姿态推理与 GPU 计时必须在 **Kaggle 云端**执行，本地 Mac 仅运行单测、独立复核（`verify_*.py`）、绘图与文档维护；公开仓库 `pjyyyds/Unstructured-ShortPose-Pred`（`main` 分支）**严禁上传**私有逐人坐标缓存（`*.npz`）、权重（`*.pt`）或原始视频，仅提交源码、单测、300-DPI 图表与聚合 JSON/`RESULTS.md`；纯代码云端启动器置于分支 `kaggle-launchers`。
- **科学诚信与预算边界**: 如实报告全部正结果、零效应（情形 C）与负结果及整片/按人自举 95% 区间；禁止预测未来骨架；总参数 $<2.5\text{ M}$（`Ours_Full = 0.471 M`，`O2BioEqJointForecaster = 0.017 M`），$T=20, N=12$ 算量经 `(AH)W` 改写后为 `790,639,788`（`< 0.80 GFLOPs`）。

---

## 3. 当前系统状态与最新进展 (Current State & Progress)
- **已完成并经本地逐权重/逐点独立复核的 17 项 Kaggle 云端实验（89/89 单测全绿）**:
  - [x] **跨站点正交等变主突破（`w1-cpu2`、`373987289`）**: 针对 `Social-Transmotion` 加框版（`0.6522 m` vs 仅轨迹 `0.5534 m`）与原五种子 `M1`（`0.5375 m` vs `M0` `0.5126 m`）的跨机位退化，提出因果局部航向正交等变 + 源站点等权 + 两阶段冻结 `scores` 纵向生物力学微调 `O(2) BioEq v2`，在 1,482 全窗取得 **`0.4593 m` ADE / `1.0332 m` FDE**（较末两点匀速 `0.4839/1.1046 m` 降低 `-5.07%/-6.46%`，配对优于同种子等变纯运动 `-0.324 mm` ADE `[−0.700, −0.030] mm` 且消除负迁移）；323 短位移窗 `minFDE₆ = 0.5559 m`。
  - [x] **匹配多时长 `T=5,10,20` 五种子评测（`374021144`，24 权重回放通过）**: 623 匹配窗首选 ADE 降至 **`0.4785 / 0.4752 / 0.4803 m`**（较原 `M1` 降 `-12.5~-13.8 cm`），贝叶斯期望轨迹单调收敛至 **`0.4755 -> 0.4715 -> 0.4700 m` ADE**（`1.0885 -> 1.0796 -> 1.0739 m` FDE）。
  - [x] **五种子 `O(2)` 奇偶等变动作轨迹联合（`374021666`、`374033892`）**: 两阶段解耦 `O(2) BioEq-Act v2`（`16,674` 参数）同时取得 **`0.4577 m` ADE / `1.0265 m` FDE**（贝叶斯 `1.0230 m`）与 **`0.3548` 五类宏 AP / `0.3230` 宏 F1 / `0.1055` Brier**（较原 `detached` 提升 `-5.04 cm` ADE 与 `+3.36 pp` 宏 AP）。
  - [x] **外部多视角零样本泛化（`374037376`、`373965984`、`374000295`、`374019965`）**: 北京 `DAIR-V2X` 四路口 54 窗侧视相机上，五种子时钟感知 `O(2) BioEq` 达 **`1.5284 m` 贝叶斯 ADE / `1.5769 m` 首选 ADE**（较实钟匀速 `1.9778 m` 降 `-44.94 cm`，严格时钟 31 窗达 **`1.3101 m`** 显著优于匀速 `1.4487 m` 与 `M1` `1.5784 m`，正交旋转摆动 `0.00 cm`）；大连理工 `DUT` 3,066 窗近俯视下揭示 `97.2%` 窗无框的二维姿态物理边界，`M1` 安全回退（`0.5416` vs `M0` `0.5754 m`），时钟感知正交回归在四正交角恒为 **`0.5023/1.0136 m`**（`0.0 cm` 摆动）；`JRDB` 197 窗印证高维原三维骨骼跨片过拟合（`+5.07 cm`）。
  - [x] **鸟瞰地图修复、可见边界与 T4 实景时延（`w1-cpu1`、`373962571`、`374020642`、`373989788`、`373995305`）**: 修复 `1-v/H -> v/H` 消除约 80% 历史轮廓伪退化，神经 `M2a` 实测差 `+0.0029 m`（情形 C）；59 窗真值在 `3` 窗横穿可见折线；美日 12 人满载实景下 `Ours_Full` 在 Tesla T4 同步墙钟均值 **`14.00–15.06 ms`**（P95 `15.00–16.19 ms`），`(AH)W` 偏差 `< 2 µm`、零模态翻转。
- **当前活跃云端作业**: `w4-gpu-pinns-corrected-full-network`（`scriptVersionId=374033219`，`GPU T4 x2` 运行中）。

---

## 4. 关键技术选型与架构决策 (Decisions & Architecture)
- **奇偶孪生 `O(2)` 正交等变性**: 将局部横向坐标正负对偶 $(h_+, h_-)$ 输入共享 GRU，偶宇称控制纵向残差与 `Start/Stop/Keep`，奇宇称控制横向残差并对称映射 `Turn-Left <-> Turn-Right` 及镜像模态对 `(Mode 3, Mode 4)`，严格保证平面旋转与镜像反射双重等变。
- **源站点等权与两阶段解耦微调**: 按站点样本数倒数均衡损失消除美国路口（占 `55.1%`）主导偏差；第二阶段冻结主干与模态打分，仅微调低速状态门控生物力学节点并由折内留一验证选择收缩系数，从结构上杜绝负迁移。

---

## 5. 核心文件与资源索引 (Key Files Map)
- `src/models/bioeq_o2_joint.py` & `tests/test_bioeq_o2_joint.py`: 严格 `O(2)` 奇偶等变多时域、变时钟与五动作联合预测器及单测。
- `src/models/biomech_equivariant.py` & `src/models/bioeq_neural.py`: 局部航向 Frenet 基、视角无关生物力学特征提取与 `BioEq v2` 模型。
- `scripts/run_pinns_matched_o2_multiduration.py` & `scripts/run_pinns_o2_joint_v2_5seed.py`: 多时长与两阶段联合训练评测入口。
- `scripts/run_dair_attached_o2_bioeq.py` & `scripts/run_dut_o2_transfer.py`: `DAIR-V2X` 与 `DUT` 外部零样本正交旋转及实测时钟评测脚本。
- `paper/ShortPose_Unstructured_Pedestrian_Paper.md` & `docs/figures/fig10_*.png`, `fig11_*.png`: 完整学术论文稿与 300-DPI 核心对比图。
- `reviews/PROJECT_REVIEW_SUMMARY.md`: 全项目实验审查与复核索引总表。

---

## 6. 避坑指南与已知约束 (Caveats & Known Issues)
- **Kaggle 自动化红线**: 严禁触碰 `/edit` 内层 `kkb-production.jupyter-proxy.kaggle.net` iframe 或本地上传弹窗；一律通过 `https://www.kaggle.com/code?import=true` 的 `GitHub` 标签导入 `kaggle-launchers` 分支纯代码 Notebook，读取状态仅访问 `/output` 页（避免 `/log` 大行数卡顿）。
- **外部时标一致性**: 跨帧率数据集（`DUT` `23.976 fps`、`DAIR-V2X` 曝光时钟）必须传入实测历史步长 `obs_dt` 与未来时标 `fut_times_s`，避免名义 `0.1 s` 步长导致的系统尺度偏差。

---

## 7. 下一步行动清单 (Next Steps)
- [ ] 待 `w4-gpu-pinns-corrected-full-network`（`374033219`）跑完后下载独立核验，补充正向地图下五种子 `Ours_Full` 对照数值，生成最终晨间总报告。

---

## 8. 最近多轮交互提要 (Recent Turns Digest)
- **Turn N-2**: 用户要求审查 Codex 线程并接续推进、及时备份 GitHub、尽量用云端算力 -> 完成私有输入建库与 Wave 1–3 十项 Kaggle 实验验证。
- **Turn N-1**: 用户询问阶段结论并授权“一直做不要停，做到明天早上” -> 完成 `O(2) BioEq v2`、`DUT` 四角度不变性及图 10 入稿。
- **Turn N (本次)**: 通宵推进 Wave 4/5 全部五项 CPU 工程（T4 实景时延/`(AH)W`、多时长 `O(2)`、五种子 `O(2) BioEq-Act v2`、`JRDB` 五姿态、可见边界、`DAIR-V2X` 四路口 54 窗零样本突破）及图 11 入稿，同步运行 `GPU T4 x2` 修正地图完整套件。
