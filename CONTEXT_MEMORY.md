# 项目与对话上下文记忆 (CONTEXT_MEMORY)

> [!NOTE]
> 本文档由 `context-memory-manager` 技能自动维护，持久化记录清华大学自动化系硕士学位论文《短时姿态观测下的非结构化道路行人动作与轨迹联合预测》（彭嘉钰，导师姚丹亚教授）的核心进展、验证指标、架构决策与隐私边界（精准控制在 ~1,700 tokens）。

---

## 1. 核心目标与项目背景 (Objective & Background)
- **项目定位**: 严格对标《彭嘉钰答辩.pptx》全部 20 页设定，面向无标线混合交通场景，在因果短时历史（主设定 10 Hz 五帧 $0.4\text{ s}$，扩展 $T \in \{2, 5, 8, 10, 20\}$）与可变质量 2D/3D 姿态下，联合预测未来 $3.0\text{ s}$（`JRDB` $2.8\text{ s}$、`JTA` $4.8\text{ s}$）$K=6$ 多模态地面轨迹与五类可观测动态动作（`Start, Stop, Turn-Left, Turn-Right, Maintain`）。
- **当前阶段状态**: 已完成 **Waves 1–20 共 53 项成功云端实验归档**的双层 SHA-256 核验、**`121/121` 项本地单测全绿**、**十一幅 300-DPI 出版级图表（`fig10..20`）**、中文学位论文（`paper/ShortPose_Unstructured_Pedestrian_Paper.md`）、**IEEE T-ITS 英文投稿手稿**（`paper/IEEE_TITS_ShortPose_Manuscript.md`）、**Overleaf 即传双栏 `IEEEtran` LaTeX 投稿源码包**（`paper/ieee_tits_latex/main.tex` + `references.bib`）、**逐页精准对齐的 20 页答辩 PPT 映射表与自动生成脚本**（`paper/PPT_20Slides_Defense_Evidence_Map.md` + `scripts/build_defense_final_pptx.py`），当前云端 `0 Active Events`、`0` 草稿会话，达到完整可答辩与可投稿状态。

---

## 2. 用户偏好与全局约束 (User Constraints & Preferences)
- **算力与隐私红线**: 重算力一律在 **Kaggle 云端**执行，本地 Mac 仅运行单测、独立复核、300-DPI 绘图与文档维护；公开仓库 `pjyyyds/Unstructured-ShortPose-Pred`（`main`）**严禁上传**私有 `.npz`、新增 `.pt` 或 `.mp4`；启动器置于 `kaggle-launchers` 分支；严禁触碰 CatBox/TUN/代理设置。
- **科学诚信准则**: 主报同口径首选（`Top-1` 对 `Top-1`）；`< 0.5 m` 小位移子组始终以末两点匀速（`0.3545 / 0.8491 m`）为参照；标准分裂共形严禁校准后重训预测器；正结果、零效应（情形 C）、负向机制发现与历史失败运行（`w13-cpu1`、`374219127/374219573`、`374265835`、`374274900`、`374388893`、`374403419 rc0=1`）同等透明呈现。

---

## 3. 核心验证结论总览 (Verified Findings Across Waves 1–20)
- [x] **研究内容一（可变质量短时姿态融合、`<0.5m` 边界与 `<50 px` 可用阈值）**:
  - 真实秩-3 `3×3` 单应（`6,930/7,410` 帧朝向非零、足底残差 `2.24 µm`）下，`<0.5m` 专域拟合 `Regime_M1_ground`（`0.3538/0.8462 m`）虽较 `Regime_M0` 降 `−0.22 cm`（`[−0.31, −0.03] cm`），但对匀速（`0.3545/0.8491 m`）的 `−0.07 cm` 区间 `[−0.20, +0.26] cm` 仍跨零，匀速未被推翻。
  - `2×2` 析因揭示无约束 `MLP` 门控负秩相关（`−0.1883 / −0.2184`）；结构单调门控 **`M1_mono_q`** 将秩相关恢复至 **`+0.7301 / +0.6691`**（真 20 帧 `3×3` 矩阵 `T=5,10,20` 恢复至 **`+0.7930 / +0.8365 / +0.8674`**，`T=5 Heavy` 降 `−1.37 cm`），`20 px` 抖动降 **`−2.31 / −4.53 cm`**；像素框高 **`< 50 px`** 或 **$q_1 < 0.75$** 时直拼/无约束姿态劣化 `+3.05` 至 `+6.15 cm`，`M1_mono_q` 自动压低门控挽回 **`−1.65` 至 `−5.28 cm`**。
- [x] **研究内容二（`20 m` 折线边界、因果径向冲突门控与外部零样本迁移）**:
  - 围绕 `pt0_ego` 的 `20 m` 折线真值穿线率达 `2.36%`（有图窗 `4.19%`），`M2a` 升 `+1.58 cm`（情形 C）；无折线因果冲突门控 **`U_M2b_mono_conflict`** 在 `PINNS` 达 **`0.4510 / 1.0097 m`**（`TTC<=4s` 较匀速降 **`−5.50 / −16.16 cm`**），较 `SO(2) Frenet` 增强版 `MCENET / HST / Social-Transmotion` 显著低 **`−6.81` 至 `−14.00 cm`**；零样本迁移至 `DUT`（`3,066` 窗）与 `IMPTC`（`564` 窗）五轴摆动精确 `0.0 m`，冲突门控较无门控降 **`−3.79 cm`（有车逼近降 `−6.98 cm`）** 与 **`−1.46 cm`**。
- [x] **研究内容三（解耦五动作校准重排、不重训局部加权共形与 `Tesla T4` 效率）**:
  - 冻结 `U_M2b_mono_conflict` 控制点解耦训练五动作并重排（`U_M3_decoupled_platt_rerank`，`0.0 m` 漂移）在 `PINNS` 创最佳首选 **`0.4505 / 1.0085 m`**（较联合微调降 **`−2.20 / −4.08 cm`**，宏 AP **`0.4051`**）；`JRDB` 大折重排达 **`0.2917 / 0.5638 m`**（`ECE` 降 **18.6–23.6x**）。
  - **全程不重训分裂共形**覆盖四大基准：`PINNS` 同站 `90.60%/94.94%`（12 跨国对 `86.48%/91.74%`），`DUT` 72 跨片对 **`89.76%/95.50%`**，`IMPTC` 12 跨序列对 **`89.51%/94.21%`**；`JRDB` 72 跨地点对经源折因果速度-离散度加权升至 **`88.89%/94.20%`**（半径缩 **`−55.8%/−52.5%`**，`>=1 m/s` 快速覆盖由 `75.06%` 升至 **`89.51%`**）。`Tesla T4` `B=12` `P50` 仅 **`2.75–4.78 ms`**（算量较 `HST` 低 **396 倍**）。

---

## 4. 核心资产索引 (Core File Index)
- `paper/ShortPose_Unstructured_Pedestrian_Paper.md` & `paper/IEEE_TITS_ShortPose_Manuscript.md`: 中英文双语完整手稿（含连续 `[1]–[20]` 引文、表 1–10 / Tables I–VI 与图 10–20）。
- `paper/ieee_tits_latex/main.tex`, `paper/thuthesis_latex/main.tex` & `scripts/build_overleaf_bundle.py`: 英文双栏 `IEEEtran` 期刊与中文 `XeLaTeX`（`ctexrep`）硕士学位论文双套 Overleaf 即传源码包与一键打包脚本。
- `paper/PPT_20Slides_Defense_Evidence_Map.md` & `scripts/build_defense_final_pptx.py`: 严格对齐原始 Slide 01–20 标题的证据链映射与 20 页清华紫结题答辩 PPTX 动态生成器。
- `tests/test_manuscript_json_consistency.py`, `tests/test_ieee_latex_submission.py`, `tests/test_thuthesis_latex_submission.py`, `tests/test_defense_pptx_and_map.py`: 中英双语 Markdown、双套 LaTeX、综述与 20 页答辩 PPTX 同 53 份验证 JSON 逐位数值/版面/引文一致性单测。

## 5. 最近交互提要 (Recent Interactions)
- `Turn 2`: 新增双栏 `IEEEtran` 投稿包 `paper/ieee_tits_latex/` 与 `test_ieee_latex_submission.py`（提交 `906271a`）。
- `Turn 3`: 将 `PPT_20Slides_Defense_Evidence_Map.md` 对齐原始 Slide 01–20，新增 20 页结题答辩 PPTX 生成器与单测（提交 `b0c3db7`）。
- `Turn 4`: 同步校准中文学位论文摘要/结论并新增 `XeLaTeX` 源码包 `paper/thuthesis_latex/`（提交 `38c073a`）。
- `Turn 5`: 用户明确下一阶段聚焦核心问题“遮挡后刚初现时五帧姿态能否比只看位置更早判断起步/停止/转向”并要求多用不同数据集 -> 查明 `low_damp` 与 `cv_low_bonus` 导致小位移下“没生成+没选中”及跨站混用动作阈值、重复裁剪漏线两处代码缺陷；新增 `O2Maneuver6SlotPrecursorNet`（显式六槽假设+时序姿态先兆+源折精度约束切换）与盲审遮挡初现分级、双判据实体障碍及 `PINNS + JRDB + JTA-228 + JAAD-2.0 + ETH/UCY + DUT + IMPTC` 多数据集评测套件（提交 `8b839a1`），通过本地 PyTorch 单测并启动并行 Kaggle `GPU T4 x2` Wave 21 作业。
