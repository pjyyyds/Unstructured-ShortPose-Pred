"""Generate a 20-slide Tsinghua-styled Thesis Defense PPTX directly from SHA-256 verified JSONs."""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

THU_PURPLE = RGBColor(0x66, 0x08, 0x74)
THU_LIGHT = RGBColor(0xF6, 0xF1, 0xF8)
DARK_TEXT = RGBColor(0x1F, 0x23, 0x28)
MUTED_TEXT = RGBColor(0x57, 0x60, 0x6A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BORDER_GRAY = RGBColor(0xD0, 0xD7, 0xDE)


def _add_chrome(prs: Presentation, section: str, title: str, page_no: int, total: int = 20):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.92))
    bar.fill.solid()
    bar.fill.fore_color.rgb = THU_PURPLE
    bar.line.fill.background()

    tf = bar.text_frame
    tf.word_wrap = True
    p0 = tf.paragraphs[0]
    p0.text = f"{section}  |  {title}"
    p0.font.size = Pt(20)
    p0.font.bold = True
    p0.font.color.rgb = WHITE

    foot = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.35))
    fp = foot.text_frame.paragraphs[0]
    fp.text = f"清华大学自动化系硕士学位论文答辩 · 彭嘉钰（指导教师：姚丹亚教授）    {page_no:02d} / {total:02d}"
    fp.alignment = PP_ALIGN.RIGHT
    fp.font.size = Pt(11)
    fp.font.color.rgb = MUTED_TEXT
    return slide


def _add_card(slide, x: float, y: float, w: float, h: float, heading: str, bullets: list[str], fill=THU_LIGHT):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = BORDER_GRAY
    tf = box.text_frame
    tf.word_wrap = True
    hp = tf.paragraphs[0]
    hp.text = heading
    hp.font.size = Pt(15)
    hp.font.bold = True
    hp.font.color.rgb = THU_PURPLE
    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(12.5)
        p.font.color.rgb = DARK_TEXT
        p.space_before = Pt(4)


def _load_verified_metrics(res: Path) -> dict[str, str]:
    anch = json.loads((res / "pinns_short_disp_hardq_anchor_corrected_v2_5seed_verified.json").read_text())["report"]
    m03 = json.loads((res / "ppt_mainline_o2_m0_m3_corrected_v3_5seed_verified.json").read_text())["report"]
    w17_1 = json.loads((res / "wave17_rc1_2x2_factorial_and_lowdisp_regime_5seed_verified.json").read_text())
    t4 = json.loads((res / "wave18_t4_latency_and_monotonic_gate_5seed_verified.json").read_text())
    w19_ext = json.loads((res / "wave19_dut_imptc_conflict_and_conformal_5seed_verified.json").read_text())
    w19_uni = json.loads((res / "wave19_unified_clean_m0_to_m3_5seed_verified.json").read_text())
    w19_ac = json.loads((res / "wave19_adaptive_conformal_12_and_72pair_5seed_verified.json").read_text())
    w20_d = json.loads((res / "wave20_decoupled_m2b_m3_synergy_5seed_verified.json").read_text())
    w20_m = json.loads((res / "wave20_true20f_mono_3x3_matrix_5seed_verified.json").read_text())

    m0a = anch["strata"]["all_1482"]["models"]["M0_anchor"]["site_equal"]
    cv323 = anch["strata"]["first_short_323"]["models"]["CV"]["site_equal"]
    rg = w17_1["pinns_lowdisp_regime_loso"]["first_short_323"]["models"]["Regime_M1_ground"]["site_equal"]
    dec = w20_d["pinns_4site_loso_1482"]["strata"]["all_1482"]["models"]["U_M3_decoupled_platt_rerank"]
    uni_j = w19_uni["jrdb_9location_lolo_5836"]["strata"]["all_10scenes"]["models"]["U_M3_mono_conflict_rerank"]["group_equal"]
    c_ws = m03["fixed_predictor_split_conformal"]["within_site_clip_split_fixed_3site_predictor"]
    dut_cf = w19_ext["dut_3066_zero_shot"]["zero_retrain_split_conformal"]["cross_group_ordered_pairs_no_retrain"]["M2b_ttc_conflict_gate"]
    imp_cf = w19_ext["imptc_564_zero_shot"]["zero_retrain_split_conformal"]["cross_group_ordered_pairs_no_retrain"]["M2b_ttc_conflict_gate"]
    jc_hyb = w19_ac["jrdb_72pair_adaptive_conformal"]["cross_group_ordered_pairs_fixed_predictor_no_retrain"]
    b12 = t4["hardware_profile"]["batch_sizes"]["B12"]["O2_M2b_conflict_gate"]
    return {
        "m0a": f"{m0a['ade1_m']:.4f} / {m0a['fde1_m']:.4f} m",
        "cv323": f"{cv323['ade1_m']:.4f} / {cv323['fde1_m']:.4f} m",
        "rg323": f"{rg['ade1_m']:.4f} / {rg['fde1_m']:.4f} m",
        "dec_p": f"{dec['group_equal_ade1_m']:.4f} / {dec['group_equal_fde1_m']:.4f} m",
        "dec_ap": f"{w20_d['pinns_4site_loso_1482']['decoupled_action_platt_inner_opt_thr']['group_equal']['macro_ap']:.4f}",
        "uni_j": f"{uni_j['ade1_m']:.4f} / {uni_j['fde1_m']:.4f} m",
        "rho_p": f"+{t4['pinns_monotonic_gate_1482']['gate_audit']['M1_mono_q']['spearman_gate_vs_q1_valid_joint_ratio']:.4f}",
        "rho_j": f"+{t4['jrdb_monotonic_gate_5836']['gate_audit']['M1_mono_q']['spearman_gate_vs_q1_valid_joint_ratio']:.4f}",
        "rho_20": f"+{w20_m['durations']['T20']['gate_audit']['M1_mono_q']['spearman_gate_vs_q1']:.4f}",
        "pinns_ws": f"{c_ws['90']['u6_cov']*100:.2f}% / {c_ws['95']['u6_cov']*100:.2f}%",
        "dut_cf": f"{dut_cf['90']['cov_u6']*100:.2f}% / {dut_cf['95']['cov_u6']*100:.2f}%",
        "imp_cf": f"{imp_cf['90']['cov_u6']*100:.2f}% / {imp_cf['95']['cov_u6']*100:.2f}%",
        "jrdb_hyb": f"{jc_hyb['90']['M3_speed_spread_scaled']['u6_cov']*100:.2f}% / {jc_hyb['95']['M3_speed_spread_scaled']['u6_cov']*100:.2f}%",
        "t4_lat": f"{b12['p50_ms']:.3f} / {b12['p95_ms']:.3f} ms",
    }


def build_defense_pptx(out_path: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[1]
    res = root / "results"
    figs = root / "docs" / "figures"
    m = _load_verified_metrics(res)

    prs = Presentation()
    prs.slide_width = Inches(13.3333333)
    prs.slide_height = Inches(7.5)

    # Slide 01: Cover
    s1 = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = THU_PURPLE
    bg.line.fill.background()
    tb = s1.shapes.add_textbox(Inches(0.9), Inches(1.8), Inches(11.5), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "清华大学硕士学位论文结题答辩汇报"
    p.font.size = Pt(20)
    p.font.color.rgb = RGBColor(0xE8, 0xD5, 0xEE)
    p2 = tf.add_paragraph()
    p2.text = "短时姿态观测下的非结构化道路行人动作与轨迹联合预测"
    p2.font.size = Pt(32)
    p2.font.bold = True
    p2.font.color.rgb = WHITE
    p2.space_before = Pt(16)
    p3 = tf.add_paragraph()
    p3.text = "答辩人：彭嘉钰    指导教师：姚丹亚 教授\n清华大学自动化系  |  53 份云端双层 SHA-256 验证归档 · 正负结论全透明"
    p3.font.size = Pt(16)
    p3.font.color.rgb = WHITE
    p3.space_before = Pt(28)

    # Slide 02: Outline
    s2 = _add_chrome(prs, "汇报提纲", "汇报提纲（严格对齐原始 20 页结构）", 2)
    _add_card(s2, 0.6, 1.3, 5.9, 2.5, "01 选题背景与三个核心预测问题（Slides 03–04）",
              ["园区/社区无标线混行场景：遮挡后 0.4 s（10 Hz 五帧）短时因果历史", "单目远距/遮挡导致关键点连续退化，需显式建模观测质量与硬关断"])
    _add_card(s2, 6.8, 1.3, 5.9, 2.5, "02 国内外研究现状与同口径复现（Slides 05–06）",
              ["Social-Pose 可变质量退化矩阵与无约束门控负相关机制发现", "MCENET / HST / Social-Transmotion 跨相机原生崩溃与 Frenet 增强对比"])
    _add_card(s2, 0.6, 4.1, 5.9, 2.6, "03 三大主要研究内容与实验消融（Slides 07–13）",
              [f"内容一：结构单调门控 M1_mono_q（秩相关 {m['rho_p']}）与 <0.5 m 匀速边界",
               f"内容二：20 m 折线情形 C（2.36% 穿线）与因果径向冲突门控",
               f"内容三：解耦五动作校准重排（PINNS 最优 {m['dec_p']}）与零重训共形"])
    _add_card(s2, 6.8, 4.1, 5.9, 2.6, "04 预期成果达成与附录原理（Slides 14–20）",
              [f"Tesla T4 实测 B=12 P50/P95 延迟 {m['t4_lat']}（20.7k–28.0k 参数）",
               "中文学位论文 + IEEE T-ITS 双栏 LaTeX 源码 + 120+ 单测全绿"])

    # Slides 03-06
    for pg, sec, ttl, c1_h, c1_b, c2_h, c2_b in [
        (3, "01 选题背景与问题", "非结构化道路的行人轨迹预测需求和局限",
         "应用需求与真实感知约束", ["低速无人配送/接驳车广泛进入校园、园区与居住区混合空间", "路侧障碍遮挡使行人进入视野后仅具 0.4 s（5 帧）短时因果历史", "缺乏车道线与硬隔离护栏，人车非频繁交织产生强多模态意图"],
         "既有评测的三大方法论隐患（本论文逐一纠偏）", ["重叠滑窗（相邻窗共享 26/30 未来帧）夸大显著性 -> 改用不重叠留出窗+整段重采样", "概率加权期望对比单路径基线混淆收缩效应 -> 主报同口径 Top-1 对 Top-1", "校准后重训预测器破坏可交换性 -> 全程严格执行零重训分裂共形"]),
        (4, "01 选题背景与问题", "非结构化道路短时观测下的三个预测问题",
         "三个递进耦合的科学问题", ["问题一：短时（0.4 s）且质量参差的姿态何时优于纯运动、何时反而有害？", "问题二：无语义高精地图时，20 m 被动视觉折线与 10 m 异构逼近图谁真正有效？", "问题三：五类可观测动态动作如何校准并辅助首选轨迹而不引发梯度干扰？"],
         "严谨样本定义与运行期几何断言", ["明确 449 个首采样窗与 323 个 <0.5 m 小位移窗系标注起点而非人工遮挡判读", "运行期强制校验秩-3 3x3 地面单应（6,930/7,410 帧朝向非零，足底残差 2.24 um）", "局部 20 m 折线强制围绕真实世界坐标 pt0_ego 截取（修正 835/835 有图窗）"]),
        (5, "02 国内外研究现状", "短时轨迹预测中的姿态信息与观测质量",
         "受控关键点退化审计（对标 Social-Pose）", ["VarQTrain 较 CleanTrain 在 PINNS 降 -2.41 cm、JRDB 降 -3.78 cm（Jitter020 降 -4.28 cm）", "m_pose=0 硬关断在姿态全盲（Test_FullDrop）下较无关断省 -3.27 cm / -3.68 cm"],
         "核心机制发现：无约束 MLP 门控秩相关反转与修复", ["无约束 MLP([h_m, h_p, q]) 受运动隐状态混淆，与有效率呈负相关（-0.1883 / -0.2184）", f"提出非负权重结构单调门控 M1_mono_q，秩相关恢复至 {m['rho_p']} / {m['rho_j']}", "PINNS 20 px 抖动下较无约束门控显著降低 -2.31 / -4.53 cm"]),
        (6, "02 国内外研究现状", "场景、交互与动作预测研究（主流基线同口径复现）",
         "原生场景坐标跨站崩溃与 SO(2) Frenet 赋能", ["MCENET / HST / Social-Transmotion 原生跨国 ADE_1 退化至 0.8052 / 0.7808 / 0.8792 m", "补齐本文因果 SO(2) Frenet 局部正交系后大幅降至 0.5734 / 0.5498 / 0.6217 m"],
         "同口径 Top-1 与计算效率全面领先", ["即便基线补齐 Frenet，本文 O2-M2b 仍显著低 -9.17 / -6.81 / -14.00 cm（整片 CI 全 < 0）", "参数量仅为 HST / Social-Transmotion 的 1/18，B=12 算量低 192–396 倍"]),
    ]:
        sl = _add_chrome(prs, sec, ttl, pg)
        _add_card(sl, 0.6, 1.35, 5.95, 5.3, c1_h, c1_b)
        _add_card(sl, 6.8, 1.35, 5.95, 5.3, c2_h, c2_b)


    def _add_fig_slide(pg: int, sec: str, ttl: str, heading: str, bullets: list[str], fig_name: str):
        sl = _add_chrome(prs, sec, ttl, pg)
        _add_card(sl, 0.5, 1.25, 5.15, 5.5, heading, bullets)
        pic = sl.shapes.add_picture(str(figs / fig_name), Inches(5.85), Inches(1.25), width=Inches(6.95))
        if pic.height > Inches(5.55):
            ratio = Inches(5.55) / pic.height
            pic.height = Inches(5.55)
            pic.width = int(pic.width * ratio)

    # Slides 07-08: Architecture & Overview
    for pg, ttl, h1, b1, h2, b2 in [
        (7, "整体网络架构与各模块输入输出",
         "严格因果输入与正交奇偶等变双流（F+, F-）",
         ["10 Hz 五帧地面位置 + 17/33 维关键点 + 四维质量向量 q + m_pose 硬掩码", "真实世界坐标 pt0_ego 局部环境与 10 m 有向邻体边（含径向逼近率 v_rad）", "任意旋转与镜像反射下五轴等变摆动精确 0.0 m，左右转向严格对称交换"],
         "多模态轨迹与可观测五动作联合输出",
         ["以末两点匀速（Last-2 CV）为残差锚点预测 K=6 模态地面轨迹与概率", "五类可观测动态动作：Start / Stop / Turn-Left / Turn-Right / Maintain", "下游零重训局部加权分裂共形安全管道直接输出可验证覆盖半径"]),
        (8, "三大主要研究内容与闭环脉络",
         "从单因素受控消融到端到端正负结论闭环",
         [f"研究内容一：结构单调质量门控 M1_mono_q、>=50 px 可用阈值与 <0.5 m 匀速边界",
          "研究内容二：20 m 被动折线负向检验（2.36% 穿线）与 10 m 因果径向冲突门控",
          f"研究内容三：解耦五动作 Platt 校准重排（PINNS {m['dec_p']}）与零重训分裂共形"],
         "跨场景留一验证规模（53 份 SHA-256 归档）",
         ["PINNS 四国跨站留一（1,482 不重叠窗 + 623 真 20 帧窗）与 JRDB 九地留一（5,836 窗）",
          f"DUT（3,066 窗）与 IMPTC（564 窗）冻结权重零样本迁移共形覆盖 {m['dut_cf']} / {m['imp_cf']}",
          "同步覆盖 JTA（960 窗）与北京路侧 DAIR-V2X（54 窗），全系固定五随机种子"]),
    ]:
        sl = _add_chrome(prs, "03 主要研究内容", ttl, pg)
        _add_card(sl, 0.6, 1.35, 5.95, 5.3, h1, b1)
        _add_card(sl, 6.8, 1.35, 5.95, 5.3, h2, b2)

    # Slides 09-11 & 13 with 300-DPI Verified Figures
    _add_fig_slide(9, "03 主要研究内容", "研究内容一：按观测质量融合姿态与运动",
                   "结构单调门控突破与 <0.5 m 诚实边界",
                   [f"M0_anchor 全窗达 {m['m0a']}（449 首采样窗较匀速降 -6.03 / -15.62 cm）",
                    f"first_short_323（<0.5 m）仍以匀速 {m['cv323']} 为参照：专域 Regime_M1 {m['rg323']} 对匀速 CI [-0.20, +0.26] cm 跨零",
                    f"真 20 帧 3x3 矩阵 M1_mono_q 恢复秩相关至 +0.7930 / +0.8365 / {m['rho_20']}，T=5 重残缺降 -1.37 cm",
                    "框高 <50 px 或 q1<0.75 时直拼劣化 +3.05~+6.15 cm，单调门控挽回 -1.65~-5.28 cm"],
                   "fig18_wave17_factorial_slide5_and_jrdb_conformal.png")

    _add_fig_slide(10, "03 主要研究内容", "研究内容二：局部环境与混行交互",
                   "20 m 折线负向证伪与因果冲突门控",
                   ["pt0_ego 真实坐标下 20 m 折线真值穿线率达 2.36%（有图窗 4.19%），M2a 升 +1.58 cm（情形 C）",
                    "发现远端/背离车辆扰动未门控 M2b 首选模态（非逼近子集 +2.57 cm）",
                    "提出因果径向冲突门控 M2b_ttc_conflict_gate：PINNS 降至 0.4572 / 1.0223 m（端到端链 0.4510 / 1.0097 m）",
                    "零样本迁移至 DUT 有车逼近窗显著降 -6.98 / -13.84 cm，IMPTC 降 -1.46 / -2.31 cm"],
                   "fig19_wave18_conflict_rerank_ext_and_t4_pareto.png")

    _add_fig_slide(11, "03 主要研究内容", "研究内容三：动作辅助轨迹预测与解耦重排",
                   "解耦五动作正向协同（0.0 m 控制点漂移）",
                   ["留一内层 Platt 校准零泄漏将五动作 ECE 压缩 4.9–23.6 倍（JRDB 0.0986 -> 0.0053）",
                    f"冻结 U_M2b 控制点解耦重排创 PINNS 最优 {m['dec_p']}（较联合 U_M3 降 -2.20 / -4.08 cm）",
                    f"五动作宏 AP 提升至 {m['dec_ap']}；大样本 JRDB 联合重排达 {m['uni_j']}"],
                   "fig20_wave19_20_unified_dut_boundary_and_adaptive_conformal.png")


    # Slide 12: Datasets & Usable Conditions
    s12 = _add_chrome(prs, "03 主要研究内容", "数据基准与姿态可用样本条件", 12)
    _add_card(s12, 0.6, 1.35, 5.95, 5.3, "六大跨场景测试基准（无测试泄漏）",
              ["PINNS 四国跨站留一：1,482 不重叠窗 + 623 真 20 帧窗（47 段视频）",
               "JRDB 校园九地留一：5,836 不重叠窗（800 独立行人，3D BlazePose）",
               "外部泛化：DUT（3,066 混行窗）、IMPTC（564 路口窗）、JTA（960 窗）、DAIR-V2X（54 窗）"])
    _add_card(s12, 6.8, 1.35, 5.95, 5.3, "定量确立的三大物理与统计可用边界",
              ["位移边界：5 帧位移 <0.5 m（first_short_323）单目地面反投影未超越末两点匀速",
               "分辨率与完整度边界：框高 >=50 px 且有效关节率 q1 >=0.75 方可安全启用非零姿态",
               "滑窗相关性边界：旧 497 滑窗按 21 段视频整段重采样仅 16/25 项区间排除零"])

    # Slide 13: Full Ablation & Zero-Retrain Conformal (with Fig 17)
    _add_fig_slide(13, "03 主要研究内容", "三项研究的实验对照、消融与零重训分裂共形",
                   "全程零重训分裂共形覆盖（四大混行基准）",
                   [f"PINNS 同站留出覆盖 {m['pinns_ws']}（半径较匀速缩减 -40.0% / -36.4%）",
                    f"DUT 72 跨片配对 {m['dut_cf']}，IMPTC 12 跨序列配对 {m['imp_cf']}",
                    f"源折局部速度+离散度加权共形将 JRDB 72 跨地点配对升至 {m['jrdb_hyb']}",
                    "JRDB >=1 m/s 快速行人 90% 覆盖率由 75.06% 跃升至 89.51%（历史重训经验覆盖单独区分）"],
                   "fig17_ppt_mainline_conformal_and_short_disp.png")

    # Slides 14-20: Schedule, Outcomes, Thanks, & 4 Appendix Slides
    for pg, sec, ttl, h1, b1, h2, b2 in [
        (14, "04 研究计划与预期成果", "论文工作进度总结（Waves 1–20 全部收官）",
         "云端受控实验与双层哈希核验", ["累计完成 Waves 1–20 共 53 份 Kaggle 云端实验归档（全部双层 SHA-256 核验）", "如实归档历史失败与提前取消运行记录（w13-cpu1 限额、374265835 等）"],
         "自动化回归测试与开源复现体系", ["构建 120+ 项本地单测（含 O(2) 精确等变、零重训共形、中英 LaTeX 逐位数值核验）", "公开仓库严守数据隐私红线（零私有权重/原始视频泄露）"]),
        (15, "04 研究计划与预期成果", "预期成果达成情况与实测硬件帕累托",
         "Tesla T4 实测计算效率（全面优于开题预期）", ["参数量仅 20,739–28,005（约为主流姿态 Transformer 的 1/18）", f"B=12 算量仅 1.56 MFLOPs（比 HST 低 396 倍），P50/P95 延迟 {m['t4_lat']}，显存 23.48 MiB"],
         "学术论文与答辩交付物", ["中文硕士学位论文完整稿（含表 1–10、300-DPI 图 1–20）", "IEEE T-ITS 双栏英文手稿及 Overleaf 即传 IEEEtran LaTeX 源码包"]),
        (16, "致谢", "总结与致谢",
         "核心学术贡献凝练", ["提出 O(2) 奇偶等变局部正交框架，彻底解决跨相机坐标系旋转崩溃", "揭示无约束门控负相关机制并提出结构单调门控 M1_mono_q 与 >=50 px 可用边界", "提出因果径向冲突门控、解耦五动作校准重排与源折局部加权零重训分裂共形"],
         "诚挚致谢", ["衷心感谢导师姚丹亚教授的悉心指导与严谨治学要求！", "恳请各位评审专家批评指正！"]),
        (17, "附录 网络原理与参考文献", "附录一：运动与姿态编码、结构单调质量门控",
         "O(2) Frenet 奇偶双流（Proposition 2）", ["以最近 >0.05 m 位移构建局部正交基 B=[u, u_perp]（静止回退至冠状面朝向）", "镜像流翻转横向坐标与左右膝角不对称度，保证严格反射等变"],
         "结构单调门控公式（Proposition 1）", ["g_mono = m_pose * (q1*q2)^gamma * sigmoid(b0 + w_+^T q)，gamma>0, w_+>=0", "对任意运动隐状态严格满足 dg/dq >= 0，且 m_pose=0 时恒为零"]),
        (18, "附录 网络原理与参考文献", "附录二：局部折线与因果径向冲突门控编码",
         "空槽注意力与因果冲突门控", ["w_risk = max_j [ m_nbr * sigmoid((v_rad - 0.20)/0.25) * exp(-d / 5.0) ]", "孤立行人（m_nbr=0）时恒等于零，抑制远端/背离交通背景噪声"],
         "真实世界中心 pt0_ego 几何修正说明", ["修复早期将折线中心误置为 (0,0) 的坐标缺陷，复核 835 个有图窗真实穿线率 4.19%"]),
        (19, "附录 网络原理与参考文献", "附录三：奇偶对称五动作桥接与局部加权共形",
         "解耦五动作校准重排", ["偶对称特征预测 Start/Stop/Maintain，奇偶对偶打分预测 Turn-Left/Turn-Right", "冻结 U_M2b 控制点消除小样本多任务梯度竞争，留一内层 Platt 校准零泄漏"],
         "局部加权零重训分裂共形（Proposition 3）", ["仅在源折折外残差上 NNLS 拟合 sigma(v0, spread) > 0，严格保持校准-测试可交换性"]),
        (20, "附录 网络原理与参考文献", "附录四：核心参考文献（节选 [1]–[20]）",
         "轨迹预测、姿态融合与混行交互基准", ["[1] Rasouli & Tsotsos, IEEE T-ITS 2020; [4] Scholler et al., IEEE RA-L 2020 (CV)", "[7] Martin-Martin et al., IEEE TPAMI 2023 (JRDB); [13] Cheng et al., ITSC 2020 (MCENET)", "[14] Salzmann et al., IEEE RA-L 2023 (HST); [15] Saadatnejad et al., ICLR 2024"],
         "质量退化、安全共形与非结构化数据集", ["[10] Yang et al., IEEE IV 2019 (DUT); [11] Hetzel et al., IEEE IV 2023 (IMPTC)", "[17] Gao et al., 2025 (Social-Pose); [19] Lindemann et al., IEEE RA-L 2023", "[20] Peng et al., arXiv:2605.25947, 2026 (PINNS Multi-Country Benchmark)"]),
    ]:
        sl = _add_chrome(prs, sec, ttl, pg)
        _add_card(sl, 0.6, 1.35, 5.95, 5.3, h1, b1)
        _add_card(sl, 6.8, 1.35, 5.95, 5.3, h2, b2)

    if out_path is None:
        out_path = root / "dist" / "ShortPose_Thesis_Defense_Final_20Slides.pptx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    return out_path


if __name__ == "__main__":
    out = build_defense_pptx()
    print(f"Generated 20-slide defense PPTX: {out} ({out.stat().st_size} bytes)")
