"""Automated verification that both Chinese and IEEE English manuscripts match SHA-256 verified JSONs."""
from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


class TestManuscriptJsonConsistency(unittest.TestCase):
    def test_both_manuscripts_match_verified_jsons_and_links(self) -> None:
        root = Path(__file__).resolve().parents[1]
        res = root / "results"
        zh = (root / "paper/ShortPose_Unstructured_Pedestrian_Paper.md").read_text(encoding="utf-8")
        en = (root / "paper/IEEE_TITS_ShortPose_Manuscript.md").read_text(encoding="utf-8")
        ppt_map = (root / "paper/PPT_20Slides_Defense_Evidence_Map.md").read_text(encoding="utf-8")

        for fn, txt, rel in (
            ("zh", zh, "paper/ShortPose_Unstructured_Pedestrian_Paper.md"),
            ("en", en, "paper/IEEE_TITS_ShortPose_Manuscript.md"),
            ("ppt_map", ppt_map, "paper/PPT_20Slides_Defense_Evidence_Map.md"),
        ):
            p = root / rel
            links = re.findall(r"\]\((?!https?://|#|file://)([^)]+)\)", txt)
            missing = [lk for lk in links if not (p.parent / lk.split("#")[0]).exists()]
            self.assertEqual(missing, [], f"Broken links in {fn}: {missing}")
            self.assertNotIn("0.26%", txt, f"Retracted mis-centered polyline rate found in {fn}")

        anch_raw = json.loads((res / "pinns_short_disp_hardq_anchor_corrected_v2_5seed_verified.json").read_text())
        anch = anch_raw.get("report", anch_raw)
        m0a_all = anch["strata"]["all_1482"]["models"]["M0_anchor"]["site_equal"]
        cv_323 = anch["strata"]["first_short_323"]["models"]["CV"]["site_equal"]
        for txt in (zh, en):
            self.assertIn(f"{m0a_all['ade1_m']:.4f} / {m0a_all['fde1_m']:.4f}", txt)
            self.assertIn(f"{cv_323['ade1_m']:.4f} / {cv_323['fde1_m']:.4f}", txt)

        m03_raw = json.loads((res / "ppt_mainline_o2_m0_m3_corrected_v3_5seed_verified.json").read_text())
        m03 = m03_raw.get("report", m03_raw)
        c_ws = m03["fixed_predictor_split_conformal"]["within_site_clip_split_fixed_3site_predictor"]
        c_12 = m03["fixed_predictor_split_conformal"]["cross_site_12pair_fixed_2site_predictor_no_retrain"]
        for txt in (zh, en):
            self.assertIn(f"{c_ws['90']['u6_cov'] * 100:.2f}%", txt)
            self.assertIn(f"{c_ws['95']['u6_cov'] * 100:.2f}%", txt)
            self.assertIn(f"{c_12['90']['u6_cov'] * 100:.2f}%", txt)
            self.assertIn(f"{c_12['95']['u6_cov'] * 100:.2f}%", txt)

        w17_1 = json.loads((res / "wave17_rc1_2x2_factorial_and_lowdisp_regime_5seed_verified.json").read_text())
        rg_g = w17_1["pinns_lowdisp_regime_loso"]["first_short_323"]["models"]["Regime_M1_ground"]["site_equal"]
        w17_3 = json.loads((res / "wave17_rc3_clean_chain_and_jrdb_conformal_5seed_verified.json").read_text())
        jc_ws = w17_3["jrdb_fixed_predictor_split_conformal"]["within_scene_temporal_split_9loc_lolo"]["M2b"]
        jc_72 = w17_3["jrdb_fixed_predictor_split_conformal"]["cross_location_72pair_fixed_7loc_predictor_no_retrain"]
        s5 = json.loads((res / "wave17_slide5_pose_noise_and_gate_response_5seed_verified.json").read_text())
        rho_p = abs(s5["pinns_4site_loso_1482"]["gate_response_audit"]["VarQTrain"]["natural_strata_M1"]["spearman_gate_vs_q1_valid_joint_ratio"])
        rho_j = abs(s5["jrdb_9location_lolo_5836"]["gate_response_audit"]["VarQTrain"]["natural_strata_M1"]["spearman_gate_vs_q1_valid_joint_ratio"])
        for txt in (zh, en):
            self.assertIn(f"{rg_g['ade1_m']:.4f} / {rg_g['fde1_m']:.4f}", txt)
            self.assertIn(f"{jc_ws['90']['cov_u6'] * 100:.2f}%", txt)
            self.assertIn(f"{jc_ws['95']['cov_u6'] * 100:.2f}%", txt)
            self.assertIn(f"{jc_72['90']['cov_u6'] * 100:.2f}%", txt)
            self.assertIn(f"{jc_72['95']['cov_u6'] * 100:.2f}%", txt)
            self.assertIn(f"{rho_p:.4f}", txt)
            self.assertIn(f"{rho_j:.4f}", txt)

        w18_cg = json.loads((res / "wave18_conflict_gated_interaction_5seed_verified.json").read_text())
        cg_p = w18_cg["pinns_4site_loso_1482"]["strata"]["all"]["models"]["M2b_ttc_conflict_gate"]["group_equal"]
        w18_ac = json.loads((res / "wave18_action_calibration_and_rerank_5seed_verified.json").read_text())
        ece_p = w18_ac["pinns_4site_loso_1482"]["action_metrics_platt_inner_opt_thr"]["group_equal"]["mean_ece"]
        f1_p = w18_ac["pinns_4site_loso_1482"]["action_metrics_platt_inner_opt_thr"]["group_equal"]["macro_f1"]
        for txt in (zh, en):
            self.assertIn(f"{cg_p['ade1_m']:.4f} / {cg_p['fde1_m']:.4f}", txt)
            self.assertIn(f"{ece_p:.4f}", txt)
            self.assertIn(f"{f1_p:.4f}", txt)

        ext = json.loads((res / "wave17_ppt_ext_baselines_loso_5seed_verified.json").read_text())
        pm = ext["pinns_4site_loso_1482"]["strata"]["all_1482"]
        jm = ext["jrdb_9location_lolo_5836"]["strata"]["all_10scenes"]
        t4 = json.loads((res / "wave18_t4_latency_and_monotonic_gate_5seed_verified.json").read_text())
        b12 = t4["hardware_profile"]["batch_sizes"]["B12"]
        rho_mp = t4["pinns_monotonic_gate_1482"]["gate_audit"]["M1_mono_q"]["spearman_gate_vs_q1_valid_joint_ratio"]
        rho_mj = t4["jrdb_monotonic_gate_5836"]["gate_audit"]["M1_mono_q"]["spearman_gate_vs_q1_valid_joint_ratio"]
        mono_p = t4["pinns_monotonic_gate_1482"]["test_conditions"]["Test_Natural"]["M1_mono_q"]
        for txt in (zh, en):
            for k in ("Ext1_MCENET", "Ext1_MCENET_Frenet", "Ext2_HST", "Ext2_HST_Frenet", "Ext3_SocialTransmotion", "Ext3_SocialTransmotion_Frenet"):
                self.assertIn(f"{pm[k]['group_equal']['ade1_m']:.4f} / {pm[k]['group_equal']['fde1_m']:.4f}", txt)
                self.assertIn(f"{jm[k]['group_equal']['ade1_m']:.4f} / {jm[k]['group_equal']['fde1_m']:.4f}", txt)
            self.assertIn(f"+{rho_mp:.4f}", txt)
            self.assertIn(f"+{rho_mj:.4f}", txt)
            self.assertIn(f"{mono_p['group_equal_ade1_m']:.4f} / {mono_p['group_equal_fde1_m']:.4f}", txt)
            self.assertIn(f"{b12['O2_M2b_conflict_gate']['p50_ms']:.3f} / {b12['O2_M2b_conflict_gate']['p95_ms']:.3f}", txt)
            self.assertIn("fig19_wave18_conflict_rerank_ext_and_t4_pareto.png", txt)

        w19_ext = json.loads((res / "wave19_dut_imptc_conflict_and_conformal_5seed_verified.json").read_text())
        dut_cg = w19_ext["dut_3066_zero_shot"]["strata"]["all"]["models"]["M2b_ttc_conflict_gate"]["group_equal"]
        dut_cf = w19_ext["dut_3066_zero_shot"]["zero_retrain_split_conformal"]["cross_group_ordered_pairs_no_retrain"]["M2b_ttc_conflict_gate"]
        imp_cg = w19_ext["imptc_564_zero_shot"]["strata"]["all"]["models"]["M2b_ttc_conflict_gate"]["group_equal"]
        imp_cf = w19_ext["imptc_564_zero_shot"]["zero_retrain_split_conformal"]["cross_group_ordered_pairs_no_retrain"]["M2b_ttc_conflict_gate"]
        w19_uni = json.loads((res / "wave19_unified_clean_m0_to_m3_5seed_verified.json").read_text())
        uni_p = w19_uni["pinns_4site_loso_1482"]["strata"]["all_1482"]["models"]["U_M2b_mono_conflict"]["group_equal"]
        uni_j = w19_uni["jrdb_9location_lolo_5836"]["strata"]["all_10scenes"]["models"]["U_M3_mono_conflict_rerank"]["group_equal"]
        for txt in (zh, en):
            self.assertIn(f"{uni_p['ade1_m']:.4f} / {uni_p['fde1_m']:.4f}", txt)
            self.assertIn(f"{uni_j['ade1_m']:.4f} / {uni_j['fde1_m']:.4f}", txt)
            self.assertIn(f"{dut_cg['ade1_m']:.4f} / {dut_cg['fde1_m']:.4f}", txt)
            self.assertIn(f"{imp_cg['ade1_m']:.4f} / {imp_cg['fde1_m']:.4f}", txt)
            self.assertIn(f"{dut_cf['90']['cov_u6'] * 100:.2f}% / {dut_cf['95']['cov_u6'] * 100:.2f}%", txt)
            self.assertIn(f"{imp_cf['90']['cov_u6'] * 100:.2f}% / {imp_cf['95']['cov_u6'] * 100:.2f}%", txt)


if __name__ == "__main__":
    unittest.main()
