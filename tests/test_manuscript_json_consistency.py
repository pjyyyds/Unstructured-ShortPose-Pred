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

        for fn, txt in (("zh", zh), ("en", en)):
            p = root / ("paper/ShortPose_Unstructured_Pedestrian_Paper.md" if fn == "zh" else "paper/IEEE_TITS_ShortPose_Manuscript.md")
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


if __name__ == "__main__":
    unittest.main()
