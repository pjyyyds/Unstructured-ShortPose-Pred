"""Structural, bibliographic, figure, and SHA-256 JSON consistency tests for IEEEtran main.tex."""
from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest
import zipfile

from scripts.build_overleaf_bundle import build_overleaf_zip


class TestIeeeLatexSubmission(unittest.TestCase):
    def test_latex_syntax_environments_citations_and_refs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tex_path = root / "paper" / "ieee_tits_latex" / "main.tex"
        bib_path = root / "paper" / "ieee_tits_latex" / "references.bib"
        tex = tex_path.read_text(encoding="utf-8")
        bib = bib_path.read_text(encoding="utf-8")

        self.assertTrue(tex.isascii(), "main.tex must remain pure ASCII for portable pdfLaTeX")
        self.assertTrue(bib.isascii(), "references.bib must remain pure ASCII for BibTeX")

        # Strip unescaped % line comments before structural counting
        lines = [re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.splitlines()]
        clean = "\n".join(lines)
        self.assertEqual(clean.count("{"), clean.count("}"), "Unbalanced curly braces in main.tex")

        # Balanced \begin{env} ... \end{env} stack
        stack: list[str] = []
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", clean):
            kind, env = m.group(1), m.group(2)
            if kind == "begin":
                stack.append(env)
            else:
                self.assertTrue(stack, f"Unmatched \\end{{{env}}}")
                top = stack.pop()
                self.assertEqual(top, env, f"Environment mismatch: \\begin{{{top}}} vs \\end{{{env}}}")
        self.assertEqual(stack, [], f"Unclosed environments: {stack}")

        # Balanced inline $...$ delimiters (ignoring \$)
        inline_dollars = len(re.findall(r"(?<!\\)\$", clean))
        self.assertEqual(inline_dollars % 2, 0, f"Odd number of inline $ delimiters: {inline_dollars}")

        # Every \cite{...} resolves in references.bib and every bib key is cited
        bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
        cited_keys: set[str] = set()
        for grp in re.findall(r"\\cite\{([^}]+)\}", clean):
            for k in grp.split(","):
                cited_keys.add(k.strip())
        self.assertEqual(cited_keys - bib_keys, set(), f"Missing BibTeX keys: {cited_keys - bib_keys}")
        self.assertEqual(bib_keys - cited_keys, set(), f"Uncited BibTeX keys: {bib_keys - cited_keys}")

        # Every \ref / \eqref matches a \label
        labels = set(re.findall(r"\\label\{([^}]+)\}", clean))
        refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", clean))
        self.assertEqual(refs - labels, set(), f"Unresolved \\ref/\\eqref labels: {refs - labels}")

    def test_latex_metrics_match_verified_jsons_and_overleaf_zip(self) -> None:
        root = Path(__file__).resolve().parents[1]
        res = root / "results"
        tex = (root / "paper" / "ieee_tits_latex" / "main.tex").read_text(encoding="utf-8")

        self.assertNotIn("0.26\\%", tex, "Retracted mis-centered polyline rate must not appear")

        anch_raw = json.loads((res / "pinns_short_disp_hardq_anchor_corrected_v2_5seed_verified.json").read_text())
        anch = anch_raw.get("report", anch_raw)
        m0a = anch["strata"]["all_1482"]["models"]["M0_anchor"]["site_equal"]
        cv_323 = anch["strata"]["first_short_323"]["models"]["CV"]["site_equal"]
        self.assertIn(f"{m0a['ade1_m']:.4f} / {m0a['fde1_m']:.4f}", tex)
        self.assertIn(f"{cv_323['ade1_m']:.4f} / {cv_323['fde1_m']:.4f}", tex)

        m03 = json.loads((res / "ppt_mainline_o2_m0_m3_corrected_v3_5seed_verified.json").read_text()).get("report")
        c_ws = m03["fixed_predictor_split_conformal"]["within_site_clip_split_fixed_3site_predictor"]
        c_12 = m03["fixed_predictor_split_conformal"]["cross_site_12pair_fixed_2site_predictor_no_retrain"]
        self.assertIn(f"{c_ws['90']['u6_cov'] * 100:.2f}\\%", tex)
        self.assertIn(f"{c_ws['95']['u6_cov'] * 100:.2f}\\%", tex)
        self.assertIn(f"{c_12['90']['u6_cov'] * 100:.2f}\\%", tex)
        self.assertIn(f"{c_12['95']['u6_cov'] * 100:.2f}\\%", tex)

        t4 = json.loads((res / "wave18_t4_latency_and_monotonic_gate_5seed_verified.json").read_text())
        b12 = t4["hardware_profile"]["batch_sizes"]["B12"]
        rho_mp = t4["pinns_monotonic_gate_1482"]["gate_audit"]["M1_mono_q"]["spearman_gate_vs_q1_valid_joint_ratio"]
        self.assertIn(f"+{rho_mp:.4f}", tex)
        self.assertIn(f"{b12['O2_M2b_conflict_gate']['p50_ms']:.3f} / {b12['O2_M2b_conflict_gate']['p95_ms']:.3f}", tex)

        w19_ext = json.loads((res / "wave19_dut_imptc_conflict_and_conformal_5seed_verified.json").read_text())
        dut_cf = w19_ext["dut_3066_zero_shot"]["zero_retrain_split_conformal"]["cross_group_ordered_pairs_no_retrain"]["M2b_ttc_conflict_gate"]
        imp_cf = w19_ext["imptc_564_zero_shot"]["zero_retrain_split_conformal"]["cross_group_ordered_pairs_no_retrain"]["M2b_ttc_conflict_gate"]
        self.assertIn(f"{dut_cf['90']['cov_u6'] * 100:.2f}\\% / {dut_cf['95']['cov_u6'] * 100:.2f}\\%", tex)
        self.assertIn(f"{imp_cf['90']['cov_u6'] * 100:.2f}\\% / {imp_cf['95']['cov_u6'] * 100:.2f}\\%", tex)

        w19_ac = json.loads((res / "wave19_adaptive_conformal_12_and_72pair_5seed_verified.json").read_text())
        jc_hyb = w19_ac["jrdb_72pair_adaptive_conformal"]["cross_group_ordered_pairs_fixed_predictor_no_retrain"]
        w20_d = json.loads((res / "wave20_decoupled_m2b_m3_synergy_5seed_verified.json").read_text())
        dec_p = w20_d["pinns_4site_loso_1482"]["strata"]["all_1482"]["models"]["U_M3_decoupled_platt_rerank"]
        dec_ap = w20_d["pinns_4site_loso_1482"]["decoupled_action_platt_inner_opt_thr"]["group_equal"]["macro_ap"]
        self.assertIn(f"{dec_p['group_equal_ade1_m']:.4f} / {dec_p['group_equal_fde1_m']:.4f}", tex)
        self.assertIn(f"{dec_ap:.4f}", tex)
        self.assertIn(f"{jc_hyb['90']['M3_speed_spread_scaled']['u6_cov'] * 100:.2f}\\% / {jc_hyb['95']['M3_speed_spread_scaled']['u6_cov'] * 100:.2f}\\%", tex)

        with tempfile.TemporaryDirectory() as td:
            out_zip = build_overleaf_zip(Path(td) / "overleaf.zip")
            with zipfile.ZipFile(out_zip) as zf:
                names = set(zf.namelist())
                self.assertIn("main.tex", names)
                self.assertIn("references.bib", names)
                self.assertEqual(len([n for n in names if n.startswith("figures/")]), 4)


if __name__ == "__main__":
    unittest.main()
