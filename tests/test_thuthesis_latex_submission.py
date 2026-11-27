"""Structural, bibliographic, figure, and SHA-256 JSON consistency tests for Chinese XeLaTeX thesis."""
from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest
import zipfile

from scripts.build_overleaf_bundle import build_thuthesis_overleaf_zip


class TestThuThesisLatexSubmission(unittest.TestCase):
    def test_xelatex_syntax_environments_citations_and_metrics(self) -> None:
        root = Path(__file__).resolve().parents[1]
        res = root / "results"
        tex_dir = root / "paper" / "thuthesis_latex"
        tex = (tex_dir / "main.tex").read_text(encoding="utf-8")
        bib = (tex_dir / "references.bib").read_text(encoding="utf-8")

        lines = [re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.splitlines()]
        clean = "\n".join(lines)
        self.assertEqual(clean.count("{"), clean.count("}"), "Unbalanced curly braces in thuthesis main.tex")

        stack: list[str] = []
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", clean):
            kind, env = m.group(1), m.group(2)
            if kind == "begin":
                stack.append(env)
            else:
                self.assertTrue(stack, f"Unmatched \\end{{{env}}}")
                self.assertEqual(stack.pop(), env)
        self.assertEqual(stack, [], f"Unclosed environments: {stack}")

        inline_dollars = len(re.findall(r"(?<!\\)\$", clean))
        self.assertEqual(inline_dollars % 2, 0, f"Odd number of inline $ delimiters: {inline_dollars}")

        bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
        cited_keys = {k.strip() for grp in re.findall(r"\\cite\{([^}]+)\}", clean) for k in grp.split(",")}
        self.assertEqual(cited_keys, bib_keys, f"Citation mismatch: {cited_keys ^ bib_keys}")

        labels = set(re.findall(r"\\label\{([^}]+)\}", clean))
        refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", clean))
        self.assertEqual(refs - labels, set(), f"Unresolved labels: {refs - labels}")
        self.assertNotIn("0.26\\%", tex)

        w19_ac = json.loads((res / "wave19_adaptive_conformal_12_and_72pair_5seed_verified.json").read_text())
        jc_hyb = w19_ac["jrdb_72pair_adaptive_conformal"]["cross_group_ordered_pairs_fixed_predictor_no_retrain"]
        w20_d = json.loads((res / "wave20_decoupled_m2b_m3_synergy_5seed_verified.json").read_text())
        dec_p = w20_d["pinns_4site_loso_1482"]["strata"]["all_1482"]["models"]["U_M3_decoupled_platt_rerank"]
        self.assertIn(f"{dec_p['group_equal_ade1_m']:.4f} / {dec_p['group_equal_fde1_m']:.4f}", tex)
        self.assertIn(f"{jc_hyb['90']['M3_speed_spread_scaled']['u6_cov']*100:.2f}\\% / {jc_hyb['95']['M3_speed_spread_scaled']['u6_cov']*100:.2f}\\%", tex)

        with tempfile.TemporaryDirectory() as td:
            out_zip = build_thuthesis_overleaf_zip(Path(td) / "thu_overleaf.zip")
            with zipfile.ZipFile(out_zip) as zf:
                names = set(zf.namelist())
                self.assertIn("main.tex", names)
                self.assertIn("references.bib", names)
                self.assertEqual(len([n for n in names if n.startswith("figures/")]), 4)


if __name__ == "__main__":
    unittest.main()
