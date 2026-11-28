"""Strict cross-file verification of all 4 external audit remediations (Waves 21-24)."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unittest


class TestWave21To24ExactAuditReconciliation(unittest.TestCase):
    def test_blind_raw_video_labels_and_json_hashes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        lbl_path = root / "data/annotations/pinns_blind_raw_video_emergence_labels.json"
        labels = json.loads(lbl_path.read_text(encoding="utf-8"))
        self.assertEqual(len(labels), 241)
        cnt = Counter(labels.values())
        self.assertEqual(cnt["veh"], 32)
        self.assertEqual(cnt["static"], 63)
        self.assertEqual(cnt["pers"], 45)
        self.assertEqual(cnt["veh"] + cnt["static"] + cnt["pers"], 140)
        self.assertEqual(cnt["border"], 27)
        self.assertEqual(cnt["open"], 74)

        exp_path = root / "data/annotations/pinns_explicit_241_visual_audit.json"
        exp_doc = json.loads(exp_path.read_text(encoding="utf-8"))
        self.assertEqual(len(exp_doc["labels"]), 241)
        ecnt = Counter(exp_doc["labels"].values())
        self.assertEqual(ecnt["veh"], 32)
        self.assertEqual(ecnt["static"], 66)
        self.assertEqual(ecnt["pers"], 20)
        self.assertEqual(ecnt["veh"] + ecnt["static"] + ecnt["pers"], 118)

        for jf in (
            "wave23_pinns_video_audit_and_6slot_ranker_5seed_verified.json",
            "wave23_jaad_autopose_and_isofpr_leadtime_5seed_verified.json",
            "wave24_pinns_raw_video_annotated_emergence_5seed_verified.json",
            "wave25_pinns_explicit_audit_and_emergence_safe_5seed_verified.json",
            "wave25_jaad_reappearance_per_event_audit_5seed_verified.json",
        ):
            payload = json.loads((root / "results" / jf).read_text(encoding="utf-8"))
            self.assertTrue(payload["verification"]["dual_sha256_verified"], jf)

    def test_zero_stale_claims_and_exact_reconciled_numbers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        zh = (root / "paper/ShortPose_Unstructured_Pedestrian_Paper.md").read_text(encoding="utf-8")
        en = (root / "paper/IEEE_TITS_ShortPose_Manuscript.md").read_text(encoding="utf-8")
        ie = (root / "paper/ieee_tits_latex/main.tex").read_text(encoding="utf-8")
        th = (root / "paper/thuthesis_latex/main.tex").read_text(encoding="utf-8")
        mp = (root / "paper/PPT_20Slides_Defense_Evidence_Map.md").read_text(encoding="utf-8")
        rv = (root / "reviews/PROJECT_REVIEW_SUMMARY.md").read_text(encoding="utf-8")

        banned_everywhere = (
            "0.6486",
            "0.6476",
            "0.4832 / 1.0764",
            "0.4835 / 1.0765",
            "0.2764",
            "235 true negatives",
            "235 真负例",
            "Test Pure 3D Pose",
            "测试纯 3D 姿态",
            "JTA-228` Urban (`3D Pose",
            "JTA-228`（城市序列，3D 关节",
            "134/134",
            "2,175",
            "2{,}175",
            "双盲确认",
            "双盲判据",
            "Dual-Confirmed",
            "dual-confirmed",
        )
        for name, doc in (("zh_md", zh), ("en_md", en), ("ieee_tex", ie), ("thu_tex", th), ("ppt_map", mp)):
            for bad in banned_everywhere:
                self.assertNotIn(bad, doc, f"Stale string {bad!r} still in {name}")
        for bad in ("0.2764", "235 true negatives", "235 真负例", "134/134"):
            self.assertNotIn(bad, rv, f"Stale string {bad!r} still in review_md")

        for name, doc in (("ieee_tex", ie), ("thu_tex", th), ("ppt_map", mp)):
            self.assertNotIn("873.1", doc, f"Conflated +873.1 ms still in {name}")
            self.assertNotIn("557.9", doc, f"Conflated +557.9 ms still in {name}")

        for name, doc in (("zh_md", zh), ("en_md", en), ("ieee_tex", ie), ("thu_tex", th)):
            for req in ("0.4382", "0.4364", "0.5249", "0.2803", "0.4888", "0.4910", "431", "0.125", "0.3919", "0.3892", "0.5602", "0.5564", "250.0", "500.0", "168", "118"):
                self.assertIn(req, doc, f"Missing reconciled {req!r} in {name}")


if __name__ == "__main__":
    unittest.main()
