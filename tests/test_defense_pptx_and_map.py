"""Automated layout, figure, link, and JSON-metric verification for the 20-slide defense PPTX."""
from __future__ import annotations

from pathlib import Path
import re
import tempfile
import unittest

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from scripts.build_defense_final_pptx import build_defense_pptx


class TestDefensePptxAndReviewLinks(unittest.TestCase):
    def test_defense_pptx_20_slides_bounds_figures_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = build_defense_pptx(Path(td) / "defense_20.pptx")
            prs = Presentation(str(out))
            self.assertEqual(len(prs.slides), 20)
            self.assertAlmostEqual(prs.slide_width.inches, 13.3333333, places=3)
            self.assertAlmostEqual(prs.slide_height.inches, 7.5, places=3)

            pic_slides: list[int] = []
            full_text_chunks: list[str] = []
            for idx, sl in enumerate(prs.slides, 1):
                for sh in sl.shapes:
                    self.assertGreaterEqual(sh.left.inches, -0.01, f"Slide {idx} left overflow")
                    self.assertGreaterEqual(sh.top.inches, -0.01, f"Slide {idx} top overflow")
                    self.assertLessEqual(sh.left.inches + sh.width.inches, 13.35, f"Slide {idx} right overflow")
                    self.assertLessEqual(sh.top.inches + sh.height.inches, 7.51, f"Slide {idx} bottom overflow")
                    if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                        pic_slides.append(idx)
                        self.assertGreaterEqual(sh.top.inches, 1.20)
                        self.assertLessEqual(sh.top.inches + sh.height.inches, 6.90)
                    if sh.has_text_frame:
                        full_text_chunks.append(sh.text_frame.text)

            self.assertEqual(pic_slides, [9, 10, 11, 13])
            joined = "\n".join(full_text_chunks)
            for expected in (
                "0.4505 / 1.0085 m",
                "0.3545 / 0.8491 m",
                "+0.7301",
                "89.76% / 95.50%",
                "88.89% / 94.20%",
                "4.782 / 5.231 ms",
            ):
                self.assertIn(expected, joined)
            self.assertNotIn("0.26%", joined)

    def test_project_review_summary_links_and_waves_19_20(self) -> None:
        root = Path(__file__).resolve().parents[1]
        rev = root / "reviews" / "PROJECT_REVIEW_SUMMARY.md"
        txt = rev.read_text(encoding="utf-8")
        links = re.findall(r"\]\((?!https?://|#|file://)([^)]+)\)", txt)
        missing = [lk for lk in links if not (rev.parent / lk.split("#")[0]).exists()]
        self.assertEqual(missing, [], f"Broken links in PROJECT_REVIEW_SUMMARY.md: {missing}")
        self.assertIn("0.4505 / 1.0085 m", txt)
        self.assertIn("88.89% / 94.20%", txt)


if __name__ == "__main__":
    unittest.main()
