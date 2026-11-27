"""Build self-contained Overleaf upload zips for IEEE T-ITS (pdfLaTeX) and Chinese Thesis (XeLaTeX)."""
from __future__ import annotations

from pathlib import Path
import re
import zipfile


def _package_tex_dir(tex_dir: Path, fig_dir: Path, out_zip: Path) -> Path:
    main_tex = tex_dir / "main.tex"
    bib_file = tex_dir / "references.bib"
    readme_file = tex_dir / "README_OVERLEAF.md"

    tex_src = main_tex.read_text(encoding="utf-8")
    figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_src)))
    if not figs:
        raise RuntimeError(f"No \\includegraphics targets found in {main_tex}")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(main_tex, arcname="main.tex")
        zf.write(bib_file, arcname="references.bib")
        zf.write(readme_file, arcname="README_OVERLEAF.md")
        for fg in figs:
            fg_path = fig_dir / fg
            if not fg_path.exists():
                raise FileNotFoundError(f"Missing figure referenced by {main_tex}: {fg_path}")
            zf.write(fg_path, arcname=f"figures/{fg}")
    return out_zip


def build_overleaf_zip(out_zip: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[1]
    if out_zip is None:
        out_zip = root / "dist" / "overleaf_ieee_tits_submission.zip"
    return _package_tex_dir(root / "paper" / "ieee_tits_latex", root / "docs" / "figures", out_zip)


def build_thuthesis_overleaf_zip(out_zip: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[1]
    if out_zip is None:
        out_zip = root / "dist" / "overleaf_thuthesis_chinese.zip"
    return _package_tex_dir(root / "paper" / "thuthesis_latex", root / "docs" / "figures", out_zip)


if __name__ == "__main__":
    ieee_zip = build_overleaf_zip()
    thu_zip = build_thuthesis_overleaf_zip()
    print(f"Built IEEE T-ITS bundle: {ieee_zip} ({ieee_zip.stat().st_size} bytes)")
    print(f"Built Chinese Thesis XeLaTeX bundle: {thu_zip} ({thu_zip.stat().st_size} bytes)")
