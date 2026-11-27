"""Build a self-contained Overleaf upload zip for the IEEE T-ITS LaTeX submission."""
from __future__ import annotations

from pathlib import Path
import re
import zipfile


def build_overleaf_zip(out_zip: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[1]
    tex_dir = root / "paper" / "ieee_tits_latex"
    fig_dir = root / "docs" / "figures"
    main_tex = tex_dir / "main.tex"
    bib_file = tex_dir / "references.bib"
    readme_file = tex_dir / "README_OVERLEAF.md"

    tex_src = main_tex.read_text(encoding="utf-8")
    figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_src)))
    if not figs:
        raise RuntimeError("No \\includegraphics targets found in main.tex")

    if out_zip is None:
        out_zip = root / "dist" / "overleaf_ieee_tits_submission.zip"
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(main_tex, arcname="main.tex")
        zf.write(bib_file, arcname="references.bib")
        zf.write(readme_file, arcname="README_OVERLEAF.md")
        for fg in figs:
            fg_path = fig_dir / fg
            if not fg_path.exists():
                raise FileNotFoundError(f"Missing figure referenced by main.tex: {fg_path}")
            zf.write(fg_path, arcname=f"figures/{fg}")
    return out_zip


if __name__ == "__main__":
    dest = build_overleaf_zip()
    print(f"Built Overleaf IEEE T-ITS bundle: {dest} ({dest.stat().st_size} bytes)")
