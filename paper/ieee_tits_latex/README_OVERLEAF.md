# IEEE T-ITS LaTeX Submission Package (Overleaf-Ready)

This directory contains the double-column `IEEEtran` journal submission source for:

> **Orthogonal-Equivariant Pedestrian Action and Trajectory Forecasting on Unstructured Roads Under Short-Term Variable-Quality Pose Observations**  
> Jiayu Peng and Danya Yao (*Member, IEEE*), Department of Automation, Tsinghua University

## Files
- `main.tex`: Complete `\documentclass[journal]{IEEEtran}` manuscript (Sections I--VI, Propositions 1--3, Tables I--VI, Figures 1--4).
- `references.bib`: Consecutive BibTeX database (`21` entries) formatted for `IEEEtran.bst`.

## Compiling Locally or on Overleaf
1. **Local TeX Live / MacTeX**:
   ```bash
   cd paper/ieee_tits_latex
   pdflatex main && bibtex main && pdflatex main && pdflatex main
   ```
   (`main.tex` sets `\graphicspath{{../../docs/figures/}{./figures/}}`, resolving repository figures directly.)
2. **One-Click Overleaf Upload Archive**:
   ```bash
   python3 scripts/build_overleaf_bundle.py
   ```
   Upload the generated `dist/overleaf_ieee_tits_submission.zip` directly via **Overleaf -> New Project -> Upload Project** (Compiler: `pdfLaTeX`).
