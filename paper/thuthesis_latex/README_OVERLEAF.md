# 清华大学硕士学位论文 XeLaTeX 源码包（Overleaf 即传就绪）

本目录提供《短时姿态观测下的非结构化道路行人动作与轨迹联合预测》（作者：彭嘉钰，指导教师：姚丹亚教授）的 `ctexrep` 标准 `XeLaTeX` 源码包，无需额外私有字体或定制模板即可在任意标准 Overleaf / TeX Live 环境直接编译。

## 文件说明
- `main.tex`：完整中文硕士学位论文源码（含封面、中英文摘要、目录、第 1–6 章、形式化命题 3.1–3.3、`booktabs` 核心实验表与 300-DPI 图 4.1–4.4）。
- `references.bib`：完整 BibTeX 文献库（`21` 篇）。

## 编译与一键 Overleaf 上传
1. **本地 MacTeX / TeX Live 编译**：
   ```bash
   cd paper/thuthesis_latex
   xelatex main && bibtex main && xelatex main && xelatex main
   ```
2. **生成 Overleaf 即传压缩包**：
   ```bash
   python3 scripts/build_overleaf_bundle.py
   ```
   将生成的 `dist/overleaf_thuthesis_chinese.zip` 上传至 **Overleaf -> New Project -> Upload Project**，并在 **Menu -> Compiler** 中选择 **`XeLaTeX`** 即可一键生成完整中文学位论文 PDF。
