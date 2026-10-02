#!/usr/bin/env python3
"""Build arxiv/paper.tex with the bibliography inlined, so arXiv never runs BibTeX.

arXiv runs BibTeX whenever the source says \\bibliography{...}; refs.bib is not uploaded,
so that step fails. The pre-built paper.bbl replaces the \\bibliographystyle and
\\bibliography lines, verbatim. Usage: prepare_arxiv.py  (from paper/, after make)
"""
import os, re

os.makedirs("arxiv", exist_ok=True)
tex = open("paper.tex", encoding="utf8").read()
bbl = open("paper.bbl", encoding="utf8").read()
tex, n_style = re.subn(r"^\\bibliographystyle\{[^}]*\}\s*\n", "", tex, flags=re.M)
tex, n_bib = re.subn(r"^\\bibliography\{[^}]*\}", lambda m: bbl.strip(), tex, flags=re.M)
assert n_bib == 1, "expected exactly one \\bibliography line"
open("arxiv/paper.tex", "w", encoding="utf8").write(tex)
print(f"prepare_arxiv: bibliography inlined ({bbl.count(chr(92) + 'bibitem')} entries), style lines removed: {n_style}")
