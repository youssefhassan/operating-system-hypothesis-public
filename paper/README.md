# paper/ : Preprint I

Source of truth is `paper.md` (pandoc markdown). LaTeX and PDF are generated; do not edit
`paper.tex` by hand.

```
make            # paper.pdf via pandoc -> tectonic
make check      # every number traces to a JSON; overclaim and voice scan
make arxiv      # arxiv_upload.tar.gz (tex, bbl, figures)
python3 build_ledger.py   # regenerate NUMBERS_LEDGER.md + numbers_flat.json from the JSONs
../.venv/bin/python make_figures.py   # the six figures (fig1, fig2, fig3_slopes, fig3_style, fig5_painterly, fig4 files) from the JSONs and PNGs
```

Rules for this directory:

- The author writes every claim sentence. Agents fetch, structure, check numbers, build
  figures and bibliography, and convert prose into LaTeX. They do not interpret results.
- Every number in `paper.md` must trace to a row in `NUMBERS_LEDGER.md` or a justified
  line in `numbers_allow.txt`. `make check` enforces it.
- `refs.bib` holds only entries checked against their source; unchecked candidates are
  never cited.
- No em dashes, no curly quotes in `paper.md`.
- `[YOU: ...]` marks a slot the author fills. `make check` does not count numbers inside them.
