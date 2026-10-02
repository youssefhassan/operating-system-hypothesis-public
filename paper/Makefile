# Preprint I build. Source is pandoc-markdown (paper.md); LaTeX and PDF are generated.
#   make          -> paper.pdf
#   make tex      -> paper.tex (what gets uploaded to arXiv, with figures/ and refs.bib)
#   make check    -> number ledger + overclaim + em-dash checks
#   make arxiv    -> arxiv_upload.tar.gz (tex + bbl + figures)
PANDOC   = pandoc
TECTONIC = tectonic
SRC      = paper.md
BIB      = refs.bib
TPL      = template.tex
FIGS     = $(wildcard figures/*.png figures/*.pdf)

all: paper.pdf

paper.tex: $(SRC) $(BIB) $(TPL)
	$(PANDOC) $(SRC) --from markdown+tex_math_dollars+raw_tex --to latex \
	  --template=$(TPL) --columns=80 --natbib --bibliography=$(BIB) --standalone -o $@
	python3 fix_tables.py $@

paper.pdf: paper.tex $(FIGS)
	$(TECTONIC) -X compile --keep-intermediates --keep-logs $< 

tex: paper.tex

check:
	python3 check_numbers.py $(SRC) NUMBERS_LEDGER.md
	python3 audit_numbers.py $(SRC)
	python3 check_claims.py $(SRC)

arxiv: paper.pdf
	# only the figures the LaTeX actually includes
	python3 prepare_arxiv.py
	rm -rf arxiv/figures && mkdir -p arxiv/figures && cp $$(grep -o 'figures/[A-Za-z0-9_./-]*\.\(png\|pdf\)' paper.tex | sort -u) arxiv/figures/
	cd arxiv && COPYFILE_DISABLE=1 tar czf ../arxiv_upload.tar.gz paper.tex figures

clean:
	rm -rf paper.tex paper.pdf paper.aux paper.bbl paper.blg paper.log paper.out arxiv_upload.tar.gz arxiv

.PHONY: all tex check arxiv clean
