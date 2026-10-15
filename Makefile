UV ?= uv
RESULTS ?= results/local
PYTHON ?= $(UV) run --locked python
TEX = pdflatex
METHOD_PDFS = draft/figures/tiling/packing.pdf draft/figures/checkpoint/execution.pdf

.PHONY: paper arxiv check benchmarks figures frontier-figures method-figures results build-tokenizer workloads source-archive

paper: frontier-figures
	$(PYTHON) analysis_tools/generate_paper.py
ifeq ($(TEX),tectonic)
	cd build/paper && tectonic --keep-logs --keep-intermediates main.tex
else
	cd build/paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd build/paper && bibtex main
	cd build/paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd build/paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
endif

arxiv: frontier-figures
	$(PYTHON) analysis_tools/generate_paper.py --target arxiv
	cd build/arxiv && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd build/arxiv && bibtex main
	cd build/arxiv && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd build/arxiv && pdflatex -interaction=nonstopmode -halt-on-error main.tex

check:
	$(PYTHON) analysis_tools/check_paper.py

benchmarks:
	cd benchmarks && $(UV) run --locked python -m harness run

figures: method-figures frontier-figures

frontier-figures:
	$(PYTHON) -m analysis_tools.plot_frontiers

method-figures: $(METHOD_PDFS)
	$(PYTHON) analysis_tools/export_figure_previews.py $(METHOD_PDFS)

draft/figures/%.pdf: draft/figures/%.tex draft/figures/%-body.tex
	mkdir -p build/figures/$(dir $*)
ifeq ($(TEX),tectonic)
	cd $(dir $<) && tectonic --keep-logs -o $(CURDIR)/build/figures/$(dir $*) $(notdir $<)
else
	cd $(dir $<) && $(TEX) -interaction=nonstopmode -halt-on-error -output-directory=$(CURDIR)/build/figures/$(dir $*) $(notdir $<)
endif
	cp build/figures/$*.pdf $@

results:
	$(PYTHON) analysis_tools/summarize_results.py --comparison $(RESULTS)/raw/comparison --ablation $(RESULTS)/raw/ablation --output $(RESULTS)

build-tokenizer:
	$(UV) run --locked --group data python -m analysis_tools.train_tokenizer

workloads:
	$(UV) run --locked --project benchmarks python -m analysis_tools.generate_workloads nvidia/Granary --output data/workloads/workloads.json

source-archive:
	mkdir -p dist
	git archive --format=zip --output=dist/flash-rnnt-paper-source.zip HEAD
