UV ?= uv
RESULTS ?= results/local
PYTHON ?= $(UV) run --locked python
TEX ?= pdflatex
FIGURE_TEX ?= $(if $(filter tectonic,$(TEX)),tectonic,xelatex)
METHOD_PDFS = draft/figures/tiling/packing.pdf draft/figures/checkpoint/execution.pdf

.PHONY: paper check benchmarks figures method-figures results build-tokenizer workloads source-archive

paper:
	$(PYTHON) analysis_tools/generate_paper.py
ifeq ($(TEX),tectonic)
	cd build/paper && tectonic --keep-logs --keep-intermediates main.tex
else
	cd build/paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd build/paper && bibtex main
	cd build/paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd build/paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
endif

check:
	$(PYTHON) analysis_tools/check_paper.py

benchmarks:
	cd benchmarks && $(UV) run --locked python -m harness run

figures: method-figures
	$(PYTHON) -m analysis_tools.plot_frontiers

method-figures: $(METHOD_PDFS)
	$(PYTHON) analysis_tools/export_figure_previews.py $(METHOD_PDFS)

draft/figures/%.pdf: draft/figures/%.tex
	mkdir -p build/figures/$(dir $*)
ifeq ($(FIGURE_TEX),tectonic)
	tectonic --keep-logs -o build/figures/$(dir $*) $<
else
	$(FIGURE_TEX) -interaction=nonstopmode -halt-on-error -output-directory=build/figures/$(dir $*) $<
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
