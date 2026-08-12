.PHONY: help install synth clean-data eda emissions models bench bench-update test lint all

PY ?= python

help:
	@echo "make install    install the package and dependencies (editable)"
	@echo "make synth      generate a synthetic 1.27M-row dataset (no Safran data needed)"
	@echo "make all        synth -> clean -> eda -> emissions -> models"
	@echo "make bench      run the vectorisation benchmark"
	@echo "make bench-update  benchmark THIS machine, then rewrite the numbers in the docs"
	@echo "make test       run the test suite"
	@echo "make lint       ruff check"

install:
	$(PY) -m pip install -e ".[dev]"

synth:
	$(PY) scripts/00_make_synthetic_data.py

clean-data:
	$(PY) scripts/01_run_cleaning.py

eda:
	$(PY) scripts/02_run_eda.py

emissions:
	$(PY) scripts/03_run_emissions.py

models:
	$(PY) scripts/04_run_models.py

bench:
	$(PY) scripts/benchmark_vectorization.py

# Measure on this machine, then propagate the numbers into README.md and
# configs/reference_results.yaml. Quote YOUR figure, not the reference one.
bench-update:
	$(PY) scripts/benchmark_vectorization.py
	$(PY) scripts/update_docs_benchmark.py

test:
	$(PY) -m pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/

all: synth clean-data eda emissions models
