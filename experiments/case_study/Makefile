PYTHON := .venv/bin/python

.PHONY: sync test run refresh clean

sync:
	uv sync

test:
	$(PYTHON) -m pytest -q tests

run:
	bash run_case_study.sh

refresh:
	bash fetch_packages.sh

clean:
	rm -rf .pytest_cache
	rm -f .coverage .coverage.* .coverage.*.json case_study_summary.json
	rm -rf htmlcov_*
