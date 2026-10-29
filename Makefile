# Developer entry points. Every target is listed in CLAUDE.md; stubs name the task that implements them.
.PHONY: setup fmt fmt-check lint test check kaggle-run kaggle-pull results report bench serve \
	docker-build docker-run deploy

UV := uv run

# Load KAGGLE_USERNAME etc. from the local, gitignored .env when it exists (CI has none).
-include .env
export

setup:
	uv sync
	$(UV) pre-commit install

fmt:
	$(UV) ruff format .

fmt-check:
	$(UV) ruff format --check .

lint:
	$(UV) ruff check .

test:
	$(UV) pytest -q

check: fmt-check lint test

# RUN is a config path for kaggle-run (e.g. configs/kaggle/env_check.yaml) and a run_id for kaggle-pull.
kaggle-run:
	$(UV) python scripts/kaggle_run.py --config $(RUN)

kaggle-pull:
	$(UV) python scripts/kaggle_pull.py $(RUN)

# Copy finished runs' small artifacts into results/ and rebuild results/experiments.csv.
results:
	$(UV) python scripts/collect_results.py

# Stubs below fail loudly so a missing target is never mistaken for a successful one.

# Regenerate generated tables inside reports/*.md from results/ (P3.6 experiments table; P5.1 the rest).
report:
	$(UV) python scripts/build_report_tables.py

bench:
	@echo "not implemented yet (P6.3)"; exit 1

serve:
	@echo "not implemented yet (P7.1)"; exit 1

docker-build:
	@echo "not implemented yet (P7.4)"; exit 1

docker-run:
	@echo "not implemented yet (P7.4)"; exit 1

deploy:
	@echo "not implemented yet (P7.7)"; exit 1
