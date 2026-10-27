# Developer entry points. Every target is listed in CLAUDE.md; stubs name the task that implements them.
.PHONY: setup fmt fmt-check lint test check kaggle-run kaggle-pull results report bench serve \
	docker-build docker-run deploy

UV := uv run

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

# Stubs below fail loudly so a missing target is never mistaken for a successful one.
kaggle-run:
	@echo "not implemented yet (P0.6): RUN=$(RUN)"; exit 1

kaggle-pull:
	@echo "not implemented yet (P0.6): RUN=$(RUN)"; exit 1

results:
	@echo "not implemented yet (P2.5)"; exit 1

report:
	@echo "not implemented yet (P5.1)"; exit 1

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
