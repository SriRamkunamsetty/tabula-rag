.PHONY: install dev-install test cov lint type fmt check demo run docker-build docker-up eval eval-ablation bench clean

VENV_PY := python3

install:
	$(VENV_PY) -m pip install -e .

dev-install:
	$(VENV_PY) -m pip install -e ".[dev]"

test:
	pytest -q

cov:
	pytest -q --cov=tabula_rag --cov-report=term-missing --cov-report=html

lint:
	ruff check src tests scripts benchmarks

fmt:
	ruff format src tests scripts benchmarks
	ruff check --fix src tests scripts benchmarks

type:
	mypy

check: lint type test
	@echo "all checks passed"

demo:
	$(VENV_PY) -m tabula_rag.cli demo

run:
	uvicorn tabula_rag.api:create_app --factory --reload --port 8090

docker-build:
	docker build -t tabula-rag:latest .

docker-up:
	docker compose --profile gpu up --build

eval:
	$(VENV_PY) -m tabula_rag.cli evaluate --dataset eval/golden.jsonl --report reports/eval.json

eval-ablation:
	$(VENV_PY) scripts/run_ablation.py

bench:
	$(VENV_PY) benchmarks/bench_pipeline.py --requests 60 --concurrency 12

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
