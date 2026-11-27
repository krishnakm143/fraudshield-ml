.PHONY: install data train evaluate drift run test lint format docker clean

install:
	pip install -e ".[dev]"

data:
	fraudshield generate --out data

train:
	fraudshield train

evaluate:
	python -m fraudshield.evaluate --json reports/metrics.json

drift:
	fraudshield drift

run:
	uvicorn fraudshield.api:app --reload --port 8000

test:
	python -m pytest -q

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

docker:
	docker build -t fraudshield:latest .

clean:
	rm -rf models .pytest_cache .ruff_cache build dist *.egg-info
