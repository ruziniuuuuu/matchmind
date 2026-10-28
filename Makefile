.PHONY: smoke-azure install smoke data dev test lint web-install web-build web-dev static-demo evaluate perf

install:
	uv sync --extra dev

smoke:
	uv run python scripts/smoke_deepseek.py

data:
	uv run python -m matchmind.matchforge

dev:
	uv run uvicorn matchmind.api.main:app --reload --port 8010

test:
	uv run pytest -q

lint:
	uv run ruff check .

web-install:
	cd web && npm install --include=dev

web-build:
	cd web && npm run build

web-dev:
	cd web && npm run dev

static-demo:
	uv run python scripts/export_static_demo.py
	cd web && npm run build:static

evaluate:
	uv run python scripts/evaluate_fixtures.py

perf:
	uv run python scripts/benchmark.py

smoke-azure:
	uv run python scripts/smoke_azure.py
