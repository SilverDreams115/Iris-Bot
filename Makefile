PYTHON := .venv/bin/python
PIP := $(PYTHON) -m pip

.PHONY: bootstrap install-dev lint typecheck test smoke check

bootstrap:
	test -d .venv || python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install-dev:
	$(PIP) install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -m iris_bot.main --help

check: lint typecheck test smoke
