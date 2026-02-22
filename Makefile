SHELL := /bin/bash

ENV_NAME ?= cbfkit
PYTHON_VERSION ?= 3.10

.PHONY: venv
venv:
	@if ! command -v conda >/dev/null 2>&1; then \
		echo "conda not found in PATH"; \
		exit 1; \
	fi
	@eval "$$(conda shell.bash hook)" && \
	if conda env list | awk '{print $$1}' | grep -qx "$(ENV_NAME)"; then \
		echo "Conda env '$(ENV_NAME)' already exists"; \
	else \
		conda create -n "$(ENV_NAME)" python="$(PYTHON_VERSION)" -y; \
	fi && \
	conda activate "$(ENV_NAME)" && \
	python -m pip install --upgrade pip && \
	python -m pip install poetry && \
	poetry config virtualenvs.create false --local && \
	poetry install
