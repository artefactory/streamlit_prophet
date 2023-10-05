SHELL := /usr/bin/env bash

IMAGE := streamlit_prophet
VERSION := latest

NO_CHECK_FLAG =  || true

ifeq ($(STRICT), 1)
	POETRY_COMMAND_FLAG =
	PIP_COMMAND_FLAG =
	SAFETY_COMMAND_FLAG =
	BANDIT_COMMAND_FLAG =
	SECRETS_COMMAND_FLAG =
	BLACK_COMMAND_FLAG =
	DARGLINT_COMMAND_FLAG =
	ISORT_COMMAND_FLAG =
	MYPY_COMMAND_FLAG =
else
	POETRY_COMMAND_FLAG = $(NO_CHECK_FLAG)
	PIP_COMMAND_FLAG = $(NO_CHECK_FLAG)
	SAFETY_COMMAND_FLAG = $(NO_CHECK_FLAG)
	BANDIT_COMMAND_FLAG = $(NO_CHECK_FLAG)
	SECRETS_COMMAND_FLAG = $(NO_CHECK_FLAG)
	BLACK_COMMAND_FLAG = $(NO_CHECK_FLAG)
	DARGLINT_COMMAND_FLAG = $(NO_CHECK_FLAG)
	ISORT_COMMAND_FLAG = $(NO_CHECK_FLAG)
	MYPY_COMMAND_FLAG = $(NO_CHECK_FLAG)
endif

ifeq ($(POETRY_STRICT), 1)
	POETRY_COMMAND_FLAG =
else ifeq ($(POETRY_STRICT), 0)
	POETRY_COMMAND_FLAG = $(NO_CHECK_FLAG)
endif

ifeq ($(PIP_STRICT), 1)
	PIP_COMMAND_FLAG =
else ifeq ($(PIP_STRICT), 0)
	PIP_COMMAND_FLAG = $(NO_CHECK_FLAG)
endif

ifeq ($(SAFETY_STRICT), 1)
	SAFETY_COMMAND_FLAG =
else ifeq ($(SAFETY_STRICT), 0)
	SAFETY_COMMAND_FLAG = $(NO_CHECK_FLAG)
endif

ifeq ($(BANDIT_STRICT), 1)
	BANDIT_COMMAND_FLAG =
else ifeq ($(BANDIT_STRICT), 0)
	BANDIT_COMMAND_FLAG = $(NO_CHECK_FLAG)
endif

ifeq ($(SECRETS_STRICT), 1)
	SECRETS_COMMAND_FLAG =
else ifeq ($(SECRETS_STRICT), 0)
	SECRETS_COMMAND_FLAG = $(NO_CHECK_FLAG)
endif

ifeq ($(BLACK_STRICT), 1)
	BLACK_COMMAND_FLAG =
else ifeq ($(BLACK_STRICT), 0)
	BLACK_COMMAND_FLAG = $(NO_CHECK_FLAG)
endif

ifeq ($(DARGLINT_STRICT), 1)
	DARGLINT_COMMAND_FLAG =
else ifeq ($(DARGLINT_STRICT), 0)
	DARGLINT_COMMAND_FLAG = $(NO_CHECK_FLAG)
endif

ifeq ($(ISORT_STRICT), 1)
	ISORT_COMMAND_FLAG =
else ifeq ($(ISORT_STRICT), 0)
	ISORT_COMMAND_FLAG = $(NO_CHECK_FLAG)
endif

ifeq ($(MYPY_STRICT), 1)
	MYPY_COMMAND_FLAG =
else ifeq ($(MYPY_STRICT), 0)
	MYPY_COMMAND_FLAG = $(NO_CHECK_FLAG)
endif

.PHONY: download-poetry
download-poetry:
	curl -sSL https://raw.githubusercontent.com/python-poetry/install.python-poetry.org/main/install-poetry.py | python

.PHONY: install
install:
	poetry env use 3.8
	poetry lock -n
	poetry install -n
ifneq ($(NO_PRE_COMMIT), 1)
	poetry run pre-commit install -t pre-commit -t pre-push
endif

.PHONY: check-safety
check-safety:
	poetry check$(POETRY_COMMAND_FLAG) && \
	poetry run pip check$(PIP_COMMAND_FLAG) && \
	poetry run safety check --full-report$(SAFETY_COMMAND_FLAG) && \
	poetry run bandit -r streamlit_prophet/$(BANDIT_COMMAND_FLAG)

.PHONY: gitleaks
gitleaks:
	commits="$$(git rev-list --ancestry-path $$(git rev-parse $$(git branch -r --sort=committerdate | tail -1))..$$(git rev-parse HEAD))"; \
	if [ "$${commits}" != "" ]; then docker run --rm -v $$(pwd):/code/ zricethezav/gitleaks --path=/code/ -v --commits=$$(echo $${commits} | paste -s -d, -); fi;

.PHONY: check-style
check-style:
	poetry run black --config pyproject.toml --diff --check ./$(BLACK_COMMAND_FLAG) && \
	poetry run darglint -v 2 **/*.py$(DARGLINT_COMMAND_FLAG) && \
	poetry run isort --settings-path pyproject.toml --check-only **/*.py$(ISORT_COMMAND_FLAG) && \
	poetry run mypy --config-file setup.cfg streamlit_prophet tests/**/*.py$(MYPY_COMMAND_FLAG)

.PHONY: format-code
format-code:
	poetry run pre-commit run

.PHONY: test
test:
	poetry run pytest

.PHONY: lint
lint: test check-safety check-style

# Example: make docker VERSION=latest
# Example: make docker IMAGE=some_name VERSION=0.1.0
.PHONY: docker
docker:
	@echo Building docker $(IMAGE):$(VERSION) ...
	docker build \
		-t $(IMAGE):$(VERSION) . \
		-f ./Dockerfile

# Example: make clean_docker VERSION=latest
# Example: make clean_docker IMAGE=some_name VERSION=0.1.0
.PHONY: clean_docker
clean_docker:
	@echo Removing docker $(IMAGE):$(VERSION) ...
	docker rmi -f $(IMAGE):$(VERSION)

.PHONY: clean_build
clean_build:
	rm -rf build/

.PHONY: clean
clean: clean_build clean_docker
