# Default target
.PHONY: all
all: help

## Clean up generated files and virtual environment
.PHONY: clean
clean:
	rm -rf .venv
	rm -rf __pycache__
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf build

## Run all tests
.PHONY: test
test:
	pytest tests/

## Check code style with ruff (linting and formatting)
.PHONY: lint
lint:
	ruff check .
	ruff format --check .

## Fix code style issues with ruff
.PHONY: format
format:
	ruff check . --fix
	ruff format .

## Check and fix code style (combined target)
.PHONY: style
style: format


## Start the Environment CLI for interacting with domain environments
.PHONY: env-cli
env-cli:
	python -m tau2.environment.utils.interface_agent

## Display online help for commonly used targets in this Makefile
.PHONY: help
help:
	@awk '/^[a-zA-Z_\/\.0-9-]+:/ {        \
		nb = sub( /^## /, "", helpMsg );  \
		if (nb)                           \
			print  $$1 "\t" helpMsg;      \
	}                                     \
	{ helpMsg = $$0 }' $(MAKEFILE_LIST) | \
	column -ts $$'\t' |                   \
	expand -t 1 |                         \
	grep --color '^[^ ]*'
