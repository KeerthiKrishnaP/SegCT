PYTHON_FILE_PATHS = `(find . -iname "*.py" -not -path "./.venv/*")`

install: ## Install dependencies
	poetry install

install-hard: ## Clear and install dependencies
## rm -rf yarn.lock node_modules && yarn install --ignore-optional
	rm -rf poetry.lock .venv & make install

poetry-update: ## Upgrade poetry and dependencies
	poetry self update
	poetry run pip install --upgrade pip wheel setuptools
	poetry update

format: ## Format code using ruff format and cargo fmt
	poetry run ruff format $(PYTHON_FILES)
	cargo fmt

format-check: ## Check code format using ruff format and cargo fmt
	poetry run ruff format --check $(PYTHON_FILES)
	cargo fmt --check

lint-check: ## Run all linters
	poetry run ruff check $(PYTHON_FILES)
	cargo clippy

lint: ## Run all linters with automated fix
	poetry run ruff --fix $(PYTHON_FILES)
	cargo clippy --fix

toml-sort: ## Reorder toml files
	poetry run toml-sort --all --in-place $(TOML_FILES)

