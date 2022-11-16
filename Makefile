PYTHON_FILE_PATHS = `(find . -iname "*.py" -not -path "./.venv/*")`

install: ## Install dependencies
	poetry install

hard-install: ## Clear and install dependencies
## rm -rf yarn.lock node_modules && yarn install --ignore-optional
	rm -rf poetry.lock .venv & make install

pytest: ## Run Pytest
	poetry run pytest tests/