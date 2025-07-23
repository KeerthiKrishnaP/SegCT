# Set the virtual environment directory to always be in the current working directory
ENV_DIR=$(shell pwd)/.venv
UV=uv

# Default target
.DEFAULT_GOAL := help

## Create a virtual environment with uv
env:
	@$(UV) venv $(ENV_DIR)
	@echo "Virtual environment created in $(ENV_DIR)"

## Install dependencies from pyproject.toml
install: env
	@$(UV) pip sync pyproject.toml
	@echo "Dependencies installed from pyproject.toml"

## Run the application
run:
	@$(UV) python main.py  # Change main.py to your entry script

## Upgrade dependencies
upgrade:
	@$(UV) pip install --upgrade
	@echo "Dependencies upgraded"

## Remove virtual environment
clean:
	@rm -rf $(ENV_DIR)
	@echo "Virtual environment removed"

## Clear Python/tool caches (__pycache__, .pytest_cache, etc.)
clear-cache:
	@echo "Removing Python caches..."
	@find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" -o -name ".hypothesis" -o -name ".ipynb_checkpoints" -o -name ".tox" \) -exec rm -rf {} +
	@find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name ".coverage" \) -delete
	@echo "Caches removed."

## Remove existing venv and create a new one
install-hard: clean env install
	@echo "Reinstalled virtual environment and dependencies"

## Show available make commands
help:
	@echo "Makefile Commands:"
	@echo "  make env       - Create virtual environment using uv"
	@echo "  make install   - Install dependencies from pyproject.toml"
	@echo "  make run       - Run the application"
	@echo "  make upgrade   - Upgrade dependencies"
	@echo "  make clean     - Remove virtual environment"
	@echo "  make reinstall - Remove and create a fresh virtual environment"
	@echo "  make help      - Show this help message"
