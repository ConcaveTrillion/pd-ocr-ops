AI ?=
LOG := .ci-ai.log

ifdef AI
_goals := $(or $(MAKECMDGOALS),ci)
.PHONY: $(_goals)
$(_goals):
	@rm -f $(LOG)
	@$(MAKE) --no-print-directory AI= $@ > $(LOG) 2>&1 \
		&& echo "✅ $@ passed (log: $(LOG))" \
		|| (echo "❌ $@ failed:"; tail -50 $(LOG); echo "(full log: $(LOG))"; exit 1)

else

.PHONY: setup lint format test ci build clean pre-commit-check

setup: ## Install dependencies
	uv sync --group dev

lint: ## Run linting checks
	uv run ruff check pd_ocr_ops tests

format: ## Format code
	uv run ruff format pd_ocr_ops tests

test: ## Run tests with parallelization
	uv run pytest -n auto

pre-commit-check: ## Run all pre-commit hooks against all files (read-only check)
	uv run pre-commit run --all-files

ci: lint test ## Run full CI pipeline

build: ## Build the project
	uv build

clean: ## Clean cache and temporary files
	rm -rf dist .venv .pytest_cache .ruff_cache .ci-ai.log

endif
