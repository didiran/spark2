.PHONY: help install test lint format run clean docker-up docker-down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install black isort flake8 mypy pytest-cov

test: ## Run tests
	python -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint: ## Run linters
	flake8 src/ tests/ main.py --max-line-length=120 --ignore=E501,W503
	mypy src/ --ignore-missing-imports --no-strict-optional

format: ## Format code
	black src/ tests/ main.py --line-length=120
	isort src/ tests/ main.py --profile black

run: ## Run the pipeline demo
	python main.py

run-small: ## Run with small dataset
	python main.py --samples 1000 --batches 2

run-large: ## Run with large dataset
	python main.py --samples 20000 --batches 5

clean: ## Clean build artifacts
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage
	rm -rf demo_feature_store demo_results logs
	rm -rf dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

docker-build: ## Build Docker image
	docker-compose -f docker/docker-compose.yml build

docker-up: ## Start services with Docker Compose
	docker-compose -f docker/docker-compose.yml up -d

docker-down: ## Stop Docker Compose services
	docker-compose -f docker/docker-compose.yml down -v

docker-logs: ## View Docker Compose logs
	docker-compose -f docker/docker-compose.yml logs -f
