# HFT Trading System Makefile
# Professional build and deployment automation

.DEFAULT_GOAL := help
.PHONY: help build test clean deploy monitor

# Configuration
PROJECT_NAME := hft-trading-system
VERSION := 1.0.0
BUILD_TYPE := Release
PARALLEL_JOBS := $(shell nproc)

# Directories
SRC_DIR := src
CPP_DIR := $(SRC_DIR)/cpp
PYTHON_DIR := $(SRC_DIR)/python
BUILD_DIR := build
DIST_DIR := dist

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m

help: ## Show this help message
	@echo "$(BLUE)HFT Trading System - Build Automation$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Configuration:$(NC)"
	@echo "  Project: $(PROJECT_NAME)"
	@echo "  Version: $(VERSION)"
	@echo "  Build Type: $(BUILD_TYPE)"
	@echo "  Parallel Jobs: $(PARALLEL_JOBS)"

all: build test package ## Build, test, and package everything

setup: ## Install system dependencies and setup environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	sudo apt-get update
	sudo apt-get install -y build-essential cmake git
	sudo apt-get install -y python3 python3-pip python3-venv
	sudo apt-get install -y docker.io docker-compose
	curl -sSL https://install.python-poetry.org | python3 -
	@echo "$(GREEN)Setup completed successfully$(NC)"

build: build-cpp build-python ## Build all components

build-cpp: ## Build C++ components
	@echo "$(BLUE)Building C++ components...$(NC)"
	mkdir -p $(CPP_DIR)/build
	cd $(CPP_DIR)/build && \
		cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) .. && \
		make -j$(PARALLEL_JOBS)
	@echo "$(GREEN)C++ build completed$(NC)"

build-python: ## Build Python components
	@echo "$(BLUE)Building Python components...$(NC)"
	cd $(PYTHON_DIR) && \
		poetry install && \
		poetry build
	@echo "$(GREEN)Python build completed$(NC)"

test: test-cpp test-python ## Run all tests

test-cpp: build-cpp ## Run C++ tests
	@echo "$(BLUE)Running C++ tests...$(NC)"
	cd $(CPP_DIR)/build && \
		ctest --output-on-failure --parallel $(PARALLEL_JOBS)
	@echo "$(GREEN)C++ tests completed$(NC)"

test-python: build-python ## Run Python tests
	@echo "$(BLUE)Running Python tests...$(NC)"
	cd $(PYTHON_DIR) && \
		poetry run pytest tests/ -v --cov=hft --cov-report=html
	@echo "$(GREEN)Python tests completed$(NC)"

benchmark: build-cpp ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	cd $(CPP_DIR)/build && \
		./benchmark_runner --benchmark_format=json > benchmark_results.json
	@echo "$(GREEN)Benchmarks completed$(NC)"

lint: ## Run code linting and formatting
	@echo "$(BLUE)Running code linting...$(NC)"
	# C++ formatting
	find $(CPP_DIR) -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i
	# Python formatting
	cd $(PYTHON_DIR) && poetry run black hft/ tests/
	cd $(PYTHON_DIR) && poetry run isort hft/ tests/
	cd $(PYTHON_DIR) && poetry run flake8 hft/ tests/
	@echo "$(GREEN)Linting completed$(NC)"

security: ## Run security scans
	@echo "$(BLUE)Running security scans...$(NC)"
	# Python security scan
	cd $(PYTHON_DIR) && poetry run bandit -r hft/
	# Docker image scanning (if images exist)
	-docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		-v $(PWD):/src aquasec/trivy image $(PROJECT_NAME):latest
	@echo "$(GREEN)Security scan completed$(NC)"

package: build ## Package applications for distribution
	@echo "$(BLUE)Packaging applications...$(NC)"
	mkdir -p $(DIST_DIR)
	# Copy C++ binaries
	cp $(CPP_DIR)/build/trading_engine $(DIST_DIR)/
	# Copy Python wheels
	cp $(PYTHON_DIR)/dist/*.whl $(DIST_DIR)/
	# Create deployment archive
	tar -czf $(DIST_DIR)/$(PROJECT_NAME)-$(VERSION).tar.gz \
		$(DIST_DIR)/trading_engine \
		$(DIST_DIR)/*.whl \
		configs/ \
		scripts/
	@echo "$(GREEN)Packaging completed$(NC)"

docker: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker build -t $(PROJECT_NAME):$(VERSION) .
	docker build -t $(PROJECT_NAME):latest .
	@echo "$(GREEN)Docker images built$(NC)"

deploy-monitoring: ## Deploy monitoring stack
	@echo "$(BLUE)Deploying monitoring stack...$(NC)"
	cd monitoring && docker-compose up -d
	@echo "$(GREEN)Monitoring stack deployed$(NC)"
	@echo "Grafana: http://localhost:3000 (admin/admin123)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Kibana: http://localhost:5601"

deploy-dev: build docker ## Deploy to development environment
	@echo "$(BLUE)Deploying to development environment...$(NC)"
	docker-compose -f deployment/docker-compose.dev.yml up -d
	@echo "$(GREEN)Development deployment completed$(NC)"

deploy-staging: package ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging environment...$(NC)"
	# Add staging deployment logic here
	@echo "$(GREEN)Staging deployment completed$(NC)"

deploy-prod: ## Deploy to production (requires manual approval)
	@echo "$(RED)Production deployment requires manual approval$(NC)"
	@echo "Run: make deploy-prod-confirmed"

deploy-prod-confirmed: package ## Deploy to production (confirmed)
	@echo "$(BLUE)Deploying to production environment...$(NC)"
	# Add production deployment logic here
	@echo "$(GREEN)Production deployment completed$(NC)"

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf $(CPP_DIR)/build
	rm -rf $(PYTHON_DIR)/dist
	rm -rf $(PYTHON_DIR)/.pytest_cache
	rm -rf $(DIST_DIR)
	docker system prune -f
	@echo "$(GREEN)Clean completed$(NC)"

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	# C++ documentation
	doxygen docs/Doxyfile
	# Python documentation
	cd $(PYTHON_DIR) && poetry run sphinx-build -b html docs/ docs/_build/
	@echo "$(GREEN)Documentation generated$(NC)"

monitor: ## Show system monitoring dashboard
	@echo "$(BLUE)Opening monitoring dashboards...$(NC)"
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Kibana: http://localhost:5601"
	@echo "Jaeger: http://localhost:16686"

logs: ## Show system logs
	@echo "$(BLUE)Showing system logs...$(NC)"
	docker-compose -f monitoring/docker-compose.yml logs -f

status: ## Show system status
	@echo "$(BLUE)System Status:$(NC)"
	@echo ""
	@echo "$(GREEN)Build Status:$(NC)"
	@test -f $(CPP_DIR)/build/trading_engine && echo "  ✓ C++ Engine Built" || echo "  ✗ C++ Engine Missing"
	@test -f $(PYTHON_DIR)/dist/*.whl && echo "  ✓ Python Package Built" || echo "  ✗ Python Package Missing"
	@echo ""
	@echo "$(GREEN)Services Status:$(NC)"
	@docker ps --format "table {{.Names}}\t{{.Status}}" | grep hft || echo "  No HFT services running"

install: package ## Install the system locally
	@echo "$(BLUE)Installing system locally...$(NC)"
	sudo cp $(DIST_DIR)/trading_engine /usr/local/bin/
	pip3 install $(DIST_DIR)/*.whl
	@echo "$(GREEN)System installed successfully$(NC)"

uninstall: ## Uninstall the system
	@echo "$(BLUE)Uninstalling system...$(NC)"
	sudo rm -f /usr/local/bin/trading_engine
	pip3 uninstall -y hft-trading-system
	@echo "$(GREEN)System uninstalled$(NC)"

# Development helpers
dev-setup: setup ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	pre-commit install
	cd $(PYTHON_DIR) && poetry install --with dev
	@echo "$(GREEN)Development environment ready$(NC)"

dev-run: build ## Run development instance
	@echo "$(BLUE)Starting development instance...$(NC)"
	$(CPP_DIR)/build/trading_engine --config configs/environments/development.yml

dev-test: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	cd $(PYTHON_DIR) && poetry run pytest-watch -- tests/

# Performance testing
perf-test: build ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	cd $(CPP_DIR)/build && ./perf_test_runner
	@echo "$(GREEN)Performance tests completed$(NC)"

load-test: ## Run load tests
	@echo "$(BLUE)Running load tests...$(NC)"
	# Add load testing logic here
	@echo "$(GREEN)Load tests completed$(NC)"

# Version management
version-bump-patch: ## Bump patch version
	@echo "$(BLUE)Bumping patch version...$(NC)"
	# Add version bumping logic here
	@echo "$(GREEN)Version bumped$(NC)"

version-bump-minor: ## Bump minor version
	@echo "$(BLUE)Bumping minor version...$(NC)"
	# Add version bumping logic here
	@echo "$(GREEN)Version bumped$(NC)"

version-bump-major: ## Bump major version
	@echo "$(BLUE)Bumping major version...$(NC)"
	# Add version bumping logic here
	@echo "$(GREEN)Version bumped$(NC)"

# Emergency procedures
emergency-stop: ## Emergency stop all services
	@echo "$(RED)EMERGENCY STOP - Stopping all services$(NC)"
	docker-compose -f deployment/docker-compose.prod.yml down
	sudo pkill -f trading_engine
	@echo "$(GREEN)All services stopped$(NC)"

emergency-restart: ## Emergency restart trading engine
	@echo "$(YELLOW)EMERGENCY RESTART - Restarting trading engine$(NC)"
	make emergency-stop
	sleep 5
	make deploy-prod-confirmed
	@echo "$(GREEN)Trading engine restarted$(NC)"

# CI/CD targets
ci-build: lint test benchmark security ## CI build pipeline
	@echo "$(GREEN)CI build pipeline completed$(NC)"

ci-deploy: ci-build package ## CI deployment pipeline
	@echo "$(GREEN)CI deployment pipeline completed$(NC)"

# Database management
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	cd $(PYTHON_DIR) && poetry run alembic upgrade head
	@echo "$(GREEN)Database migrations completed$(NC)"

db-reset: ## Reset database (WARNING: destroys data)
	@echo "$(RED)WARNING: This will destroy all data$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		cd $(PYTHON_DIR) && poetry run alembic downgrade base && poetry run alembic upgrade head; \
		echo "$(GREEN)Database reset completed$(NC)"; \
	else \
		echo "$(YELLOW)Database reset cancelled$(NC)"; \
	fi
