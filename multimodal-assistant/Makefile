.PHONY: serve fmt test test-cov test-watch train-image train-text install dev-install clean help

help:
	@echo "Available commands:"
	@echo "  serve        - Start the development server"
	@echo "  fmt          - Format code with black, ruff, isort"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage report"
	@echo "  test-watch   - Run tests in watch mode"
	@echo "  train-image  - Train image classifier"
	@echo "  train-text   - Train text classifier"
	@echo "  install      - Install dependencies"
	@echo "  dev-install  - Install dev dependencies and pre-commit"
	@echo "  clean        - Clean cache files"

serve:
	uvicorn app.server:app --reload --port 8000 --host 0.0.0.0

fmt:
	ruff check --fix . || true
	black .
	isort .

test:
	pytest -v --tb=short

test-cov:
	pytest --cov=app --cov-report=html --cov-report=term-missing

test-watch:
	pytest-watch --runner "pytest -v"

install:
	pip install -r requirements.txt

dev-install: install
	pre-commit install
	@echo "Development environment setup complete!"

train-image:
	python -m train.lightning.train_image_classifier --config train/configs/image_classifier.yaml

train-text:
	python -m train.lightning.train_text_classifier --config train/configs/text_classifier.yaml

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "Cache files cleaned!"
