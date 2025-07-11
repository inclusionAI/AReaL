.PHONY: docs

docs:
	docker compose down
	cd docs && make html
	docker compose up --build