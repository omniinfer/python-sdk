build:
	@.venv/bin/python3 -m build

upload: build
	@.venv/bin/python3 -m twine upload dist/*
	@rm -rf dist

docs:
	@cd docs && make html