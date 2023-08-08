build:
	@rm -rf dist
	@.venv/bin/python3 -m build

upload: build
	@.venv/bin/python3 -m twine upload dist/*

docs:
	@cd docs && make html