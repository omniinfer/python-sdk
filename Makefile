build:
	@python -m build

upload: build
	@python3 -m twine upload dist/*

docs:
	@cd docs && make html
