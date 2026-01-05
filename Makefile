.PHONY: run run-all inspect setup

PYTHON = venv/bin/python
TOP_N = 5

run:
	$(PYTHON) -m dog_cat_classifier $(IMG)

run-all:
	$(PYTHON) -m dog_cat_classifier --all --top $(TOP_N) $(IMG)

inspect:
	$(PYTHON) -m dog_cat_classifier.inspect_model

setup:
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt
