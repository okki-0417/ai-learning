.PHONY: run inspect setup

run:
	python classify_dog_cat.py $(IMG)

inspect:
	python inspect_model.py

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
