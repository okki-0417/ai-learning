.PHONY: setup train resume generate

PYTHON = venv/bin/python
TEXT = data/input.txt
EPOCHS = 10
START = The
LENGTH = 500
MODEL = model.pth

setup:
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt

train:
	$(PYTHON) -m transformer $(TEXT) --epochs $(EPOCHS) --start "$(START)" --length $(LENGTH)

resume:
	$(PYTHON) -m transformer $(TEXT) --resume $(MODEL) --epochs $(EPOCHS) --start "$(START)" --length $(LENGTH)

generate:
	$(PYTHON) -m transformer --load $(MODEL) --start "$(START)" --length $(LENGTH)
