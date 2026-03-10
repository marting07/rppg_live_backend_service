PYTHON := python3
VENV_DIR := .venv
VENV_PY := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

.PHONY: venv install install-replay api-run test live-evaluate live-aggregate

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip

install: venv
	$(VENV_PIP) install -r requirements.txt

install-replay: install
	$(VENV_PIP) install -r requirements-replay.txt

api-run:
	$(VENV_PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8008

test:
	$(VENV_PY) -m unittest discover -s tests -p "test_*.py"

live-evaluate:
	$(VENV_PY) scripts/live_replay_evaluate.py --manifest "$(MANIFEST)" $(if $(OUT),--out "$(OUT)",)

live-aggregate:
	$(VENV_PY) scripts/aggregate_live_liveness.py --input "$(INPUT)" $(if $(OUT),--out "$(OUT)",)
