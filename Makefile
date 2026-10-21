PYTHON := .venv/bin/python
PYTHONPATH := src
VENV_ACTIVATE := .venv/bin/activate
SHELL := /bin/bash
EPOCHS ?= 3
CHECKPOINT ?= artifacts/checkpoint_latest.npz
HISTORY ?= artifacts/history_latest.npz
USERNAME ?=
DATA ?= data/processed/split.npz
OPTIMIZER ?= sgd
ACTIVATION ?= relu
HIDDEN_SIZES ?= 256

.PHONY: help test demo download prepare train plot evaluate api web shell activate

help:
	@printf '%s\n' \
	  'ChessMoveNet commands:' \
	  '  make test                         Run automated tests' \
	  '  make demo                         Run the small gradient-descent demo' \
	  '  make download                     Download public games from Chess.com' \
	  '  make prepare USERNAME=name         Build a dataset from data/raw/name.pgn' \
	  '  make train                        Train for 3 epochs (change with EPOCHS=15)' \
	  '  make train EPOCHS=15 OPTIMIZER=adam ACTIVATION=tanh HIDDEN_SIZES=256,128' \
	  '                                   Custom optimizer / activation / hidden sizes' \
	  '  make plot                         Create loss and accuracy charts' \
	  '  make evaluate                     Evaluate the latest checkpoint on the test set' \
	  '  make api                          Start the FastAPI server' \
	  '  make web                          Start the React frontend' \
	  '  make shell                        Open an interactive shell with the venv activated' \
	  '  make activate                      Print the command to activate the venv manually'

activate:
	@echo "source $(VENV_ACTIVATE) && export PYTHONPATH=$(PYTHONPATH)"

shell:
	bash --rcfile <(echo 'source $(VENV_ACTIVATE); export PYTHONPATH=$(PYTHONPATH); PS1="(chessmovenet) \u@\h:\w\$ "') -i

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/ -v

demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/three_layer_gradient_descent.py

download:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m chessmovenet.downloader

prepare:
	@[ -n "$(USERNAME)" ] || (echo "Usage: make prepare USERNAME=your_chesscom_name"; exit 2)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/prepare_dataset.py \
		--pgn data/raw/$(USERNAME).pgn --username $(USERNAME) \
		--output $(DATA) --overwrite

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m chessmovenet.train \
		--epochs $(EPOCHS) --batch-size 128 --lr 0.001 --seed 0 \
		--optimizer $(OPTIMIZER) --activation $(ACTIVATION) --hidden-sizes $(HIDDEN_SIZES) \
		--checkpoint $(CHECKPOINT) --history $(HISTORY)

plot:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/plot_history.py $(HISTORY)

evaluate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/evaluate_checkpoint.py \
		--checkpoint $(CHECKPOINT)

api:
	PYTHONPATH=$(PYTHONPATH) .venv/bin/uvicorn backend.app:app --reload

web:
	cd frontend && npm run dev
