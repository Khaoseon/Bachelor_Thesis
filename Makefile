PYTHON ?= python
PYTHON_BOOTSTRAP ?= python3
LATEXMK ?= latexmk

CREATE_ENV ?= false
VENV ?= .venv

DATASET ?= cifar10
MODEL_NAME ?= vgg_small_bn
MODEL_TAG := $(MODEL_NAME)
DATASET_TAG := $(DATASET)

EPOCHS ?= 100
BATCH_SIZE ?= 256
NUM_WORKERS ?= 12
SEED ?= 42

OPTIMIZER ?= sgd
LR ?= 0.1
MOMENTUM ?= 0.9
WEIGHT_DECAY ?= 5e-4
SCHEDULER ?= cosine

EPSILON ?= 10/255
HEATMAP_BATCH_SIZE ?= 4096
EVAL_BATCH_SIZE ?= $(BATCH_SIZE)
RANDOM_TRIALS ?= 10

CHECKPOINT_DIR ?= outputs/models
RUN_DIR ?= outputs/runs
FIGURE_DIR ?= outputs/figures

VGG_CHECKPOINT ?= $(CHECKPOINT_DIR)/$(MODEL_TAG)_$(DATASET_TAG)_final.pth
HEATMAP_CHECKPOINT ?= $(VGG_CHECKPOINT)
HEATMAP_OUTPUT_DIR ?= $(RUN_DIR)/fourier_$(MODEL_TAG)_$(DATASET_TAG)_eps10_255
RANDOM_CHECKPOINT ?= $(VGG_CHECKPOINT)
RANDOM_OUTPUT_DIR ?= $(RUN_DIR)/random_noise_$(MODEL_TAG)_$(DATASET_TAG)_eps10_255
SFA_SWEEP_CHECKPOINT ?= $(VGG_CHECKPOINT)
SFA_SWEEP_OUTPUT_DIR ?= $(RUN_DIR)/sfa_epsilon_sweep_$(MODEL_TAG)_$(DATASET_TAG)
SFA_SWEEP_EPSILONS ?= 1/255 2/255 3/255 5/255 10/255
SFA_FREQUENCY_I ?= 16
SFA_FREQUENCY_J ?= 24

.PHONY: help figures check-figure-inputs clean-figures report pdf report-from-scratch train-vgg heatmap-vgg random-vgg sfa-eps-sweep-vgg clean-pyc

help:
	@printf "Commandes principales:\n"
	@printf "  make figures              Genere toutes les figures finales dans outputs/figures.\n"
	@printf "  make clean-figures        Supprime uniquement outputs/figures.\n"
	@printf "  make report | make pdf    Genere les figures puis compile report/main.tex.\n"
	@printf "  make report-from-scratch  Verifie les artefacts, genere les figures, compile le PDF.\n"
	@printf "\nArtefacts experimentaux utiles en cas de clone propre:\n"
	@printf "  make train-vgg            Entraine le checkpoint VGGSmallBN final.\n"
	@printf "  make heatmap-vgg          Genere la heatmap Fourier finale.\n"
	@printf "  make random-vgg           Genere la baseline bruit aleatoire.\n"
	@printf "  make sfa-eps-sweep-vgg    Genere le sweep epsilon SFA.\n"
	@printf "\nVariables:\n"
	@printf "  PYTHON=/chemin/python     Interpreteur Python a utiliser.\n"
	@printf "  CREATE_ENV=true|false     Cree .venv avant report-from-scratch si true.\n"
	@printf "  DATASET=cifar10|cifar100  Dataset pour les cibles d'artefacts.\n"
	@printf "  FIGURE_DIR=outputs/figures Dossier de sortie des figures finales.\n"

figures:
	$(PYTHON) scripts/build_report_figures.py --figure-dir $(FIGURE_DIR)

check-figure-inputs:
	$(PYTHON) scripts/build_report_figures.py --check-inputs --figure-dir $(FIGURE_DIR)

clean-figures:
	rm -rf $(FIGURE_DIR)

report pdf: figures
	$(LATEXMK) -pdf -cd report/main.tex

report-from-scratch:
	@if [ "$(CREATE_ENV)" = "true" ]; then \
		if [ ! -x "$(VENV)/bin/python" ]; then \
			$(PYTHON_BOOTSTRAP) -m venv $(VENV); \
		fi; \
		$(VENV)/bin/python -m pip install --upgrade pip setuptools wheel; \
		$(VENV)/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1; \
		$(VENV)/bin/python -m pip install numpy scipy pandas matplotlib scikit-learn pillow pyyaml psutil gitpython wandb; \
		$(MAKE) report-from-scratch CREATE_ENV=false PYTHON=$(VENV)/bin/python FIGURE_DIR=$(FIGURE_DIR) LATEXMK=$(LATEXMK); \
	elif [ "$(CREATE_ENV)" = "false" ]; then \
		$(MAKE) check-figure-inputs PYTHON=$(PYTHON) FIGURE_DIR=$(FIGURE_DIR); \
		$(MAKE) figures PYTHON=$(PYTHON) FIGURE_DIR=$(FIGURE_DIR); \
		$(LATEXMK) -pdf -cd report/main.tex; \
	else \
		printf "CREATE_ENV doit valoir true ou false. Exemple: make report-from-scratch CREATE_ENV=false\n"; \
		exit 2; \
	fi

train-vgg:
	$(PYTHON) -m src.train \
		--model-name $(MODEL_NAME) \
		--dataset $(DATASET) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--num-workers $(NUM_WORKERS) \
		--optimizer $(OPTIMIZER) \
		--lr $(LR) \
		--momentum $(MOMENTUM) \
		--weight-decay $(WEIGHT_DECAY) \
		--scheduler $(SCHEDULER) \
		--seed $(SEED)

heatmap-vgg:
	$(PYTHON) scripts/generate_fourier_heatmap.py \
		--checkpoint $(HEATMAP_CHECKPOINT) \
		--dataset $(DATASET) \
		--epsilon $(EPSILON) \
		--batch-size $(HEATMAP_BATCH_SIZE) \
		--num-workers $(NUM_WORKERS) \
		--output-dir $(HEATMAP_OUTPUT_DIR)

random-vgg:
	$(PYTHON) scripts/evaluate_random_noise_baseline.py \
		--checkpoint $(RANDOM_CHECKPOINT) \
		--dataset $(DATASET) \
		--epsilon $(EPSILON) \
		--batch-size $(EVAL_BATCH_SIZE) \
		--num-workers $(NUM_WORKERS) \
		--device auto \
		--num-trials $(RANDOM_TRIALS) \
		--seed $(SEED) \
		--output-dir $(RANDOM_OUTPUT_DIR)

sfa-eps-sweep-vgg:
	$(PYTHON) scripts/evaluate_sfa_epsilon_sweep.py \
		--checkpoint $(SFA_SWEEP_CHECKPOINT) \
		--dataset $(DATASET) \
		--frequency-i $(SFA_FREQUENCY_I) \
		--frequency-j $(SFA_FREQUENCY_J) \
		--epsilons $(SFA_SWEEP_EPSILONS) \
		--batch-size $(EVAL_BATCH_SIZE) \
		--num-workers $(NUM_WORKERS) \
		--device auto \
		--output-dir $(SFA_SWEEP_OUTPUT_DIR) \
		--seed $(SEED)

clean-pyc:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
