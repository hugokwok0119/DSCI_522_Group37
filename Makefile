.PHONY: all clean help setup cl env build run up stop

.DEFAULT_GOAL := help

# ----------------------------------------------------------------------------
# Analysis Pipeline Targets 
# ----------------------------------------------------------------------------

all: reports/breast_cancer_predictor_report.pdf reports/breast_cancer_predictor_report.html

data/raw/breast_cancer_raw.csv: scripts/1_download_data.py
	python scripts/1_download_data.py

data/processed/breast_cancer_cleaned.csv \
data/processed/clean_train.csv \
data/processed/clean_test.csv: scripts/2_clean_data.py data/raw/breast_cancer_raw.csv
	python scripts/2_clean_data.py

results/images/corr_chart.png \
results/images/dist_chart.png \
results/images/pair_chart.png: scripts/3_eda.py data/processed/clean_train.csv
	python scripts/3_eda.py

results/models/ct.pickle \
results/images/svm_heatmap.png \
results/images/con_mat_heatmap.png: scripts/4_model.py data/processed/clean_train.csv data/processed/clean_test.csv
	python scripts/4_model.py

reports/breast_cancer_predictor_report.html: reports/breast_cancer_predictor_report.qmd \
                                              reports/references.bib \
                                              results/images/corr_chart.png \
                                              results/images/dist_chart.png \
                                              results/images/pair_chart.png \
                                              results/images/svm_heatmap.png \
                                              results/images/con_mat_heatmap.png \
                                              results/models/ct.pickle
	quarto render reports/breast_cancer_predictor_report.qmd --to html

reports/breast_cancer_predictor_report.pdf: reports/breast_cancer_predictor_report.qmd \
                                             reports/references.bib \
                                             results/images/corr_chart.png \
                                             results/images/dist_chart.png \
                                             results/images/pair_chart.png \
                                             results/images/svm_heatmap.png \
                                             results/images/con_mat_heatmap.png \
                                             results/models/ct.pickle
	quarto render reports/breast_cancer_predictor_report.qmd --to typst

clean:
	rm -rf data/raw/*
	rm -f data/processed/*.csv
	rm -f results/models/*.pickle
	rm -f results/images/*.png
	rm -f reports/breast_cancer_predictor_report.html \
	      reports/breast_cancer_predictor_report.pdf

# ----------------------------------------------------------------------------
# Environment & Utilities Targets 
# ----------------------------------------------------------------------------

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## runs the targets: cl, env, build
	make cl
	make env
	make build

cl: ## create conda lock for multiple platforms
	conda-lock lock \
		--file environment.yml \
		-p linux-64 \
		-p linux-aarch64 \
		-p osx-64 \
		-p osx-arm64 \
		-p win-64

env: ## remove previous and create environment from lock file
	conda env remove -n dockerlock || true
	conda-lock install -n dockerlock conda-lock.yml

build: ## build the docker image from the Dockerfile
	docker build -t dockerlock --file Dockerfile .

run: ## alias for the up target
	make up

up: ## stop and start docker-compose services
	make stop
	docker-compose up -d

stop: ## stop docker-compose services
	docker-compose stop