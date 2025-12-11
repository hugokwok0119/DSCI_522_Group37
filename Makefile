.PHONY: all clean

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
	quarto render reports/breast_cancer_predictor_report.qmd --to pdf

clean:
	rm -rf data/raw/*
	rm -f data/processed/*.csv
	rm -f results/models/*.pickle
	rm -f results/images/*.png
	rm -f reports/breast_cancer_predictor_report.html \
	      reports/breast_cancer_predictor_report.pdf