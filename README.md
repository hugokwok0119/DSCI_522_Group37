# Breast Cancer Predictor

**Authors**: Sameel Syed, Hoi Hin Kwok, Lavanya Gupta & Yusheng Li

A reproducible data analysis project investigating breast cancer tumor classification using Support Vector Machines (SVM). This project is part of the DSCI 522 (Data Science Workflows) course in the Master of Data Science program at the University of British Columbia.

## Project Overview

### The Challenge
Breast cancer diagnosis often relies on the visual interpretation of fine needle aspirate (FNA) images. The core challenge is to accurately distinguish between **benign** (non-harmful) and **malignant** (harmful) tumors based on geometric measurements of cell nuclei. In this medical context, minimizing false negatives is critical, as missing a malignant case can delay necessary life-saving treatment.

### The Solution
We developed a binary classification model using the **Support Vector Machine (SVM)** algorithm with **GridSearchCV** for hyperparameter tuning. The pipeline features a robust data validation framework and automated reproducibility.

## Key Analysis Insights

Our exploratory data analysis (EDA) revealed critical patterns that directly informed our modeling strategy:

* **Handling Skewed Data**: We observed that features like `Area` and `Perimeter` spanned vast magnitudes (values > 2000) compared to features like `Smoothness` (< 0.1). We applied **Symmetric Log (Symlog) transformation** to visualize these distributions effectively without losing information from extreme values.
* **Outliers as Signals**: Statistical outliers were detected, particularly in malignant samples. Domain investigation confirmed these were not data errors but characteristic biological signals of tumor growth; thus, they were retained to preserve diagnostic information.
* **Multicollinearity Strategy**: We identified near-perfect correlation between `Radius`, `Perimeter`, and `Area`. To improve model stability, we identified these as geometrically redundant and prioritized feature selection.

## Model Performance

The final SVM model achieved strong predictive power on the unseen test set (UCI Machine Learning Repository).

* **Overall Accuracy**: 99%
* **Test Set Performance**: Correctly predicted **113 out of 114** cases.
* **Critical Evaluation**: The model produced **1 False Negative** (predicting benign when actual was malignant). While statistically excellent, we discuss the clinical risks of this single error in our full report and suggest future cost-sensitive training methods to mitigate this risk.

## Report

The full analysis, including code and visualizations, can be viewed here:
[**Read the Full Analysis Report**](reports/breast_cancer_predictor_report.pdf)

## Project Structure

```text
root/
├── data/
│   ├── processed/          # Cleaned data ready for modelling
│   └── raw/                # Immutable original data
├── notebooks/              # Jupyter notebooks for exploration
├── reports/                # Generated analysis reports
│   ├── breast_cancer_predictor_report.pdf
│   ├── breast_cancer_predictor_report.html
│   └── references.bib
├── results/                # Exported artifacts
├── scripts/                # Source code for the pipeline
│   ├── 1_download_data.py
│   ├── 2_clean_data.py
│   ├── 3_eda.py
│   └── 4_model.py
├── Dockerfile              # Container definition
├── docker-compose.yml      # Service orchestration
├── Makefile                # Automation commands
├── environment.yml         # Local dependency lock
└── README.md

## Usage

To ensure reproducibility, we support two execution methods.

### Method 1: Using Docker (Recommended)

Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running.

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/hugokwok0119/DSCI_522_Group37.git](https://github.com/hugokwok0119/DSCI_522_Group37.git)
    cd DSCI_522_Group37
    ```

2.  **Run the analysis:**
    To execute the entire pipeline (download, clean, analyze, and report) and launch the Jupyter Lab interface:

    ```bash
    docker compose up
    #or
    make up
    ```

3.  **Access Jupyter Lab:**
    Look for a URL in the terminal starting with `http://127.0.0.1:8888/lab?token=...`. Copy and paste this into your browser.

4. **Run Makefile commands inside Jupyter terminal:**
    Open a new terminal in Jupyter Lab and run:

    ```bash
    make all
    ```

    *(To reset the project state, run `make clean`)*

5.  **Clean up:**
    To shut down the container and remove resources:

    ```bash
    docker compose rm
    ```

### Method 2: Local Development

If you prefer to run the project locally, ensure you have `conda` installed.

1.  **Setup Environment:**

    ```bash
    conda env create -f environment.yml
    conda activate MDS_group37
    ```

2.  **Run with Make (Automated):**
    Since a `Makefile` is provided, you can run the entire analysis with one command:

    ```bash
    make all
    ```

    *(To reset the project state, run `make clean`)*

3.  **Run Scripts Manually (Alternative):**
    If you wish to run the steps individually via the terminal:

    Using defalut value:
    ```bash
    # 1. Download Data
    python scripts/1_download_data.py

    # 2. Clean Data
    python scripts/2_clean_data.py

    # 3. Exploratory Data Analysis
    python scripts/3_eda.py

    # 4. Modelling
    python scripts/4_model.py
    ```
   
   Or specifying input/output paths:

    ```bash
    # 1. Download Data
    python scripts/1_download_data.py \
       --dataset-id 17 \
       --output-file data/raw/breast_cancer_raw.csv

    # 2. Clean Data
    python scripts/2_clean_data.py \
       --input-file data/raw/breast_cancer_raw.csv \
       --output-file data/processed/breast_cancer_cleaned.csv

    # 3. Exploratory Data Analysis
    python scripts/3_eda.py \
       --input-file data/processed/breast_cancer_cleaned.csv \
       --output-dir results

    # 4. Modelling
    python scripts/4_model.py \
       --input-file data/processed/breast_cancer_cleaned.csv \
       --output-dir results
    ```

## Data Source

The data used in this project is the **Breast Cancer Wisconsin (Diagnostic) Data Set**.

  * **Source**: UCI Machine Learning Repository
  * **Creators**: Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian (University of Wisconsin, Madison).
  * **Original URL**: [UCI Archive](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+\(Diagnostic\))

## Developer Notes

### Dependencies

  * Python 3.10+ and standard data science libraries (pandas, scikit-learn, altair).
  * See `environment.yml` for the complete list.

### Adding Dependencies

1.  Add the new package to `environment.yml`.
2.  Update the lock file:
    ```bash
    conda-lock -k explicit --file environment.yml -p linux-64
    ```
3.  Rebuild the Docker image locally to verify.

## License

  * **Report & Documentation**: Licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  * **Software Source Code**: Licensed under the [MIT License](https://www.google.com/search?q=LICENSE).

## References

1.  **Dua, D. and Graff, C. (2019)**. UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
2.  **Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993)**. Nuclear feature extraction for breast tumor diagnosis. In *Biomedical Image Processing and Biomedical Visualization* (pp. 861-870). SPIE. doi: 10.1117/12.148698.
