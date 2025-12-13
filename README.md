# Breast Cancer Predictor

**Authors**: Sameel Syed, Hoi Hin Kwok, Lavanya Gupta & Yusheng Li

A reproducible data analysis project investigating breast cancer tumor classification using Support Vector Machines (SVM). This project is part of the DSCI 522 (Data Science Workflows) course in the Master of Data Science program at the University of British Columbia.

## Project Overview

Breast cancer classification has been extensively studied in the machine learning literature; however, reproducibility, transparent feature selection, and clinically motivated error analysis remain ongoing challenges. The Breast Cancer Wisconsin (Diagnostic) dataset was selected due to its widespread use as a benchmark dataset, physician-verified labels, and interpretable feature set derived from real diagnostic imaging. Using this dataset allows our work to be directly comparable to prior studies while focusing on building a fully reproducible, well-documented pipeline that emphasizes clinically relevant evaluation and error analysis.

### The Challenge
Breast cancer diagnosis often relies on the visual interpretation of fine needle aspirate (FNA) images. The core challenge is to accurately distinguish between **benign** (non-harmful) and **malignant** (harmful) tumors based on geometric measurements of cell nuclei. In this medical context, minimizing false negatives is critical, as missing a malignant case can delay necessary life-saving treatment.

### The Solution
We developed a binary classification model using the **Support Vector Machine (SVM)** algorithm with **GridSearchCV** for hyperparameter tuning. The pipeline features a robust data validation framework and automated reproducibility.

## Key Analysis Insights

Our exploratory data analysis (EDA) revealed critical patterns that directly informed our modeling strategy:

* **Outliers as Signals**: Statistical outliers were detected, particularly in malignant samples. Domain investigation confirmed these were not data errors but characteristic biological signals of tumor growth; thus, they were retained to preserve diagnostic information.
* **Multicollinearity Strategy**: We identified near-perfect correlation between `Radius`, `Perimeter`, and `Area`. To improve model stability, we identified these as geometrically redundant and prioritized feature selection.

## Model Performance

The final SVM model achieved strong predictive power on the unseen test set (UCI Machine Learning Repository).

* **Overall Accuracy**: 95.6%
* **Test Set Performance**: Correctly predicted **109 out of 114** cases.
* **Critical Evaluation**: The model produced **1 False Negative** (predicting benign when actual was malignant). While statistically excellent, we discuss the clinical risks of this single error in our full report and suggest future cost-sensitive training methods to mitigate this risk.

## Importance

These results demonstrate that a carefully tuned SVM model can effectively distinguish between benign and malignant tumors using non-invasive image-derived features. The high predictive performance suggests potential utility as a decision-support or preliminary screening tool, where automated assessments could assist clinicians by prioritizing high-risk cases for further evaluation.

## Limitations

Despite strong performance, several limitations must be acknowledged. The dataset is relatively small and derived from a single source, which may limit generalizability to broader patient populations. While accuracy was used as the primary performance metric, clinical reliability is more closely tied to sensitivity (recall), as failing to detect malignant cases carries disproportionate risk. As such, high accuracy alone may overstate real-world safety in a screening context. Additionally, the model was optimized for overall accuracy rather than explicitly minimizing false negatives, an important consideration in clinical deployment. The presence of even a single false negative underscores the need for cost-sensitive learning approaches and further validation on external datasets. These findings also assume that the dataset is representative of broader patient populations, an assumption that may not hold given the controlled nature and limited scope of the available features.

Future work should focus on incorporating clinically weighted loss functions, expanding evaluation across diverse cohorts, and analyzing misclassified cases to improve model robustness and safety for real-world clinical use.

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
├── src/                    # Modular Functions
├── test/                   # Tests for Modular Functions    
├── Dockerfile              # Container definition
├── docker-compose.yml      # Service orchestration
├── Makefile                # Automation commands
├── environment.yml         # Local dependency lock
└── README.md
```
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

1.  **American Cancer Society (2007)**. Breast cancer facts & figures. American Cancer Society.
2.  **PDQ Adult Treatment Editorial Board (2025)**. Breast Cancer Treatment (PDQ®). In *PDQ cancer information summaries [internet]*. National Cancer Institute (US).
3.  **Dua, D. and Graff, C. (2017)**. UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
4.  **Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993)**. Nuclear feature extraction for breast tumor diagnosis. In *Biomedical Image Processing and Biomedical Visualization* (Vol. 1905, pp. 861-870). International Society for Optics and Photonics.
5.  **Canadian Cancer Statistics Advisory Committee (2019)**. Canadian Cancer Statistics 2019. Toronto, ON: Canadian Cancer Society [http://cancer.ca/Canadian-Cancer-Statistics-2019-EN].
6.  **McKinney, W. (2010)**. Data Structures for Statistical Computing in Python. In S. van der Walt & J. Millman (Eds.), *Proceedings of the 9th Python in Science Conference* (pp. 56-61). doi: 10.25080/Majora-92bf1922-00a.
7.  **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., et al. (2011)**. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12(85), 2825-2830 [http://jmlr.org/papers/v12/pedregosa11a.html].
8.  **Harris, C.R., Millman, K.J., Van Der Walt, S.J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Oliphant, T.E., Haberland, M., Reddy, T., et al. (2020)**. Array programming with NumPy. *Nature*, 585(7825), 357-362. doi: 10.1038/s41586-020-2649-2.
9.  **VanderPlas, J., Granger, B., Heer, J., Moritz, D., Wongsuphasawat, K., Satyanarayan, A., Lees, E., Timofeev, I., Welsh, B., & Sievert, S. (2018)**. Altair: Interactive Statistical Visualizations for Python. *Journal of Open Source Software*, 3(32), 1057. doi: 10.21105/joss.01057 [https://doi.org/10.21105/joss.01057].
10.  **Van Rossum, G. and Drake, F.L. (2009)**. Python 3 Reference Manual. Scotts Valley, CA: CreateSpace.
11.  **Docker, Inc. (2024)**. Docker: Lightweight Linux Containers for Consistent Development and Deployment [https://www.docker.com/].
12.  **Cortes, C. and Vapnik, V. (1995)**. Support-vector networks. *Machine Learning*, 20(3), 273-297
