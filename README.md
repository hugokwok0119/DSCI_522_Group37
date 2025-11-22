# Breast Cancer Predictor

  - Authors:  Sameel Syed, Hoi Hin Kwok, Lavanya Gupta & Yusheng Li

This project is made as the Group Project for group 37 as the data analysis project for 
DSCI 522 (Data Science Workflows); as the part of requirements in a course in the Master 
of Data Science program at the University 
of British Columbia.

## About

Here we attempt to build a Breast Cancer classification model using the SVC 
algorithm which can use breast cancer tumour image 
measurements to predict whether a newly discovered breast cancer tumour 
is benign (i.e., is not harmful and does not require treatment) or 
malignant (i.e., is harmful and requires treatment intervention). 
Our final classifier performed fairly well on an unseen test data set, 
with an overall accuracy calculated to be 0.99. On the 114 test data cases, 
it correctly predicted 113. 
It incorrectly predicted 1 case, predicting that a tumour is benign 
when in fact it is actually malignant. 
These kind of incorrect predictions could cause the patient 
to miss out on necessary treatment, 
and as such we recommend further research to improve the model 
before it is ready to be put into production in the clinic.

The data set that was used in this project is of digitized breast cancer
image features created by Dr. William H. Wolberg, W. Nick Street, and
Olvi L. Mangasarian at the University of Wisconsin, Madison (Street,
Wolberg, and Mangasarian 1993). It was sourced from the UCI Machine
Learning Repository (Dua and Graff 2017) and can be found
[here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+\(Diagnostic\)),
specifically [this
file](http://mlr.cs.umass.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data).
Each row in the data set represents summary statistics from measurements
of an image of a tumour sample, including the diagnosis (benign or
malignant) and several other measurements (e.g., nucleus texture,
perimeter, area, etc.). Diagnosis for each image was conducted by
physicians.

## Report

The final report can be found
[here](https://github.com/hugokwok0119/DSCI_522_Group37/blob/main/notebooks/breast_cancer_predictor_report.html).

## Usage

First time running the project,
run the following from the root of this repository:

``` bash
conda-lock install --name breast-cancer-predictor conda-lock.yml
```

To run the analysis,
run the following from the root of this repository:

``` bash
jupyter lab 
```

Open `notebooks/breast_cancer_predict_report.ipynb` in Jupyter Lab
and under Switch/Select Kernel choose 
"Python [conda env:MDS_Group37]".

Next, under the "Kernel" menu click "Restart Kernel and Run All Cells...".

## Dependencies

- `conda` (version 23.9.0 or higher)
- `conda-lock` (version 2.5.7 or higher)
- `jupyterlab` (version 4.0.0 or higher)
- `nb_conda_kernels` (version 2.3.1 or higher)
- Python and packages listed in [`environment.yml`](environment.yml)

## License

The Breast Cancer Predictor report contained herein are licensed under the
[Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
See [the license file](LICENSE.md) for more information. . If
re-using/re-mixing please provide attribution and link to this webpage.
The software code contained within this repository is licensed under the
MIT license. See [the license file](LICENSE.md) for more information.

## References

Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.”
University of California, Irvine, School of Information; Computer
Sciences. <http://archive.ics.uci.edu/ml>.

Street, W. Nick, W. H. Wolberg, and O. L. Mangasarian. 1993. “Nuclear
feature extraction for breast tumor diagnosis.” In *Biomedical Image
Processing and Biomedical Visualization*, edited by Raj S. Acharya and
Dmitry B. Goldgof, 1905:861–70. International Society for Optics;
Photonics; SPIE. <https://doi.org/10.1117/12.148698>.
