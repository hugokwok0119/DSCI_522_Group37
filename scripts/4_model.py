"""
4_model.py

Reads cleaned data, trains an SVM (with preprocessing in a pipeline),
performs GridSearchCV, and writes output to results and results/images.

"""

import os
from pathlib import Path
import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import altair as alt

alt.data_transformers.enable('vegafusion')

@click.command()
@click.option(
    "--input-file", "-i",
    default="data/processed/breast_cancer_cleaned.csv",
    type=click.Path(exists=True),
    help="Path to cleaned CSV produced by script 2."
)
@click.option(
    "--output-dir", "-o",
    default="results",
    type=str,
    help="Directory where model results and images will be saved."
)
@click.option(
    "--test-size", "-t",
    default=0.2,
    type=float,
    help="Proportion of data to hold out for testing."
)
@click.option(
    "--random-state", "-r",
    default=123,
    type=int,
    help="Random seed for reproducibility."
)
def main(input_file, output_dir, test_size, random_state):
    click.echo(f"Loading cleaned data from: {input_file}")
    df = pd.read_csv(input_file)

    # Prepare output directories
    out_dir = Path(output_dir)
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Features & target
    if "Diagnosis" not in df.columns:
        raise ValueError("Input data must contain 'Diagnosis' column.")

    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    click.echo(f"Data shape: X={X.shape}, y={y.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    click.echo(f"Train/Test split: {X_train.shape[0]} train, {X_test.shape[0]} test")

    numeric_feats = [
        'radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
        'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'smoothness_se', 'compactness_se', 'concavity_se',
        'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_max', 'texture_max', 'smoothness_max', 'compactness_max',
        'concavity_max', 'concave_points_max', 'symmetry_max', 'fractal_dimension_max'
    ]

    drop_feats = [
        'perimeter_mean', 'area_mean',
        'perimeter_se', 'area_se',
        'texture_se', 'smoothness_se', 'symmetry_se',
        'perimeter_max', 'area_max'
    ]

    ct = make_column_transformer(
        (StandardScaler(), numeric_feats),
        ("drop", drop_feats),
        remainder="passthrough" 
    )

    pipe = Pipeline([
        ("preprocess", ct),
        ("svc", SVC())
    ])

    param_grid = {
        "svc__gamma": [0.001, 0.01, 0.1, 1.0, 10, 100],
        "svc__C": [0.001, 0.01, 0.1, 1.0, 10, 100]
    }

    click.echo("Starting GridSearchCV")
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=15,
        n_jobs=-1,
        return_train_score=True
    )

    gs.fit(X_train, y_train)
    click.echo(click.style("GridSearchCV complete.", fg="green"))

    results = pd.DataFrame(gs.cv_results_)
    results_path = out_dir / "svm_grid_results.csv"
    results.to_csv(results_path, index=False)
    click.echo(f"Saved full cv results to {results_path}")

    best_performing = results[['param_svc__C', 'param_svc__gamma', 'mean_test_score']].sort_values(
        by='mean_test_score', ascending=False
    ).head(10)
    best_path = out_dir / "svm_top10.csv"
    best_performing.to_csv(best_path, index=False)
    click.echo(f"Saved top-10 results to {best_path}")

    heatmap_data = results[['param_svc__C', 'param_svc__gamma', 'mean_test_score']].copy()
    heatmap_data['C'] = heatmap_data['param_svc__C'].astype(str)
    heatmap_data['gamma'] = heatmap_data['param_svc__gamma'].astype(str)

    heatmap = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X('gamma:N', title='gamma'),
        y=alt.Y('C:N', title='C'),
        color=alt.Color('mean_test_score:Q', title='mean_test_score', scale=alt.Scale(scheme='viridis')),
        tooltip=['C', 'gamma', 'mean_test_score']
    ).properties(
        width=400,
        height=400,
        title='SVM GridSearchCV Mean Test Scores'
    )

    svm_heatmap_path = images_dir / "svm_heatmap.png"
    heatmap.save(str(svm_heatmap_path))
    click.echo(f"SVM heatmap to {svm_heatmap_path}")

    # Evaluate
    y_pred = gs.predict(X_test)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().drop(columns=["support"], errors="ignore")
    report_path = out_dir / "classification_report.csv"
    report_df.to_csv(report_path)
    click.echo(f"Saved classification report to {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=gs.classes_)
    cm_df = pd.DataFrame(cm, index=gs.classes_, columns=gs.classes_)
    cm_path = out_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    click.echo(f"Saved conf matriz to {cm_path}")

    # Confusion matrix heatmap
    cm_melted = cm_df.reset_index().melt(id_vars='index')
    cm_melted.columns = ['Actual', 'Predicted', 'Count']

    cm_heatmap = alt.Chart(cm_melted).mark_rect().encode(
        x=alt.X('Predicted:N', title='Predicted'),
        y=alt.Y('Actual:N', title='Actual'),
        color=alt.Color('Count:Q', scale=alt.Scale(scheme='viridis'))
    ).properties(
        width=400,
        height=400,
        title='Confusion Matrix Heatmap'
    )

    cm_text = alt.Chart(cm_melted).mark_text(color='white').encode(
        x='Predicted:N',
        y='Actual:N',
        text='Count:Q'
    )

    cm_combined = (cm_heatmap + cm_text)

    con_mat_heatmap_path = images_dir / "con_mat_heatmap.png"
    cm_combined.save(str(con_mat_heatmap_path))
    click.echo(f"Saved conf matrix heatmap to {con_mat_heatmap_path}")

    click.echo(click.style("Completed successfully.", fg="green"))


if __name__ == "__main__":
    main()