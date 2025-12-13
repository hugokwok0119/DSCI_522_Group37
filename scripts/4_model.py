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
import pickle
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
    "--train-file", "-tr",
    default="data/processed/scaled_train.csv",  # CORRECTED
    type=click.Path(exists=True),
    help="Path to scaled training CSV."
)
@click.option(
    "--test-file", "-te",
    default="data/processed/scaled_test.csv",  # CORRECTED
    type=click.Path(exists=True),
    help="Path to scaled test CSV."
)
@click.option(
    "--output-dir", "-o",
    default="results",
    type=str,
    help="Directory where model results and images will be saved."
)
def main(train_file, test_file, output_dir):
    click.echo(f"Loading scaled training data from: {train_file}")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Prepare output directories
    out_dir = Path(output_dir)
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Extract features and target
    # The preprocessing passes through 'Diagnosis' column
    target_cols = [c for c in train_df.columns if 'Diagnosis' in c or 'remainder' in c]
    
    if not target_cols:
        raise ValueError("Cannot find Diagnosis column in scaled data")
    
    target_col = target_cols[0]
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    click.echo(f"Data shape: X_train={X_train.shape}, X_test={X_test.shape}")

    svc = SVC(kernel='rbf')

    param_grid = {
        "gamma": [0.001, 0.01, 0.1, 1.0, 10, 100],
        "C": [0.001, 0.01, 0.1, 1.0, 10, 100]
    }

    click.echo("Starting GridSearchCV on pre-scaled data")
    gs = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        cv=15,
        n_jobs=-1,
        return_train_score=True
    )

    gs.fit(X_train, y_train)
    click.echo(click.style("GridSearchCV complete.", fg="green"))

    # Save results
    results = pd.DataFrame(gs.cv_results_)
    results_path = out_dir / "svm_grid_results.csv"
    results.to_csv(results_path, index=False)
    click.echo(f"Saved full cv results to {results_path}")

    best_performing = results[['param_C', 'param_gamma', 'mean_test_score']].sort_values(
        by='mean_test_score', ascending=False
    ).head(10)
    best_path = out_dir / "svm_top10.csv"
    best_performing.to_csv(best_path, index=False)
    click.echo(f"Saved top-10 results to {best_path}")

    # Heatmap
    heatmap_data = results[['param_C', 'param_gamma', 'mean_test_score']].copy()
    heatmap_data['C'] = heatmap_data['param_C'].astype(str)
    heatmap_data['gamma'] = heatmap_data['param_gamma'].astype(str)

    heatmap = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X('gamma:N', title='gamma'),
        y=alt.Y('C:N', title='C'),
        color=alt.Color('mean_test_score:Q', title='Mean Test Score', scale=alt.Scale(scheme='viridis')),
        tooltip=['C', 'gamma', 'mean_test_score']
    ).properties(
        width=400,
        height=400,
        title='SVM GridSearchCV Mean Test Scores'
    )

    svm_heatmap_path = images_dir / "svm_heatmap.png"
    heatmap.save(str(svm_heatmap_path))
    click.echo(f"Saved SVM heatmap to {svm_heatmap_path}")

    # Evaluate on test set
    y_pred = gs.predict(X_test)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = out_dir / "classification_report.csv"
    report_df.to_csv(report_path)
    click.echo(f"Saved classification report to {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=gs.classes_)
    cm_df = pd.DataFrame(cm, index=gs.classes_, columns=gs.classes_)
    cm_path = out_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    click.echo(f"Saved confusion matrix to {cm_path}")

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
        title='Confusion Matrix'
    )

    cm_text = alt.Chart(cm_melted).mark_text(color='white', fontSize=14).encode(
        x='Predicted:N',
        y='Actual:N',
        text='Count:Q'
    )

    cm_combined = (cm_heatmap + cm_text)
    cm_heatmap_path = images_dir / "con_mat_heatmap.png"
    cm_combined.save(str(cm_heatmap_path))
    click.echo(f"Saved confusion matrix heatmap to {cm_heatmap_path}")

    click.echo(click.style("Modeling completed successfully!", fg="green"))


if __name__ == "__main__":
    main()