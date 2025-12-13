import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing import preprocess_data

def clean_and_split_data(df, output_file, output_dir=None):
    """
    Core logic function to clean, split, and trigger preprocessing.
    Refactored for testing purposes.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    # Validation: Ensure no "Unnamed" index column exists
    if any(df.columns.str.contains("Unnamed")):
        raise AssertionError("Error: Unnamed index column detected! Please check if the raw data was saved with index=False.")

    # Clean Column Names
    clean_columns = []
    for col in df.columns:
        if col.endswith('1'):
            clean_name = col[:-1] + '_mean'
        elif col.endswith('2'):
            clean_name = col[:-1] + '_se'
        elif col.endswith('3'):
            clean_name = col[:-1] + '_max'
        else:
            clean_name = col
        clean_columns.append(clean_name)
    
    df.columns = clean_columns
    
    # Clean Target Column ('Diagnosis')
    if 'Diagnosis' in df.columns:
        df['Diagnosis'] = df['Diagnosis'].map({
            'M': 'Malignant', 'B': 'Benign'})
    
    # Create directory if it doesn't exist
    target_dir = output_dir if output_dir else os.path.dirname(output_file)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Save cleaned full data
    df.to_csv(output_file, index=False)
    
    # Split Data
    clean_train_df, clean_test_df = train_test_split(
        df, test_size=0.2, random_state=123, stratify=df.get('Diagnosis')
    )
    
    # Save splits
    # Note: In a real scenario you might want these paths dynamic, keeping hardcoded to match original script style
    clean_train_df.to_csv('data/processed/clean_train.csv', index=False)
    clean_test_df.to_csv('data/processed/clean_test.csv', index=False)
    
    # Define features
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

    # Call Preprocessing
    preprocess_data(
        train_df=clean_train_df,
        test_df=clean_test_df,
        numeric_features=numeric_feats,
        drop_features=drop_feats,
        output_pickle_path="results/models/preprocessor.pickle",
        save_transformed_csv=True,
        train_output_path='data/processed/scaled_train.csv',
        test_output_path='data/processed/scaled_test.csv'
    )
    
    return clean_train_df, clean_test_df

@click.command()
@click.option('--input-file', '-i',
              default='data/raw/breast_cancer_raw.csv',
              help='Path to the raw input CSV file.')
@click.option('--output-file', '-o',
              default='data/processed/breast_cancer_cleaned.csv',
              type=str,
              help='Path/filename where the cleaned data will be saved.')
def main(input_file, output_file):
    click.echo(f"Reading data from {input_file}...")

    try:
        # Read the data
        df = pd.read_csv(input_file)
        
        # Call the core logic function
        clean_and_split_data(df, output_file)
        
        click.echo(click.style(f"Successfully saved processed data to {output_file}", fg='green'))
        click.echo(click.style(f"Successfully saved scaled data and preprocessor", fg='green'))

    except Exception as e:
        click.echo(click.style(f"Processing failed: {e}", fg='red'))

if __name__ == '__main__':
    main()