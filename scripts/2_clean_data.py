import os
import click
import pandas as pd

@click.command()
@click.option('--input-file', '-i',
              default='data/raw/breast_cancer_raw.csv',
              type=click.Path(exists=True),
              help='Path to the raw input CSV file.')
@click.option('--output-file', '-o',
              default='data/processed/breast_cancer_cleaned.csv',
              type=str,
              help='Path/filename where the cleaned data will be saved.')
def main(input_file, output_file):
    """
    Reads raw data, performs cleaning and preprocessing, and saves the result.

    This script loads the raw CSV, checks for formatting issues ,
    renames columns to be more descriptive.
    Converting suffix 1/2/3 to _mean/_se/_max,
    and maps the target variable 'Diagnosis' to full labels.

    Parameters
    ----------
    input_file : str
        The path to the raw data file.
        Default: data/raw/breast_cancer_raw.csv
    output_file : str
        The path where the processed data should be saved.
        Default: data/processed/breast_cancer_cleaned.csv
    """
    
    click.echo(f"Reading data from {input_file}...")

    try:
        # Read the data
        df = pd.read_csv(input_file)
        
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
            click.echo("Target column 'Diagnosis' mapped to labels.")
        else:
            click.echo(click.style("Warning: 'Diagnosis' column not found.", fg='yellow'))

        # Save the processed data
        # Ensure output directory exists (data/processed/)
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            click.echo(f"Created directory: {output_dir}")
        
        df.to_csv(output_file, index=False)
        click.echo(click.style(f"Successfully saved processed data to {output_file}", fg='green'))

    except Exception as e:
        click.echo(click.style(f"Processing failed: {e}", fg='red'))

if __name__ == '__main__':
    main()