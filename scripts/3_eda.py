import click
import pandas as pd
import sys
import os

# Add project root to sys.path to import src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.eda import perform_eda

@click.command()
@click.option('--input-file', '-i', 
              default='data/processed/breast_cancer_cleaned.csv', 
              help='Path to the cleaned input CSV file.')
@click.option('--output-dir', '-o', 
              default='results', 
              help='Directory where the EDA artifacts will be saved.')
def main(input_file, output_dir):
    """
    Driver script for Exploratory Data Analysis.
    """
    click.echo(f"Loading data from {input_file}...")
    
    try:
        df = pd.read_csv(input_file)
        
        # --- Special Logic for Breast Cancer Dataset ---
        # Define specific columns for the pair plot to avoid overcrowding
        # This keeps the 'src' function clean and reusable.
        target = 'Diagnosis'
        cols_mean = [c for c in df.columns if '_mean' in c]
        
        # Ensure target is in the subset for plotting
        if target in df.columns:
            pair_plot_subset = cols_mean + [target]
        else:
            pair_plot_subset = cols_mean

        # Call the modular function
        perform_eda(
            df=df, 
            output_dir=output_dir, 
            target_col=target, 
            pair_plot_cols=pair_plot_subset
        )
        
        click.echo(click.style("EDA script completed successfully!", fg='green'))

    except Exception as e:
        click.echo(click.style(f"Error in EDA script: {e}", fg='red'))
        sys.exit(1)

if __name__ == '__main__':
    main()