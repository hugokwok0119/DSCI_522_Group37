import os
import io
import click
import pandas as pd
import altair as alt
import altair_ally as aly
import warnings  # To suppress any warnings during EDA

alt.data_transformers.enable('vegafusion')

warnings.filterwarnings('ignore', category=DeprecationWarning) 
warnings.filterwarnings('ignore', module='altair')

@click.command()
@click.option('--input-file', '-i', 
              default='data/processed/breast_cancer_cleaned.csv', 
              type=click.Path(exists=True),
              help='Path to the cleaned input CSV file.')
@click.option('--output-dir', '-o', 
              default='results', 
              type=str,
              help='Directory where the EDA artifacts (tables and images) will be saved.')
def main(input_file, output_dir):
    """
    Performs Exploratory Data Analysis (EDA) on the cleaned dataset.

    This script generates summary tables (info, describe) and visualizations
    (correlation, pair plots, distribution) using Altair and Altair Ally.

    Parameters
    ----------
    input_file : str
        Path to the cleaned data (e.g., data/processed/breast_cancer_cleaned.csv).
    output_dir : str
        Directory to save results. Images will be saved in a sub-directory 'images'.
    """
    
    click.echo(f"Starting EDA on {input_file}...")

    try:
        # 1. Ensure Output Directories Exist
        images_dir = os.path.join(output_dir, 'images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
            click.echo(f"Created output directories at {output_dir}")

        # 2. Load Data
        df = pd.read_csv(input_file)

        # -------------------------------------------------------
        # Part A: Textual Summaries (Tables)
        # -------------------------------------------------------
        
        # Output 1: DataFrame Info (Data types, non-null counts)
        # buffer capture is needed because df.info() prints to stdout
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        info_path = os.path.join(output_dir, 'eda_info.txt')
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(info_str)
        click.echo(f"Saved data info to {info_path}")

        # Output 2: Descriptive Statistics (Mean, std, min, max)
        describe_path = os.path.join(output_dir, 'eda_summary.csv')
        df.describe(include='all').to_csv(describe_path)
        click.echo(f"Saved descriptive statistics to {describe_path}")

        # -------------------------------------------------------
        # Part B: Visualizations (Altair & Altair Ally)
        # -------------------------------------------------------
        click.echo("Generating visualizations...")

        # Chart 1: Multicollinearity (Correlation)
        corr_chart = aly.corr(df)
        corr_chart.save(os.path.join(images_dir, 'corr_chart.png'))
        corr_chart.save(os.path.join(images_dir, 'corr_chart.svg'))
        click.echo(" - Saved Correlation chart")

        # Chart 2: Pair Plot (Only Mean columns + Target)
        # Filter columns as requested
        cols_mean = [c for c in df.columns if '_mean' in c] + ['Diagnosis']
        
        # Safety check: ensure columns exist before plotting
        existing_cols = [c for c in cols_mean if c in df.columns]
        
        pair_chart = aly.pair(df[existing_cols], color='Diagnosis:N')
        pair_chart.save(os.path.join(images_dir, 'pair_chart.png'))
        pair_chart.save(os.path.join(images_dir, 'pair_chart.svg'))
        click.echo(" - Saved Pair plot")

        # Chart 3: Distribution Plot
        dist_chart = aly.dist(df, color='Diagnosis')
        dist_chart.save(os.path.join(images_dir, 'dist_chart.png'))
        dist_chart.save(os.path.join(images_dir, 'dist_chart.svg'))
        click.echo(" - Saved Distribution chart")

        click.echo(click.style(f"EDA Analysis complete. Results saved in '{output_dir}/'", fg='green'))

    except Exception as e:
        click.echo(click.style(f"EDA failed: {e}", fg='red'))

if __name__ == '__main__':
    main()