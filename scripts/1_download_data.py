import click
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
  
try:
    from src.download_data import download_data
except ImportError as e:
    click.echo(click.style(f"Error importing 'src': {e}", fg='red'))
    sys.exit(1)


@click.command()
@click.option('--dataset-id', '-i', default=17, type=int,
              help='The ID of the dataset to fetch from UCI repository.')
@click.option('--output-file', '-o', type=str,
              default='data/raw/breast_cancer_raw.csv',
              help='The local path/filename where the data will be saved (e.g., data/raw/data.csv).')
def main(dataset_id, output_file):
    """
    Downloads a dataset from the UCI Machine Learning Repository.
    
    This script acts as a CLI wrapper around the 'download_data' function.
    """
    
    click.echo(f"Starting process for Dataset ID: {dataset_id}...")

    try:
        df = download_data(uci_id=dataset_id, output_path=output_file)
        
        if df is not None:
            click.echo(click.style(f"Process completed! Data shape: {df.shape}", fg='green'))
        
    except Exception as e:
        click.echo(click.style(f"Error occurred: {e}", fg='red'))
        sys.exit(1)


if __name__ == '__main__':
    main()