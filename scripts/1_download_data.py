import os
import click
import pandas as pd
from ucimlrepo import fetch_ucirepo


@click.command()
@click.option('--dataset-id', '-i', default=17, type=int,
              help='The ID of the dataset to fetch from UCI repository.')
@click.option('--output-file', '-o', type=str,
              default='data/raw/breast_cancer_raw.csv',
              help='The local path/filename where the data will be saved (e.g., data/raw/data.csv).')
def main(dataset_id, output_file):
    """
    Downloads a dataset from the UCI Machine Learning Repository.

    This script fetches the dataset features and targets
    based on the provided UCI ID, concatenates them into a single DataFrame,
    and exports the result to the specifiedlocal path.

    Parameters
    ----------
    dataset_id : int
        The unique identifier for the dataset in the UCI repository.
        Default is 17 (Breast Cancer Wisconsin Diagnostic).
    output_file : str
        The full path (including filename)
        where the CSV file should be written.
        If the parent directories do not exist, they will be created.

    Returns
    -------
    None
        The function saves a file to disk and prints a confirmation message.
    """

    click.echo(f"Downloading dataset with ID: {dataset_id}...")

    try:
        # Fetch data from UCI repo
        raw_data = fetch_ucirepo(id=dataset_id)
  
        # Extract features and targets
        raw_X = raw_data.data.features
        raw_y = raw_data.data.targets
    
        # Combine into one DataFrame
        raw_df = pd.concat([raw_X, raw_y], axis=1)
        
        # Create directory if it doesn't exist (Handle Path issues)
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            click.echo(f"Created directory: {output_dir}")

        # Save to CSV
        raw_df.to_csv(output_file, index=False)
        click.echo(click.style(f"Successfully saved data to {output_file}", fg='green'))

    except Exception as e:
        click.echo(click.style(f"Error occurred: {e}", fg='red'))


if __name__ == '__main__':
    main()