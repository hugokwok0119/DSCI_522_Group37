import os
import pandas as pd
import importlib.util


def download_data(url=None, uci_id=None, output_path=None):
    """
    Downloads data from a specified URL or the UCI Machine Learning Repository, 
    returns a DataFrame, and optionally saves it to a file.

    Parameters
    ----------
    url : str, optional
        The direct URL to the dataset (e.g., a CSV file). 
        Required if `uci_id` is None.
    uci_id : int, optional
        The ID of the dataset in the UCI ML Repository. 
        If provided, the function attempts to use `ucimlrepo` to fetch data.
    output_path : str, optional
        The file path to save the data locally.
        e.g., "data/raw/data.csv". If None, data is not saved to disk.

    Returns
    -------
    pd.DataFrame
        The downloaded data as a pandas DataFrame.

    Raises
    ------
    ValueError
        If neither `url` nor `uci_id` is provided.
    ImportError
        If `uci_id` is provided but `ucimlrepo` is not installed.x q
    Exception
        If the download fails or data cannot be parsed.
    """

    # 1. Parameter Validation
    if url is None and uci_id is None:
        raise ValueError("You must provide either a 'url' or a 'uci_id'.")

    # 2. Fetch Data Logic
    df = None

    # Priority: If uci_id is provided, try to use the library
    if uci_id is not None:
        if _check_ucimlrepo_installed():
            print(f"Download dataset ID {uci_id} from ucimlrepo")
            df = _fetch_from_uci(uci_id)
        else:
            raise ImportError("The 'ucimlrepo' package is not installed.")
     
    # If uci_id was not provided, use URL
    elif url is not None:
        print(f"Download data from URL: {url}")
        df = _fetch_from_url(url)

    # 3. Output Logic
    if output_path:
        _save_data(df, output_path)

    return df


def _check_ucimlrepo_installed():
    """Checks if ucimlrepo is installed without importing it globally."""
    spec = importlib.util.find_spec("ucimlrepo")
    return spec is not None


def _fetch_from_uci(uci_id):
    """Helper function to fetch from UCI repo."""
    from ucimlrepo import fetch_ucirepo

    try:
        # fetch dataset
        dataset = fetch_ucirepo(id=uci_id)

        # Assemble DataFrame (features + targets)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        return df
    except Exception as e:
        raise Exception(f"Failed to fetch data from UCI: {e}")


def _fetch_from_url(url):
    """Helper function to fetch from direct URL."""
    try:
        # Handle compression automatically if url ends in .zip/.gzip etc.
        return pd.read_csv(url)
    except Exception as e:
        raise Exception(f"Failed to download from URL: {e}")


def _save_data(df, path):
    """Helper function to save data to disk."""
    try:
        # Ensure the directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        # Save based on extension
        if path.endswith('.csv'):
            df.to_csv(path, index=False)
        elif path.endswith('.pkl') or path.endswith('.pickle'):
            df.to_pickle(path)
        else:
            # Default to CSV if unknown extension
            df.to_csv(path, index=False)
        
        print(f"Data successfully saved to {path}")
    except Exception as e:
        print(f"Warning: Could not save data to {path}. Error: {e}")