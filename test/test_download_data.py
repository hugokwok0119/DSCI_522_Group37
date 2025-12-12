import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# --- Path Setup ---
# Calculate the path to the project root (one level up from this 'test' folder)
# This ensures we can import modules from the 'src' directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the function to be tested
from src.download_data import download_data

# --- Fixtures ---

@pytest.fixture
def mock_df():
    """Returns a simple DataFrame to use as a mock return value."""
    return pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

# --- Test Cases ---

def test_download_data_uci_success(mock_df):
    """
    Test Case: Valid UCI ID provided and 'ucimlrepo' is installed.
    Expected: Data is fetched via _fetch_from_uci and saved.
    """
    # Patch internal helpers to isolate the logic of the main function
    with patch('src.download_data._check_ucimlrepo_installed', return_value=True), \
         patch('src.download_data._fetch_from_uci', return_value=mock_df) as mock_fetch, \
         patch('src.download_data._save_data') as mock_save:
        
        # Execute
        result = download_data(uci_id=17, output_path="data/test.csv")
        
        # Verify result matches mock
        assert result.equals(mock_df)
        
        # Verify specific helper calls
        mock_fetch.assert_called_once_with(17)
        mock_save.assert_called_once()
        
        # Verify the correct path was passed to the save function
        args, _ = mock_save.call_args
        assert args[1] == "data/test.csv"

def test_download_data_url_success(mock_df):
    """
    Test Case: URL provided (UCI ID is None).
    Expected: Data is fetched via _fetch_from_url.
    """
    # Patch URL fetcher and save function
    with patch('src.download_data._fetch_from_url', return_value=mock_df) as mock_fetch, \
         patch('src.download_data._save_data') as mock_save:
        
        test_url = "http://example.com/data.csv"
        result = download_data(url=test_url, output_path="data/test.csv")
        
        assert result.equals(mock_df)
        mock_fetch.assert_called_once_with(test_url)
        mock_save.assert_called_once()

def test_no_args_raises_error():
    """
    Test Case: No URL and no UCI ID provided.
    Expected: Raises ValueError.
    """
    with pytest.raises(ValueError, match="must provide either"):
        download_data(url=None, uci_id=None)

def test_uci_not_installed_raises_error():
    """
    Test Case: UCI ID provided, but 'ucimlrepo' is NOT installed.
    Expected: Raises ImportError.
    """
    # Simulate '_check_ucimlrepo_installed' returning False
    with patch('src.download_data._check_ucimlrepo_installed', return_value=False):
        with pytest.raises(ImportError, match="package is not installed"):
            download_data(uci_id=17)

def test_fetch_uci_failure_propagates_error():
    """
    Test Case: Underlying UCI fetch fails.
    Expected: The Exception from helper bubbles up.
    """
    with patch('src.download_data._check_ucimlrepo_installed', return_value=True), \
         patch('src.download_data._fetch_from_uci', side_effect=Exception("Connection Failed")):
        
        with pytest.raises(Exception, match="Connection Failed"):
            download_data(uci_id=17)

def test_save_not_called_if_no_path(mock_df):
    """
    Test Case: No output_path provided.
    Expected: _save_data is NOT called.
    """
    with patch('src.download_data._fetch_from_url', return_value=mock_df), \
         patch('src.download_data._save_data') as mock_save:
        
        download_data(url="http://fake.com", output_path=None)
        
        mock_save.assert_not_called()