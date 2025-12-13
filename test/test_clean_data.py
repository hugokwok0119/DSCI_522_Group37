import pytest
import pandas as pd
import os
import sys
import importlib.util
from unittest.mock import patch, MagicMock

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

script_name = "2_clean_data"
file_path = os.path.join(project_root, "scripts", f"{script_name}.py")

# Debugging check
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Script not found at: {file_path}")

spec = importlib.util.spec_from_file_location(script_name, file_path)
clean_data_module = importlib.util.module_from_spec(spec)
sys.modules[script_name] = clean_data_module
spec.loader.exec_module(clean_data_module)

# Access the function
clean_and_split_data = clean_data_module.clean_and_split_data

# --- Fixtures ---

@pytest.fixture
def mock_raw_df():
    """Create a mock raw DataFrame with original column names."""
    # INCREASED SIZE: 20 rows to allow stratified split with test_size=0.2
    return pd.DataFrame({
        'radius1': [10.0] * 20,
        'texture2': [20.0] * 20,
        'perimeter3': [30.0] * 20,
        'Diagnosis': ['M', 'B'] * 10
    })

# --- Test Cases ---

def test_clean_data_success(mock_raw_df):
    """
    Test Case: Valid DataFrame input.
    Expected: Columns renamed, target mapped, files saved, and preprocessing called.
    """
    with patch('os.makedirs') as mock_makedirs, \
         patch('os.path.exists', return_value=False), \
         patch('pandas.DataFrame.to_csv') as mock_to_csv, \
         patch.object(clean_data_module, 'preprocess_data') as mock_preprocess:

        output_file = 'data/processed/breast_cancer_cleaned.csv'
        train_df, test_df = clean_and_split_data(mock_raw_df, output_file)

        # 1. Verify Directory Creation
        assert mock_makedirs.called

        # 2. Verify File Saving (Full, Train, Test)
        assert mock_to_csv.call_count == 3
        
        # 3. Verify Logic: Column Renaming
        assert 'radius_mean' in train_df.columns
        assert 'texture_se' in train_df.columns
        assert 'perimeter_max' in train_df.columns
        assert 'radius1' not in train_df.columns

        # 4. Verify Logic: Target Mapping
        assert 'Malignant' in train_df['Diagnosis'].values
        assert 'M' not in train_df['Diagnosis'].values

        # 5. Verify Preprocessing Call
        assert mock_preprocess.called
        _, kwargs = mock_preprocess.call_args
        assert kwargs['save_transformed_csv'] is True
        assert 'radius_mean' in kwargs['numeric_features']

def test_unnamed_col_raises_error(mock_raw_df):
    """
    Test Case: DataFrame contains 'Unnamed' column (bad index save).
    Expected: Raises AssertionError.
    """
    mock_raw_df['Unnamed: 0'] = [0] * 20
    
    with pytest.raises(AssertionError, match="Unnamed index column detected"):
        clean_and_split_data(mock_raw_df, 'results/output.csv')

def test_invalid_input_raises_error():
    """
    Test Case: Input is not a DataFrame.
    Expected: Raises TypeError.
    """
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        clean_and_split_data("invalid input", 'results/output.csv')

def test_existing_directory_does_not_call_makedirs(mock_raw_df):
    """
    Test Case: Output directory already exists.
    Expected: os.makedirs is NOT called.
    """
    with patch('os.path.exists', return_value=True), \
         patch('os.makedirs') as mock_makedirs, \
         patch('pandas.DataFrame.to_csv'), \
         patch.object(clean_data_module, 'preprocess_data'):
        
        clean_and_split_data(mock_raw_df, 'existing_folder/output.csv')
        
        mock_makedirs.assert_not_called()