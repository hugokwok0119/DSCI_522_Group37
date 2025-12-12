import pytest
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.eda import perform_eda

@pytest.fixture
def mock_df():
    """Create a small mock DataFrame."""
    return pd.DataFrame({
        'radius_mean': [10, 11, 12],
        'texture_mean': [20, 21, 22],
        'other_col': [1, 1, 1],
        'Diagnosis': ['M', 'B', 'M']
    })

def test_perform_eda_runs_successfully(mock_df):
    """
    Test that perform_eda calls all plotting and saving functions.
    """
    # Mock output directory interactions
    # 关键修改：增加了 patch('os.path.exists', return_value=False)
    with patch('os.makedirs') as mock_makedirs, \
         patch('os.path.exists', return_value=False), \
         patch('builtins.open', new_callable=MagicMock) as mock_open, \
         patch('pandas.DataFrame.to_csv') as mock_to_csv, \
         patch('altair_ally.corr') as mock_corr, \
         patch('altair_ally.pair') as mock_pair, \
         patch('altair_ally.dist') as mock_dist:
        
        # Create mocks for the Chart objects returned by altair_ally
        mock_chart_obj = MagicMock()
        mock_corr.return_value = mock_chart_obj
        mock_pair.return_value = mock_chart_obj
        mock_dist.return_value = mock_chart_obj
        
        # Action
        perform_eda(mock_df, 'results', target_col='Diagnosis', pair_plot_cols=['radius_mean', 'Diagnosis'])
        
        # Assertions
        
        # 1. Check directories created
        # 现在因为 os.path.exists 永远返回 False，代码一定会去创建文件夹
        assert mock_makedirs.called
        
        # 2. Check text summaries saved
        assert mock_open.called # For info.txt
        assert mock_to_csv.called # For summary.csv
        
        # 3. Check Charts generated
        assert mock_corr.called
        assert mock_pair.called
        assert mock_dist.called
        
        # 4. Check Charts saved (called 3 times: corr, pair, dist)
        assert mock_chart_obj.save.call_count == 3

def test_pair_plot_filtering(mock_df):
    """
    Test that the pair plot function receives the correctly filtered dataframe.
    """
    with patch('altair_ally.pair') as mock_pair, \
         patch('altair_ally.corr'), \
         patch('altair_ally.dist'), \
         patch('builtins.open'), \
         patch('pandas.DataFrame.to_csv'), \
         patch('os.makedirs'):
        
        # Return a mock chart to avoid AttributeError on .save()
        mock_pair.return_value = MagicMock()
        
        # Pass a subset of columns
        subset = ['radius_mean', 'Diagnosis']
        perform_eda(mock_df, 'results', target_col='Diagnosis', pair_plot_cols=subset)
        
        # Verify that altair_ally.pair was called
        args, _ = mock_pair.call_args
        passed_df = args[0]
        
        # Check that the dataframe passed to pair() only has the subset columns
        assert list(passed_df.columns) == subset
        assert 'other_col' not in passed_df.columns
        
def test_perform_eda_raises_error_on_invalid_input():
    """
    Test Case: Invalid input type (not a DataFrame).
    Expected: Raises TypeError.
    """
    invalid_input = ["not", "a", "dataframe"]
    
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        perform_eda(invalid_input, 'results')