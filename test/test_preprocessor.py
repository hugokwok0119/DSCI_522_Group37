import pytest
import pandas as pd
import os
import tempfile
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.preprocessing import preprocess_data


def test_basic_preprocessing():
    """Test that preprocessing runs without errors"""
    train_data = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0],
        'feature2': [10.0, 20.0, 30.0, 40.0],
        'target': ['A', 'B', 'A', 'B']
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pickle_path = os.path.join(tmpdir, "preprocessor.pkl")
        
        result = preprocess_data(
            train_df=train_data,
            numeric_features=['feature1', 'feature2'],
            output_pickle_path=pickle_path
        )
        
        assert result['train_transformed'] is not None
        assert len(result['train_transformed']) == 4


def test_pickle_file_created():
    """Test that the pickle file is saved"""
    train_data = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [4.0, 5.0, 6.0]
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pickle_path = os.path.join(tmpdir, "preprocessor.pkl")
        
        preprocess_data(
            train_df=train_data,
            numeric_features=['feature1'],
            output_pickle_path=pickle_path
        )
        
        assert os.path.exists(pickle_path)


def test_scaling_works():
    """Test that StandardScaler produces mean of 0"""
    train_data = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0]
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pickle_path = os.path.join(tmpdir, "preprocessor.pkl")
        
        result = preprocess_data(
            train_df=train_data,
            numeric_features=['feature1'],
            output_pickle_path=pickle_path
        )
        
        scaled_values = result['train_transformed'].iloc[:, 0]
        
        assert abs(scaled_values.mean()) < 0.0001


def test_empty_dataframe_error():
    """Test that empty DataFrame raises an error"""
    empty_data = pd.DataFrame()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pickle_path = os.path.join(tmpdir, "preprocessor.pkl")
        
        # This should raise a ValueError
        with pytest.raises(ValueError):
            preprocess_data(
                train_df=empty_data,
                output_pickle_path=pickle_path
            )


def test_with_train_and_test():
    """Test preprocessing with both train and test data"""
    train_data = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0],
        'feature2': [10.0, 20.0, 30.0, 40.0]
    })
    
    test_data = pd.DataFrame({
        'feature1': [5.0, 6.0],
        'feature2': [50.0, 60.0]
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pickle_path = os.path.join(tmpdir, "preprocessor.pkl")
        
        result = preprocess_data(
            train_df=train_data,
            test_df=test_data,
            numeric_features=['feature1', 'feature2'],
            output_pickle_path=pickle_path
        )
        
        assert result['train_transformed'] is not None
        assert result['test_transformed'] is not None
        assert len(result['test_transformed']) == 2