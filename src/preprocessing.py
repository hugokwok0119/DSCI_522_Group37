import os
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Union
from pathlib import Path


def preprocess_data(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    numeric_features: Optional[List[str]] = None,
    drop_features: Optional[List[str]] = None,
    output_pickle_path: str = "results/models/preprocessor.pickle",
    save_transformed_csv: bool = False,
    train_output_path: str = "data/processed/scaled_train.csv",
    test_output_path: str = "data/processed/scaled_test.csv"
) -> dict:
 
    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("train_df must be a pandas DataFrame")
    
    if train_df.empty:
        raise ValueError("train_df cannot be empty")
    
    if test_df is not None and not isinstance(test_df, pd.DataFrame):
        raise TypeError("test_df must be a pandas DataFrame or None")
    
    if numeric_features is None:
        numeric_features = train_df.select_dtypes(include=['number']).columns.tolist()
    
    if drop_features is None:
        drop_features = []
    
    missing_numeric = set(numeric_features) - set(train_df.columns)
    if missing_numeric:
        raise ValueError(f"Numeric features not found in dataframe: {missing_numeric}")
    
    missing_drop = set(drop_features) - set(train_df.columns)
    if missing_drop:
        raise ValueError(f"Drop features not found in dataframe: {missing_drop}")
    
    transformers = []
    
    if numeric_features:
        transformers.append((StandardScaler(), numeric_features))
    
    if drop_features:
        transformers.append(("drop", drop_features))
    
    ct = make_column_transformer(
        *transformers,
        remainder="passthrough"
    )
    
    ct.fit(train_df)
    
    train_transformed_array = ct.transform(train_df)
    feature_names = ct.get_feature_names_out()
    train_transformed_df = pd.DataFrame(
        train_transformed_array, 
        columns=feature_names,
        index=train_df.index
    )
    
    test_transformed_df = None
    if test_df is not None:
        test_transformed_array = ct.transform(test_df)
        test_transformed_df = pd.DataFrame(
            test_transformed_array,
            columns=feature_names,
            index=test_df.index
        )
    
    output_dir = os.path.dirname(output_pickle_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(ct, f)
    
    if save_transformed_csv:
        train_dir = os.path.dirname(train_output_path)
        if train_dir:
            os.makedirs(train_dir, exist_ok=True)
        train_transformed_df.to_csv(train_output_path, index=False)
        
        if test_df is not None and test_output_path:
            test_dir = os.path.dirname(test_output_path)
            if test_dir:
                os.makedirs(test_dir, exist_ok=True)
            test_transformed_df.to_csv(test_output_path, index=False)
    
    return {
        'preprocessor': ct,
        'train_transformed': train_transformed_df,
        'test_transformed': test_transformed_df,
        'feature_names': list(feature_names)
    }