import os
import io
import pandas as pd
import altair as alt
import altair_ally as aly
import warnings

# Suppress warnings inside the module to keep output clean
warnings.filterwarnings('ignore', category=DeprecationWarning) 
warnings.filterwarnings('ignore', module='altair')

def perform_eda(df, output_dir, target_col=None, pair_plot_cols=None):
    """
    Performs Exploratory Data Analysis (EDA) on a given DataFrame and saves artifacts.

    This function generates:
    1. Textual summaries (info, describe)
    2. Correlation chart
    3. Distribution chart
    4. Pair plot (optional subset)

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    output_dir : str
        The directory where results will be saved.
    target_col : str, optional
        The name of the target column to use for coloring plots (e.g., 'Diagnosis').
    pair_plot_cols : list of str, optional
        A subset of columns to use for the pair plot. 
        If None, all columns are used (be careful with large datasets).

    Returns
    -------
    None
        Saves files to disk.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"The argument 'df' must be a pandas DataFrame, got {type(df).__name__} instead.")
    
    # 1. Setup Directories
    images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Created directory: {images_dir}")

    print("Generating textual summaries...")
    
    # 2. Save Info (Capture stdout)
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    info_path = os.path.join(output_dir, 'eda_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(info_str)
    
    # 3. Save Description
    describe_path = os.path.join(output_dir, 'eda_summary.csv')
    df.describe(include='all').to_csv(describe_path)

    print("Generating visualizations...")
    
    # Enable vegafusion for handling larger datasets efficiently if installed
    # try:
    #     alt.data_transformers.enable('vegafusion')
    # except:
    #     pass

    # 4. Correlation Plot
    corr_chart = aly.corr(df)
    corr_chart.save(os.path.join(images_dir, 'corr_chart.png'))
    
    # 5. Pair Plot (Handle the special subset logic here via arguments)
    if pair_plot_cols:
        # Validate columns exist
        valid_cols = [c for c in pair_plot_cols if c in df.columns]
        plot_df = df[valid_cols]
    else:
        plot_df = df

    pair_chart = aly.pair(plot_df, color=f"{target_col}:N" if target_col else None)
    pair_chart.save(os.path.join(images_dir, 'pair_chart.png'))

    # 6. Distribution Plot
    dist_chart = aly.dist(df, color=target_col if target_col else None)
    dist_chart.save(os.path.join(images_dir, 'dist_chart.png'))

    print(f"EDA artifacts saved to {output_dir}")