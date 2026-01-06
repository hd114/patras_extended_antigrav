import pandas as pd
import numpy as np

def preprocess_ciciot_data(input_csv_path: str, output_parquet_path: str, show_plot: bool = False) -> pd.DataFrame:
    """
    Preprocess the CICIoT dataset: clean NaNs, remove constant columns, convert types, and save as Parquet.

    Args:
        input_csv_path (str): Path to the raw CSV file.
        output_parquet_path (str): Path to save the processed data in Parquet format.
        show_plot (bool): Whether to display a bar plot of label distribution.

    Returns:
        pd.DataFrame: The cleaned and processed dataframe.
    """
    # Load data
    df = pd.read_csv(input_csv_path)

    # Remove rows where 'label' is NaN
    df = df.dropna(subset=['label']).reset_index(drop=True)

    # Drop columns with constant values
    constant_value_columns = df.columns[df.nunique() <= 1].tolist()
    df = df.drop(columns=constant_value_columns)

    # Convert all columns to float except 'label'
    cols_to_convert = df.columns.difference(['label'])
    df[cols_to_convert] = df[cols_to_convert].astype(float)

    # Save processed data
    df.to_parquet(output_parquet_path, engine='pyarrow')

    # Optional: Plot label distribution
    if show_plot:
        import matplotlib.pyplot as plt
        value_counts = df['label'].value_counts()
        plt.figure(figsize=(10, 6))
        value_counts.plot(kind='bar')
        plt.xlabel('Attack Types')
        plt.ylabel('Number of Attacks')
        plt.title('Number of Attacks per Type')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    return df

def preprocess_edgeiiot_data(input_parquet_path: str, subsample_fraction: float = 0.6, random_state: int = 42) -> pd.DataFrame:
    """
    Preprocess the EdgeIIoT dataset: normalize data and create a stratified subsample.

    Args:
        input_parquet_path (str): Path to the processed parquet file.
        subsample_fraction (float): Fraction of data to sample per class.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Subsampled and normalized dataframe.
    """
    df = pd.read_parquet(input_parquet_path)

    # Stratified subsample
    sample_size = int(subsample_fraction * len(df))
    stratum_sizes = df['Label'].value_counts(normalize=True) * sample_size
    samples = [df[df['Label'] == stratum].sample(n=int(size), random_state=random_state) for stratum, size in stratum_sizes.items()]
    subsample = pd.concat(samples)

    # Normalize features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(subsample.drop('Label', axis=1).values)
    y = subsample['Label'].values

    return X_normalized.astype("float32"), y, scaler
