import pandas as pd
import logging


def data_loader(filepath):
    """
    Load a CSV file into a DataFrame.

    Parameters:
    - filepath (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"CSV file loaded successfully from {filepath}.")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file from {filepath}: {e}")
        return None
