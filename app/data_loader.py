import pandas as pd

def load_csv_data(filepath):
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(filepath)

def preprocess_text(df):
    """Convert DataFrame rows to a list of text strings.
    Each row is joined into a single string with all column values."""
    return df.apply(lambda row: " ".join(map(str, row.values)), axis=1).tolist()