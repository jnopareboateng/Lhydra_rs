import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path):
    """
    Load, preprocess, and return a cleaned pandas DataFrame with MinMax scaling applied to the 'plays' column.

    :param file_path: str, the path to the CSV file to be loaded.
    :return: pd.DataFrame, the cleaned and scaled DataFrame.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop rows with any missing values
    df = df.dropna()

    # Remove duplicate records, keeping the first occurrence
    df = df.drop_duplicates()

    # Apply a log transformation to reduce skewness in the 'plays' column
    df["log_plays"] = np.log1p(df["plays"])

    # Initialize the MinMaxScaler to scale between 1 and 5
    scaler = MinMaxScaler(feature_range=(1, 5))

    # Fit and transform the 'log_plays' data
    df["scaled_ratings"] = scaler.fit_transform(df[["log_plays"]])

    # Drop the 'log_plays' column as it's no longer needed
    df.drop("log_plays", axis=1, inplace=True)

    # Convert categorical columns to numerical
    # df["gender"] = df["gender"].astype("category").cat.codes
    # df["education"] = df["education"].astype("category").cat.codes

    # Handle missing values
    df["featured_artists"].fillna("No Featured Artists", inplace=True)
    df["genre"].fillna("Unknown", inplace=True)

    # Save the cleaned and scaled data to a new CSV file
    df.to_csv("../data/cleaned_data.csv", index=False)

    return df



# load_data("../data/dataset.csv")