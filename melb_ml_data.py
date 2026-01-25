#!/usr/bin/env python3
"""
Data for Melbourne Machine Learning Model
"""
import pandas as pd

MELBOURNE_FILE_PATH = '/Users/duey/CodingProjects/KaggleIntroML/data/melb_data.csv'
melbourne_data = pd.read_csv(MELBOURNE_FILE_PATH)

# dropna() removes rows/columns that have any missing values
melbourne_data = melbourne_data.dropna(axis=0) # axis=0 drops rows, axis = 1 drops columns instead

# "Price" is one of the columns in the data, and is our prediction target
y = melbourne_data.Price

# The columns that are inputted into the model (and later used to make preducations) are called "features".
# We select multiple features by providing a list of column names, by convention this data is called X:
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

def preview_data():
    """
    Preview and check data
    """
    # Display columns:
    print(melbourne_data.columns)

    # Reviewing the data:
    # describe() gives a statistical summary of the data
    print(X.describe())

    # head() gives the first n rows of the data, by default n=5
    print(X.head())


if __name__ == "__main__":
    preview_data()
