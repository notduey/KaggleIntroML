#!/usr/bin/env python3
"""
Data for Machine Learning
"""
import pandas as pd

MELBOURNE_FILE_PATH = '/Users/duey/CodingProjects/KaggleIntroML/data/melb_data.csv'
melbourne_data = pd.read_csv(MELBOURNE_FILE_PATH)


# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.
# Your Iowa data doesn't have missing values in the columns you use.
# So we will take the simplest option for now, and drop houses from our data.
# Don't worry about this much for now, though the code is:

# dropna() removes rows/columns that have any missing values
melbourne_data = melbourne_data.dropna(axis=0) # axis=0 drops rows, axis = 1 drops columns instead


# Dot-notation:
# You can pull out a variable with dot-notation and the column is stored in a Series (a one-dimensional array).
# We'll use dot-notation to pull the column we want to predict (called the Prediction Target).
# By convention, the predication target is called y, so the code to save house prices is:

# "Price" is one of the columns in the data, and is our prediction target
y = melbourne_data.Price


# Column list selection:
# The columns that are inputted into the model (and later used to make preducations) are called "features".
# In our case, they are the columns used to determine the house prices.
# Sometimes you will use all columns except the target column as features, other time it is better to use a subset (small number of columns).

# For now we'll build a model with only a few features. Later on we'll iterate and compare models built with different features.
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


# Visually checking your data is an important part of the data science process.
# You'll frequently find that the data you collected isn't what you expected it to be.


if __name__ == "__main__":
    preview_data()
