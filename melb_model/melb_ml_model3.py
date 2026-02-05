#!/usr/bin/env python3
"""
Test fitting a model (underfitting and overfitting)
"""
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """
    Calculates the mean absolute error between predicted and actual values.
    """
    # max_lead_nodes limits how many leaves (final prediction groups) a decision tree can have
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

def main():
    """
    Main function
    """
    # Load data
    melbourne_file_path = '/Users/duey/CodingProjects/KaggleIntroML/data/melb_data.csv'
    melbourne_data = pd.read_csv(melbourne_file_path)

    # Filter rows with missing price values
    filtered_melbourne_data = melbourne_data.dropna(axis=0)

    # Choose target and features
    y = filtered_melbourne_data.Price
    melbourne_features = [
        'Rooms',
        'Bathroom',
        'Landsize',
        'BuildingArea',
        'YearBuilt',
        'Lattitude',
        'Longtitude'
    ]
    X = filtered_melbourne_data[melbourne_features]

    # Split data into training and validation data, for both features and target
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

    # Compare MAE with dffering values of max_leaf_nodes
    for max_leaf_nodes in [5, 50, 500, 5000]:
        my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

    # Max leaf nodes: 5                Mean Absolute Error:  347380
    # Max leaf nodes: 50               Mean Absolute Error:  258171
    # Max leaf nodes: 500              Mean Absolute Error:  243495
    # Max leaf nodes: 5000             Mean Absolute Error:  255015

    # The model with 500 lead nodes has the lowest MAE, so it is the most optimal from this list

    # max_leaf_nodes means how many leaves a decision tree can have before it stops
    # too many leaves can cause overfitting, the model captures patterns that won't recur in new data, and makes predictions that are too specific
    # accurate on training data since captures every pattern it sees, no matter how specific, but performs poorly on new data because those hyperspecific patterns don't recur
    # too few leaves can cause underfitting, the model makes predictions that are too generalized (not accurate on both training and new data)

if __name__ == '__main__':
    main()
