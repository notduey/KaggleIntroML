#!/usr/bin/env python3
"""
Using random forest ml model

-A deep trees overfit and each prediction is based on very specific features, making it's predictions too specific as it picks up on random patterns in its training data that aren't present in the validation data
-A shallow tree underfits and each prediction is based on not enough features, making it's predictions too general as it now won't pick up on real patterns that exist in the validation data
-All models have a trade-off between deep and shallow trees, called bias-variance tradeoff

-Instead of trying to optimize a single tree, we can use a random forest model, which builds many decision trees and averages their predictions
-Each tree in a random forest is trained on different random subset of rows for training data and columns(features) at each iteration
-The final prediction is the average of all predictions from the trees

-Final prediction is the mean (average) for regressions
-Final prediction is the mode (majority vote) for classification

-The idea behind a random forest is that each tree is overfit but averaging makes the noise average out, real patterns become more visible
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Load data
MELBOURNE_FILE_PATH = '/Users/duey/CodingProjects/KaggleIntroML/data/melb_data.csv'
melbourne_data = pd.read_csv(MELBOURNE_FILE_PATH)

# Filter rows with missing values
melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features
y = melbourne_data.Price
melbourne_features = [
    'Rooms',
    'Bathroom',
    'Landsize',
    'BuildingArea',
    'YearBuilt',
    'Lattitude',
    'Longtitude'
]
X = melbourne_data[melbourne_features]

# Split data into training and validation data, for both features and target
# The split is based on a random number generator, giving a value to random_state guarantees you get the same split every run
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Define and fit model
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)

# Get predictions
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

# MAE: 191656.64906176025
# Better than the best decision tree error
