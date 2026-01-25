#!/usr/bin/env python3
"""
First Machine Learning Model

Steps to building and using a model:
1. Define: what type of model will it be? A decision tree? Some other type of model? Some other paramesters of the model type are specified too.
2. Fit: capture patterns provided by the date. This is the heart of modeling.
3. Predict: just what it sounds like.
4. Evaluate: Determine how accurate the model's predictions are.
"""
from sklearn.tree import DecisionTreeRegressor
from melb_ml_data import X, y


# Define the model:
# DecisionTreeRegressor() is a class in sklearn library
# Decision Tree is a model that uses a tree structure to make predictions
melbourne_model = DecisionTreeRegressor(random_state=1) # random_state is a seed value

# Fit the model:
# fit() is a method in sklearn library, it means "train the model"
# After fit() is called, the model is trained, internal parameters are set, and it's ready to make predictions
melbourne_model.fit(X, y)

# Predict the model:
print("Making predictions for the following 5 houses:")
print(X.head()) # print the first 5 rows of the data and their implied features
print("The predictions are:")

# predict() is a method in sklearn library, it means "use the model to make predictions"
print(melbourne_model.predict(X.head())) # print the predictions for X.head()

print("The actual prices are:")
print(y.head().to_string()) # print the actual house prices from the data


# Evaluate the model:
# The predictions are the same as the actual values when the file is run
