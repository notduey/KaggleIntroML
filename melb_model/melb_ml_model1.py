#!/usr/bin/env python3
"""
First Machine Learning Model

Scikit-learn is one of the most popular libraries for ML in Python.

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
# It asks a sequence of yes/no questions about data and ends with a numerical prediction
# Regressor means it predicts a numerical value

melbourne_model = DecisionTreeRegressor(random_state=1) # random_state is a seed value

# random_state initializes a pseudo-random number generator, which makes "random" choices deterministic
# it fixes the random choices inside the algorithm so you get the same model every time you run the code
# The value 1 doesn't mean anything, that's just a set starting point, it could be any integer

# Many ML algorithms use randomness internally, even if you didn't write random code, the algorithm might:
# break ties randomly
# randomly choose between equally good options
# shuffle data before learning
# all of these factors might mean the algorithm might produce different results each time you run it


# Fit the model:
# fit() is a method in sklearn library, it means "train the model"
# it takes two arguments X and y, X is the data and y is the prediction target

# In this case, the model will look at each row in X, viewing the columns: 
# ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
# Then it compares the values in each column to the prediction target y, which is the house price

melbourne_model.fit(X, y)

# After fit() is called, the model is trained, internal parameters are set, and it's ready to make predictions
# For a decision tree, the model builds a tree of decisions based on the data
# It chooses which features to split on, deiciding where to split, repeating until the stopping criterion is met


# Predict the model:
print("Making predictions for the following 5 houses:")
# visually check the data to later confirm the model is predicting the same type of data it was trained on
print(X.head()) # print the first 5 rows of the data and their implied features
print("The predictions are:")

# predict() is a method in sklearn library, it means "use the model to make predictions"
# The parameters passed to predict() are the data to predict, in this case X.head()
# Using the patterns during fit(), predict the target values (house prices) for these examples
print(melbourne_model.predict(X.head())) # print the predictions for X.head()

print("The actual prices are:")
print(y.head().to_string()) # print the actual house prices from the data
# to_string() has to be called because if it isn't
# then "Name: Price, dtype: float64" will be printed at the end
