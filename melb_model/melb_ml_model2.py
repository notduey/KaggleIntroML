#!/usr/bin/env python3
"""
Validating a ML Model
"""
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
MELBOURNE_FILE_PATH = '/Users/duey/CodingProjects/KaggleIntroML/data/melb_data.csv'
melbourne_data = pd.read_csv(MELBOURNE_FILE_PATH)

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

# Define model
melbourne_model = DecisionTreeRegressor()

# Fit model
melbourne_model.fit(X, y)

#==================================================================================================#
# Validating a ML Model

# Mean Absolute Error (MAE) is a common metric for summarizing model accuracy.
# It is the average of the absolute value of the difference between the predicted and actual values.
# So if a house costs $150,000 and the model predicts $100,000, the (absolute) error is $50,000.

# mean_absolute_error() is a function in sklearn library
# It takes two arguments, the actual values and the predicted values, and returns the MAE
predicted_home_prices = melbourne_model.predict(X)
in_sample_mae = mean_absolute_error(y, predicted_home_prices)
print(f"MAE for in-sample data: {in_sample_mae}")

# The problem with "In-Sample" scores
# We used a single "sample" of houses for both building and evaluating the model.

# Here's how it's bad
# Imagine in the real-estate market, door color is unrelated to home price.
# In the sample date you used to build the mode, all homes with green doors were very expensive.
# The model will find this pattern and will always predict high prices for homes with green doors.
# Since this pattern was derived from the training data, the model will appear to be accurate.

#==================================================================================================#

# The the model's practical value is making predications on new data
# We measure performance on data that wasn't used to build the model
# The most common way is to exclude some data from the model-building process
# Then we will use those to test the model's accuracy on data it hasn't seen before
# This is called Validation Data

# Split data into training and validation data, for both features and target
# The split is based on a random number generator
# Giving random_state a value is used to ensure you always get the same split of the data

# train_test_split() "shuffles" the data, then splits them
# By default, 75% of the data is used for training, and 25% for validation
# In essence, the model splits the data, learning patterns from train_X and train_y
# Then it tests the model on val_X and is graded on val_y (the model never sees val_y)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

#train_X is features data used to train the model
#val_X is features data used to validate (test) the model
#train_y are the target values for training, i.e. the correct answers
#val_y are the target values for validation, i.e. the correct answers

# Define model
melbourne_model = DecisionTreeRegressor()

# Fit model
melbourne_model.fit(train_X, train_y)

# Get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
out_of_sample_mae = mean_absolute_error(val_y, val_predictions)
print(f"MAE for out-of-sample data: {out_of_sample_mae}")

# MAE for in-sample data: 434.71594577146544
# MAE for out-of-sample data: 262489.06262104586

# This is the difference between a model that is almost exactly right and a model that is outright unusable for practical purposes