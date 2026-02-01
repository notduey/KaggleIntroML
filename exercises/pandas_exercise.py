#!/usr/bin/env python3
"""
Course Step 2/7 Basic Data Exploration
"""

import pandas as pd
# save filepath to variable for easier access
MELBOURNE_FILE_PATH = '/Users/duey/CodingProjects/KaggleIntroML/data/melb_data.csv'

# read the data and sotre data in DataFrame called melbourne_data
melbourne_data = pd.read_csv(MELBOURNE_FILE_PATH)

# print a summary of the data
print(melbourne_data.describe())
