# Kaggle Intro to Machine Learning
This repository contains my work from the Kaggle Intro to Machine Learning course.
The goal of this project is learning and practice, not building a production-ready model.

It documents my progression through core machine learning concepts using Python, Jupyter Notebook, and scikit-learn.

## What I Learned

Through this project, I learned and practiced:

- Loading and exploring datasets using pandas

- Selecting features and target variables

- Training machine learning models with scikit-learn

- Using decision trees and random forests

- Understanding underfitting vs overfitting

- Evaluating models using mean absolute error (MAE)

- Splitting data into training and validation sets

Transitioning from Jupyter notebooks to Python scripts

## Models and Experiments

The melb_model folder contains multiple versions of a housing price prediction model.
Each version builds on the previous one to improve performance and understanding:

- Model 1 – Basic decision tree

- Model 2 – Feature refinement

- Model 3 – Model validation improvements

- Model 4 – Random forest model

This progression helped me understand how model choice and validation affect accuracy.

## How to Run the Code
1. Create a virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```
2. Install dependencies
```
pip install -r requirements.txt
```
4. Run notebooks or scripts

Open notebooks with:
```
jupyter notebook
```

Or run Python scripts directly:
```
python melb_model/melb_ml_model4.py
```

## Notes

This repository is intended as a learning log and reference.

Code and structure reflect experimentation and practice
