import time  # Checking how long they run to measure

import os
import copy
import pickle
import sympy
import functools
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from error_injection import MissingValueError, SamplingError, Injector
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import mutual_info_score, auc, roc_curve, roc_auc_score, f1_score
from scipy.optimize import minimize as scipy_min
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, Bounds, linprog
from sympy import Symbol as sb
from sympy import lambdify
from tqdm.notebook import trange, tqdm
from IPython.display import display, clear_output
from random import choice

class style():
    RED = '\033[31m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    RESET = '\033[0m'

np.random.seed(1)

# Ignore all the warnings
import warnings
warnings.filterwarnings('ignore')

# Load only relevant columns, TV shows from biggest markets of Netflix by subscirber count
netflix_data = pd.read_csv('netflix_titles.csv', usecols=['type', 'director', 'country', 'rating', 'duration'])
tv_shows = netflix_data[
    (netflix_data['type'] == 'TV Show') & 
    (netflix_data['country'].isin(['United States', 'Brazil', 'United Kingdom', 'Germany', 'France']))
].drop('type', axis=1).head(50)

# Convert 'duration' to numeric values, handling "Seasons" separately
tv_shows['duration'] = tv_shows['duration'].apply(lambda x: int(x.split()[0]) if 'Season' in x else None)
tv_shows = tv_shows.dropna(subset=['duration'])

# Initialize the LabelEncoder
rating_encoder = LabelEncoder()
director_encoder = LabelEncoder()
tv_shows['rating'] = rating_encoder.fit_transform(tv_shows['rating'])
tv_shows['director'] = director_encoder.fit_transform(tv_shows['director'])

# Create dictionaries for original encodings
rating_encoding = {original: encoded for original, encoded in zip(rating_encoder.classes_, range(len(rating_encoder.classes_)))}
director_encoding = {original: encoded for original, encoded in zip(director_encoder.classes_, range(len(director_encoder.classes_)))}

# Display the original encodings
print("Rating Encoding:", rating_encoding)
print("Director Encoding:", director_encoding)

# Print the final DataFrame for verification (optional)
print(tv_shows.head())

def load_netflix(tv_shows):
    # Fetch dataset
    features = ['country', 'rating', 'director']
    X = tv_shows[features]
    y = tv_shows['duration']

    # With this random seed, no null value is included in the test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train = copy.deepcopy(X_train).reset_index(drop=True)
    X_test = copy.deepcopy(X_test).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test

# First impute the data and make it hypothetically clean
def load_netflix_cleaned(tv_shows):
    # Fetch dataset
    features = ['country', 'rating', 'director']
    X = tv_shows[features]
    y = tv_shows['duration']

    # Encode categorical variables using LabelEncoder
    for feature in features:
        if X[feature].dtype == 'object':
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature].astype(str))

    # Impute missing values using KNNImputer
    imputer = KNNImputer(n_neighbors=10)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Split the data into training and testing sets with a random seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train = copy.deepcopy(X_train).reset_index(drop=True)
    X_test = copy.deepcopy(X_test).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test

# Load the cleaned data
X_train, X_test, y_train, y_test = load_netflix_cleaned(tv_shows)

# Print the training data for verification (optional)
print(X_train)
