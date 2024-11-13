# train_boundary_model.py

import numpy as np
from sklearn.tree import DecisionTreeClassifier

def train_boundary_model(X_train, boundary_indices):
    array_indexes = np.zeros(len(X_train))
    perc = int(0.1 * len(X_train))
    
    for i in range(perc):
        index = boundary_indices[i]
        array_indexes[index] = 1

    model = DecisionTreeClassifier().fit(X_train, array_indexes)
    return model
