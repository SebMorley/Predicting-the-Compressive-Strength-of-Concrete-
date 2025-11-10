from datetime import date
from tkinter import _test
from turtle import pd
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd

 #Load dataset
data = pd.read_csv("fish_data.csv")

from sklearn.model_selection import train_test_split
# Load CSV data (skip header row)
print(data)
print(data.shape)

data = data.to_numpy() 
print(data)
print (data.shape)
print(type(data))

y = data[:,0]
print(y)

X = data[:,1:]

# Split into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# STEP 4: Standardise the data (zero mean, unit variance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn scaling from training data
X_test_scaled = scaler.transform(X_test)         # apply same scaling to test data

# Verify your data
print("Training features shape:", X_train_scaled.shape)
print("Testing labels example:", y_test[:5])

# STEP 4: Initialise classifiers
log_clf = LogisticRegression(max_iter=100, C=2.8)
knn_clf = KNeighborsClassifier(n_neighbors=11, weights='distance', metric='euclidean')
rf_clf = RandomForestClassifier(n_estimators=520, max_depth=20, min_samples_split=4, min_samples_leaf=2, random_state=42)

# STEP 5: Train classifiers
log_clf.fit(X_train_scaled, y_train)
knn_clf.fit(X_train_scaled, y_train)
rf_clf.fit(X_train_scaled, y_train)

# STEP 6: Generate predictions
y_pred_log = log_clf.predict(X_test_scaled)
y_pred_knn = knn_clf.predict(X_test_scaled)
y_pred_rf = rf_clf.predict(X_test_scaled)

# STEP 7: Evaluate performance
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("K-Nearest Neighbors Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load your dataset
data = pd.read_csv('fish_data.csv')
X = data.iloc[:, 1:]  # adjust if label column is elsewhere
y = data.iloc[:, 0]   # adjust as above

# 2. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# 3. Standardise features if needed - comment out if using only categorical data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define models and param grids
models = {
    'LogisticRegression': LogisticRegression(solver='liblinear', max_iter=250),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'RandomForestClassifier': RandomForestClassifier(random_state=42)
}
param_grids = {
    'LogisticRegression': {'C': [100, 10, 1.0, 0.1, 0.01], 'tol': [1e-4, 5e-4, 1e-5, 5e-5]},
    'KNeighborsClassifier': {'n_neighbors': [3,5,7,9,11], 'weights':['uniform','distance']},
    'RandomForestClassifier': {'n_estimators': [100, 200, 400], 'max_depth': [10, 20, 40], 'min_samples_split': [2,4], 'min_samples_leaf':[1,2]}
}

# 5. 5-fold cross-validation, manual gridsearch, and sklearn GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    print(f"\n--- {model_name} ---")
    best_score = 0
    best_params = None
    print("Manual Grid Search:")
    grid = param_grids[model_name]
    keys, values = list(grid.keys()), list(grid.values())
    import itertools
    for param_combo in itertools.product(*values):
        params = dict(zip(keys, param_combo))
        scores = []
        for train_idx, val_idx in kf.split(X_train_scaled):
            model.set_params(**params)
            model.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
            val_preds = model.predict(X_train_scaled[val_idx])
            scores.append(accuracy_score(y_train.iloc[val_idx], val_preds))
        avg_score = np.mean(scores)
        print(f"Params: {params}, Avg CV score: {avg_score:.4f}")
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
    print(f"Best manual: {best_params}, Best Avg CV score: {best_score:.4f}")

    print("Automated GridSearchCV:")
    gs = GridSearchCV(model, grid, cv=5, scoring='accuracy')
    gs.fit(X_train_scaled, y_train)
    print(f"Best CV params: {gs.best_params_}")
    print(f"Best CV score: {gs.best_score_:.4f}")

    print("Test set accuracy (best GS params):")
    test_preds = gs.best_estimator_.predict(X_test_scaled)