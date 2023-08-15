import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = fetch_openml(name="adult", version=2)
df = pd.DataFrame(data.data, columns=data.feature_names)
df["income"] = data.target
# Rename and encode the target variable for clarity
df['income'] = df['income'].apply(lambda x: 1 if x == ">50K" else 0)

# Strategy 1: Pre-processing - Equal representation resampling
male_data = df[df["sex"] == "Male"]
female_data = df[df["sex"] == "Female"]

# Undersample for balanced representation
sample_size = min(len(male_data), len(female_data))
df_balanced = pd.concat([male_data.sample(sample_size), female_data.sample(sample_size)])

X = df_balanced[["age", "hours-per-week", "sex"]]
y = df_balanced["income"]

X["sex"] = X["sex"].apply(lambda x: 1 if x == "Male" else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy after pre-processing (resampling): {accuracy_score(y_test, y_pred)}")





# Strategy 2: Post-processing - Adjusting decision threshold

# Rename and encode the target variable for clarity
df['income'] = df['income'].apply(lambda x: 1 if x == ">50K" else 0)

# Strategy 1: Pre-processing - Equal representation resampling
male_data = df[df["sex"] == "Male"]
female_data = df[df["sex"] == "Female"]

# Undersample for balanced representation

# Hyperparameter optimization using Grid Search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

clf = RandomForestClassifier()

grid_search = RandomizedSearchCV(clf, param_distributions=param_grid,
                                 cv=3, n_jobs=1, verbose=2, scoring='accuracy')

grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_

y_pred = best_clf.predict(X_test)

print(f"Accuracy after pre-processing with optimized parameters: {accuracy_score(y_test, y_pred)}")
