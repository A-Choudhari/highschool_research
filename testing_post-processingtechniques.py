import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve
import numpy as np
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

sex = df_balanced["sex"]

# Splitting into train, validation, and test sets
X_train_full, X_test, y_train_full, y_test, sex_train_full, sex_test = train_test_split(X, y, sex, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val, sex_train, sex_val = train_test_split(X_train_full, y_train_full, sex_train_full, test_size=0.25, random_state=42)


probs = clf.predict_proba(X_val)[:, 1]

# Compute ROC curve for each group
fpr_m, tpr_m, thresholds_m = roc_curve(y_val[sex_val == "Male"], probs[sex_val == "Male"])
fpr_f, tpr_f, thresholds_f = roc_curve(y_val[sex_val == "Female"], probs[sex_val == "Female"])

# Function to find the threshold for equalizing TPRs
def find_threshold(tpr, thresholds, target_tpr):
    return thresholds[(np.abs(tpr - target_tpr)).argmin()]

# Equalize TPR
target_tpr = (tpr_m.sum() + tpr_f.sum()) / (len(tpr_m) + len(tpr_f))
threshold_m = find_threshold(tpr_m, thresholds_m, target_tpr)
threshold_f = find_threshold(tpr_f, thresholds_f, target_tpr)

# Apply thresholds to the test set
test_probs = clf.predict_proba(X_test)[:, 1]
test_predictions = np.zeros_like(test_probs)
test_predictions[(sex_test == "Male") & (test_probs > threshold_m)] = 1
test_predictions[(sex_test == "Female") & (test_probs > threshold_f)] = 1

# Measure accuracy or other metrics
accuracy = accuracy_score(y_test, test_predictions)
print(f"Post-processed accuracy: {accuracy}")



