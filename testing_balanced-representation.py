import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
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
features_with_sex = ["age", "hours-per-week", "sex"]
X = df[features_with_sex]
y = df["income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert "sex" to integer for modeling but keep the original for thresholding
X_train["sex_int"] = X_train["sex"].apply(lambda x: 1 if x == "Male" else 0)
X_test["sex_int"] = X_test["sex"].apply(lambda x: 1 if x == "Male" else 0)

clf = RandomForestClassifier()
clf.fit(X_train[["age", "hours-per-week", "sex_int"]], y_train)

# Adjust decision threshold based on gender
threshold_male = 0.6
threshold_female = 0.4

y_pred_proba = clf.predict_proba(X_test[["age", "hours-per-week", "sex_int"]])[:, 1]
df_test = X_test.copy()
df_test["true_income"] = y_test
df_test["pred_proba"] = y_pred_proba

df_test["predicted_income"] = df_test.apply(
    lambda x: 1 if (x["sex"] == "Male" and x["pred_proba"] > threshold_male) or
              (x["sex"] == "Female" and x["pred_proba"] > threshold_female) else 0, axis=1)

print(f"Accuracy after post-processing (threshold adjustment): {accuracy_score(df_test['true_income'], df_test['predicted_income'])}")
