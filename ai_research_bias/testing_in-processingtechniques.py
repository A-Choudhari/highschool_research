import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
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

features = ["age", "hours-per-week"]

train_df, test_df = train_test_split(df_balanced, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(train_df[features].values, dtype=torch.float32)
y_train_tensor = torch.tensor(train_df["income"].values, dtype=torch.float32).view(-1, 1)
sex_train_tensor = torch.tensor((train_df["sex"] == "Male").values, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(test_df[features].values, dtype=torch.float32)
y_test_tensor = torch.tensor(test_df["income"].values, dtype=torch.float32).view(-1, 1)

class PrimaryClassifier(nn.Module):
    def __init__(self):
        super(PrimaryClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Define the adversary
class Adversary(nn.Module):
    def __init__(self):
        super(Adversary, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

primary = PrimaryClassifier()
adversary = Adversary()

primary_optimizer = optim.Adam(primary.parameters(), lr=0.001)
adversary_optimizer = optim.Adam(adversary.parameters(), lr=0.001)

criterion = nn.BCELoss()

# Adversarial training loop
epochs = 30
for epoch in range(epochs):
    # Train primary classifier
    primary_optimizer.zero_grad()
    predictions = primary(X_train_tensor)
    primary_loss = criterion(predictions, y_train_tensor)
    primary_loss.backward()
    primary_optimizer.step()

    # Train adversary
    adversary_optimizer.zero_grad()
    adversary_predictions = adversary(predictions.detach())
    adversary_loss = criterion(adversary_predictions, sex_train_tensor)
    adversary_loss.backward()
    adversary_optimizer.step()

    # Update primary classifier against adversary
    primary_optimizer.zero_grad()
    predictions = primary(X_train_tensor)
    adversary_predictions = adversary(predictions)
    adversary_loss = criterion(adversary_predictions, sex_train_tensor)
    # The negative sign ensures that primary tries to decrease the adversary's performance
    (-adversary_loss).backward()
    primary_optimizer.step()


# Evaluating accuracy
with torch.no_grad():
    test_predictions = primary(X_test_tensor)
test_predictions_labels = (test_predictions > 0.5).float()

accuracy = accuracy_score(y_test_tensor.numpy(), test_predictions_labels.numpy())
print(f"Accuracy on test set: {accuracy}")
