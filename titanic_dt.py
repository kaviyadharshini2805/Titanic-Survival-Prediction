import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import joblib
import pickle
import os
import matplotlib.pyplot as plt
from sklearn import tree

# Load dataset
data = pd.read_csv("titanic.csv")

print("\nFIRST 5 ROWS:")
print(data.head())

print("\nNULL VALUES:")
print(data.isnull().sum())

# Remove cabin (too many nulls)
data = data.drop(["cabin"], axis=1)

# Prepare features
X = data[["p_class", "sex", "age", "fare"]].copy()
X["sex"] = X["sex"].map({"male": 0, "female": 1})

X["age"] = X["age"].fillna(X["age"].mean())
X["fare"] = X["fare"].fillna(X["fare"].mean())

y = data["survived"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Evaluate
predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)
print("\nACCURACY:", accuracy)

cm = confusion_matrix(y, predictions)
print("\nCONFUSION MATRIX:")
print(cm)

# Visualize tree
plt.figure(figsize=(12, 8))
tree.plot_tree(model, feature_names=["p_class", "sex", "age", "fare"], filled=True)
plt.show()

# Save model
with open("titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)

joblib.dump(model, "titanic_model.joblib")

print("\nMODEL SAVED SUCCESSFULLY!")
