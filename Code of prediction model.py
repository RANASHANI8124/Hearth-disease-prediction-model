# Heart Disease Prediction using Machine Learning

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
heart_data = pd.read_csv("heart.csv")

# Basic info
print(heart_data.head())
print(heart_data.shape)
print(heart_data.isnull().sum())

# Features and target
X = heart_data.drop(columns="target", axis=1)
Y = heart_data["target"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, stratify=Y, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Training
model.fit(X_train, Y_train)

# Training accuracy
train_pred = model.predict(X_train)
train_acc = accuracy_score(Y_train, train_pred)
print("Training Accuracy:", train_acc * 100)

# Testing accuracy
test_pred = model.predict(X_test)
test_acc = accuracy_score(Y_test, test_pred)
print("Testing Accuracy:", test_acc * 100)

# Confusion matrix + report
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, test_pred))
print("\nClassification Report:\n", classification_report(Y_test, test_pred))

# Prediction on new data
input_data = (59,1,1,140,221,0,1,164,1,0.0,2,0,2)

input_array = np.asarray(input_data)
input_reshaped = input_array.reshape(1, -1)

prediction = model.predict(input_reshaped)

if prediction[0] == 0:
    print("\nResult: The person does NOT have heart disease")
else:
    print("\nResult: The person HAS heart disease")
