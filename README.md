# project
# AI-Based Network Intrusion Detection System (NIDS)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# 1. Load Dataset
# -----------------------------
# Make sure 'nsl_kdd.csv' is in the same folder
data = pd.read_csv("nsl_kdd.csv")

print("Dataset Preview:")
print(data.head())


# -----------------------------
# 2. Encode Categorical Features
# -----------------------------
encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = encoder.fit_transform(data[column])


# -----------------------------
# 3. Separate Features & Target
# -----------------------------
# 'label' column should contain 0 = Normal, 1 = Attack
X = data.drop("label", axis=1)
y = data["label"]


# -----------------------------
# 4. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)


# -----------------------------
# 5. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# -----------------------------
# 6. Train ML Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
print("\nModel training completed successfully.")


# -----------------------------
# 7. Prediction
# -----------------------------
y_pred = model.predict(X_test)


# -----------------------------
# 8. Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------
# 9. Real-Time Detection Function
# -----------------------------
def detect_intrusion(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        return "Normal Traffic"
    else:
        return "⚠ Intrusion Detected"


# -----------------------------
# 10. Sample Test
# -----------------------------
sample_input = X_test[0]
result = detect_intrusion(sample_input)
print("\nReal-Time Detection Result:", result)
