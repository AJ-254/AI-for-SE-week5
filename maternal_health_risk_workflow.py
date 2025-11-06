"""
Assignment: Understanding the AI Development Workflow
Case Study: Maternal Health Risk Prediction
Author: Juliet Asiedu
Course: AI for Software Engineering
"""


# 1. Problem Definition
# Problem: Predict maternal health risk level (Low, Medium, or High)
# Objectives:
#   1. Use patient vital signs (Age, Blood Pressure, Heart Rate, etc.) to predict risk level.
#   2. Help healthcare workers prioritize high-risk patients.
#   3. Reduce maternal mortality by early detection of complications.
# Stakeholders: Pregnant women, healthcare professionals.
# KPI: Model accuracy and recall for high-risk prediction.


# 2. Data Collection & Preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
# Load dataset
data_path = r"C:\Users\Juliet Asiedu\Desktop\plp-africa\AI-specialization\AIweek5\Maternal health Risk Data Set.csv"
df = pd.read_csv(data_path)

print("Dataset Preview:")
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing values (if any) with column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Features and target
X = df.drop('RiskLevel', axis=1)
y = df['RiskLevel']

# Encode target if categorical
y = y.map({'low risk': 0, 'mid risk': 1, 'high risk': 2})

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 3. Model Development
# Choose Random Forest for interpretability and robustness
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=5, 
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'Medium', 'High'], 
            yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Maternal Health Risk')
plt.show()

# 5. Deployment Simulation
# Save model and scaler
joblib.dump(model, "maternal_risk_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Simulate reloading and making prediction
loaded_model = joblib.load("maternal_risk_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

# Sample input (Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate)
sample_data = np.array([[32, 130, 80, 8.5, 98.6, 75]])
sample_scaled = loaded_scaler.transform(sample_data)
sample_prediction = loaded_model.predict(sample_scaled)

risk_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
print("\nSample Prediction:", risk_labels[sample_prediction[0]])

print("\n✅ Workflow complete! Model trained, evaluated, and deployed successfully.")
