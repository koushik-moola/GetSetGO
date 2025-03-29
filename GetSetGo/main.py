import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the preprocessed dataset
df = pd.read_csv("data/cleaned_student_data.csv")

# Drop unnecessary columns (if not already dropped)
if "Fid" in df.columns:
    df.drop(columns=["Fid", "Studentname"], inplace=True)

# Convert categorical variables to numerical using One-Hot Encoding
categorical_features = ["Gender", "Major", "University", "Institution_Type"]
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Define features (X) and target variable (y)
X = df.drop(columns=["Graduation_Status"])  # Features
y = df["Graduation_Status"]  # Target variable (Pass/Fail)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42)
}

# Create directory to save models
MODEL_DIR = "flask_app/models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Train and evaluate models
best_model = None
best_accuracy = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    model_filename = os.path.join(MODEL_DIR, f"{name.lower().replace(' ', '_')}.pkl")
    joblib.dump(model, model_filename)
    print(f"✅ {name} model saved as {model_filename}")

    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save the best-performing model
best_model_path = os.path.join(MODEL_DIR, "best_graduation_predictor.pkl")
joblib.dump(best_model, best_model_path)
print(f"\n🏆 Best model saved as '{best_model_path}'")
