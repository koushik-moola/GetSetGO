import numpy as np
import joblib
import os

# Define paths for saved models
BASE_DIR = os.path.dirname(__file__)  # flask_app directory
MODELS_DIR = os.path.join(BASE_DIR, "models")  # Corrected path

model_paths = {
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.pkl"),
    "Random Forest": os.path.join(MODELS_DIR, "random_forest.pkl"),
    "XGBoost": os.path.join(MODELS_DIR, "xgboost.pkl"),
    "SVM": os.path.join(MODELS_DIR, "svm.pkl"),
}

# Load models
models = {name: joblib.load(path) for name, path in model_paths.items() if os.path.exists(path)}


def ensemble_predict(data):
    """Predict graduation status using hard voting ensemble."""
    if not models:
        return {"error": "No models found"}

    predictions = []

    for model in models.values():
        try:
            y_pred = model.predict([data])
            predictions.append(y_pred[0])
        except Exception as e:
            return {"error": f"Model prediction failed: {e}"}

    # Hard Voting: Take majority vote
    final_prediction = np.bincount(predictions).argmax()
    return final_prediction
