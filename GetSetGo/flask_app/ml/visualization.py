import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Get flask_app directory (one level up from ml/)
STATIC_DIR = os.path.join(BASE_DIR, "static", "visualizations")  # Save images in static/visualizations/
MODEL_DIR = os.path.join(BASE_DIR, "models")  # Save trained models in flask_app/models/
DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "cleaned_student_data.csv")  # Adjust dataset path inside data/

# Ensure static/visualizations directory exists
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def plot_pie_chart(data, column, title, filename):
    """Generates and saves a pie chart."""
    plt.figure(figsize=(6, 6))
    data[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=["lightblue", "orange"])
    plt.title(title)
    plt.ylabel("")
    plt.savefig(os.path.join(STATIC_DIR, filename))
    plt.close()


def plot_bar_chart(data, column, title, xlabel, ylabel, filename):
    """Generates and saves a bar chart."""
    plt.figure(figsize=(8, 5))
    sns.countplot(x=column, data=data, hue=column, palette="pastel", legend=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(STATIC_DIR, filename))
    plt.close()



def plot_confusion_matrix(y_true, y_pred, filename):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(STATIC_DIR, filename))
    plt.close()


def plot_histogram(predictions, title, filename):
    """Generates and saves a histogram of prediction probabilities."""
    plt.figure(figsize=(7, 5))
    sns.histplot(predictions, bins=10, kde=True, color="skyblue")
    plt.title(title)
    plt.xlabel("Probability of Passing")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(STATIC_DIR, filename))
    plt.close()


def generate_visualizations():
    """Generates and saves visualizations for the dataset and model performance."""
    print("Generating visualizations...")

    # Load dataset
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        return

    df = pd.read_csv(DATASET_PATH)
    df.drop(columns=[col for col in ["Fid", "Studentname"] if col in df.columns], inplace=True, errors="ignore")

    categorical_features = ["Gender", "Major", "University", "Institution_Type"]
    df = pd.get_dummies(df, columns=[col for col in categorical_features if col in df.columns], drop_first=True)

    if "Graduation_Status" not in df.columns:
        print("Error: 'Graduation_Status' column missing in dataset!")
        return

    X = df.drop(columns=["Graduation_Status"])
    y = df["Graduation_Status"]

    model_paths = {
        "Logistic Regression": os.path.join(MODEL_DIR, "logistic_regression.pkl"),
        "Random Forest": os.path.join(MODEL_DIR, "random_forest.pkl"),
        "XGBoost": os.path.join(MODEL_DIR, "xgboost.pkl"),
        "SVM": os.path.join(MODEL_DIR, "svm.pkl"),
    }

    models = {name: joblib.load(path) for name, path in model_paths.items() if os.path.exists(path)}

    if not models:
        print("Error: No valid models found!")
        return

    predictions = {}

    for name, model in models.items():
        try:
            y_pred = model.predict(X)
            predictions[name] = y_pred
            accuracy = accuracy_score(y, y_pred)
            print(f"{name} Accuracy: {accuracy * 100:.2f}%")

            plot_confusion_matrix(y, y_pred, f"{name.lower().replace(' ', '_')}_confusion_matrix.png")

            prediction_df = pd.DataFrame({"Predicted Graduation Status": y_pred})
            plot_bar_chart(prediction_df, "Predicted Graduation Status", f"{name} Predictions Distribution",
                           "Graduation Status", "Count", f"{name.lower().replace(' ', '_')}_bar_chart.png")

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]
                plot_histogram(y_prob, f"{name} Prediction Probabilities",
                               f"{name.lower().replace(' ', '_')}_probability_histogram.png")

        except Exception as e:
            print(f"Error while predicting with {name}: {e}")

    if predictions:
        final_predictions = np.array(list(predictions.values())).T
        final_y_pred = [np.bincount(row).argmax() for row in final_predictions if len(row) > 0]
        if final_y_pred:
            plot_confusion_matrix(y, final_y_pred, "ensemble_confusion_matrix.png")

    plot_pie_chart(df, "Graduation_Status", "Pass vs Fail Distribution", "pass_fail_pie_chart.png")
    if "Gender_Male" in df.columns:
        plot_bar_chart(df, "Gender_Male", "Gender Distribution", "Gender", "Count", "gender_bar_chart.png")

    print("Visualizations saved in the visualizations folder in static folder.")


if __name__ == "__main__":
    generate_visualizations()
