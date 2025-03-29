import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import pandas as pd
from ml.visualization import generate_visualizations
from ml.model import ensemble_predict

app = Flask(__name__)
app.secret_key = "super secretkey"  # Change this for security
app.permanent_session_lifetime = 1800  # 30 minutes session timeout

# Get absolute paths for datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../data/Datasets.csv")  # Update path
NEW_DATA_PATH = os.path.join(BASE_DIR, "../data/cleaned_student_data.csv")  # Update path

# Load datasets with error handling
try:
    old_data = pd.read_csv(DATASET_PATH)  # Raw dataset
except FileNotFoundError:
    old_data = None  # Handle missing dataset case

try:
    new_data = pd.read_csv(NEW_DATA_PATH)  # Preprocessed dataset
except FileNotFoundError:
    new_data = None  # Handle missing dataset case

# Dummy users (Replace this with a database later)
USERS = {"batch2": "7410"}


@app.route("/")
def index():
    """Redirect to log instead of showing the dashboard immediately."""
    if "user" in session:
        return redirect(url_for("visualizations"))  # If logged in, go to visualizations
    return redirect(url_for("login"))  # Otherwise, go to log

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Handle signup logic here (e.g., saving data to the database)
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Add your validation and processing code here

        # Redirect to login page or any other page after signup
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login."""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username in USERS and USERS[username] == password:
            session["user"] = username
            session.permanent = True  # Keep session active
            return redirect(url_for("visualizations"))  # Redirect to visualizations after login
        return render_template("login.html", error="Invalid credentials!")
    return render_template("login.html")


@app.route("/logout")
def logout():
    """Handle user logout."""
    session.pop("user", None)
    return redirect(url_for("index"))  # Redirect to login after logout


@app.route("/visualizations")
def visualizations():
    """Render a page with visualizations of dataset and model results (login required)."""
    if "user" not in session:
        return redirect(url_for("index"))

    if old_data is None or new_data is None:
        flash("Error: Datasets not found. Please upload them.", "error")
        return redirect(url_for("dashboard"))  # Redirect to an appropriate page

    try:
        plot_filenames = generate_visualizations()  # Generate and return plot filenames
    except Exception as e:
        flash(f"Error generating visualizations: {str(e)}", "error")
        return redirect(url_for("dashboard"))

    return render_template(
        "visualizations.html",
        old_data_shape=old_data.shape,
        new_data_shape=new_data.shape,
        plot_filenames=plot_filenames,  # Pass filenames for rendering in the template
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Predict graduation status using ensemble voting."""
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json()
    prediction = ensemble_predict(data)
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
