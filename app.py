"""
app.py
=======
Student Marks Prediction Based on Study Hours - Flask Web Application

Loads the trained Linear Regression model produced by train_model.py and
serves a simple web interface where a user enters a number of study hours
and receives a predicted marks value.

Run with:
    python app.py

Then open:
    http://127.0.0.1:5000
"""

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Paths (relative, so this works on any machine / OS)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "marks_prediction_model.pkl")

MAX_STUDY_HOURS = 24  # a student cannot study more than 24 hours in a day

# ---------------------------------------------------------------------------
# Load the trained model once at startup (do NOT retrain on every request).
# If the model file is missing, the app still starts but the prediction page
# shows a clear message instead of crashing.
# ---------------------------------------------------------------------------
model = None
load_error = None

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    load_error = (
        "Trained model file was not found. "
        "Please run 'python train_model.py' first to train and save the model."
    )
except Exception as exc:  # noqa: BLE001
    load_error = f"An error occurred while loading the model: {exc}"


def interpret_marks(marks):
    """Return a simple performance category for the predicted marks."""
    if marks >= 90:
        return "Excellent", "success"
    elif marks >= 75:
        return "Very Good", "primary"
    elif marks >= 60:
        return "Good", "info"
    else:
        return "Needs Improvement", "warning"


def validate_study_hours(raw_value):
    """
    Validate the study hours input.
    Returns (value, error_message). error_message is None if valid.
    """
    raw_value = (raw_value or "").strip()

    if raw_value == "":
        return None, "Please enter the number of study hours."

    try:
        value = float(raw_value)
    except ValueError:
        return None, "Study hours must be a valid number."

    if value < 0:
        return None, "Study hours cannot be negative."

    if value > MAX_STUDY_HOURS:
        return None, f"Study hours cannot be more than {MAX_STUDY_HOURS} hours in a day."

    return value, None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", error=None, max_hours=MAX_STUDY_HOURS)

    # POST - handle form submission
    if load_error:
        return render_template("predict.html", error=load_error, max_hours=MAX_STUDY_HOURS)

    study_hours, error = validate_study_hours(request.form.get("study_hours"))

    if error:
        return render_template("predict.html", error=error, max_hours=MAX_STUDY_HOURS)

    try:
        input_df = pd.DataFrame([[study_hours]], columns=["Study_Hours"])
        prediction = model.predict(input_df)[0]
        prediction = float(np.clip(prediction, 0, 100))
        prediction = round(prediction, 2)

        category, badge_class = interpret_marks(prediction)

        return render_template(
            "result.html",
            study_hours=study_hours,
            prediction=prediction,
            category=category,
            badge_class=badge_class,
        )
    except Exception as exc:  # noqa: BLE001
        return render_template(
            "predict.html",
            error=f"Something went wrong while making the prediction: {exc}",
            max_hours=MAX_STUDY_HOURS,
        )


@app.errorhandler(404)
def page_not_found(_e):
    return render_template("about.html"), 404


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
