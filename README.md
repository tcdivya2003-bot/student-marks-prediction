# Student Marks Prediction Based on Study Hours Using Machine Learning

## Introduction

This is a beginner-friendly Machine Learning project that focuses on one
simple, well-known idea: students who study for more hours tend to score
higher marks. The project builds a Machine Learning model that learns this
relationship from historical data and predicts a student's expected marks
for a given number of study hours.

Unlike a full student-performance system, this project deliberately keeps
things focused on a single input and a single output so that the core
Machine Learning concept — **Linear Regression** — can be understood
clearly, demonstrated confidently, and explained in a viva without
distraction from other factors.

```text
Study Hours  -->  Machine Learning Model  -->  Predicted Marks
```

## Problem Statement

Students spend different amounts of time studying and obtain different
marks as a result. This project develops a Machine Learning model that
learns from historical data containing only:

- Study Hours
- Marks

and predicts the expected marks for a new number of study hours. It
intentionally does **not** use other attributes such as attendance,
previous marks, assignment marks, sleep hours, or internal marks, so the
project stays focused on demonstrating the core study-hours-to-marks
relationship.

## Objective

Build and train a Linear Regression model that:

1. Learns the relationship between study hours and marks from sample data.
2. Predicts marks for any new study-hours value entered by a user.
3. Is deployed through a simple, easy-to-use Flask web application.

## Technologies Used

**Language:** Python 3.11+

**Machine Learning:** Pandas, NumPy, Scikit-learn, Matplotlib

**Backend:** Flask

**Frontend:** HTML5, CSS3, JavaScript, Bootstrap 5

**Model Storage:** Joblib

**Development Environment:** VS Code

## Dataset

- **File:** `dataset/student_marks.csv`
- **Columns:** `Study_Hours`, `Marks`
- **Number of records:** 100 rows (98 remain after removing 2 duplicate rows during preprocessing)
- **Source:** This is a **synthetically generated sample/educational dataset**
  (created by `generate_dataset.py`). It does **not** come from real students.
  It was generated to reflect a realistic, mostly-linear relationship between
  study hours and marks, with a small amount of random variation added so
  the data resembles real-world observations rather than a perfect straight
  line.

If you have access to a real dataset with the same two columns, you can
replace `dataset/student_marks.csv` with it and everything else in the
project will continue to work unchanged.

## Machine Learning Algorithm

This project uses **Linear Regression**, chosen because:

- The input (study hours) is numerical.
- The output (marks) is numerical.
- The goal is to understand and quantify a simple relationship between one
  feature and one target.
- Linear Regression produces a simple, interpretable equation rather than a
  "black box" model.

The learned relationship takes the form:

```text
y = mx + c
```

where `y` is the predicted marks, `x` is the study hours, `m` is the
learned slope/coefficient, and `c` is the learned intercept.

## Methodology

```text
Dataset
   |
Data Preprocessing (missing values, duplicates, type checks)
   |
Exploratory Data Analysis (scatter plot, distribution)
   |
Train-Test Split (80% / 20%)
   |
Linear Regression Training
   |
Prediction on Test Data
   |
Evaluation (MAE, MSE, RMSE, R2)
   |
Flask Deployment
```

## Evaluation Metrics

- **MAE (Mean Absolute Error):** average of the absolute differences between actual and predicted marks.
- **MSE (Mean Squared Error):** average of the squared differences; penalizes larger errors more.
- **RMSE (Root Mean Squared Error):** square root of MSE, expressed in the same unit as marks.
- **R² Score:** proportion of the variance in marks explained by study hours (1.0 = perfect fit).

### Actual Results (from running `train_model.py` on the included dataset)

```text
Learned relationship: Marks = 6.4673 * Study_Hours + 26.9448

MAE  : 2.779
MSE  : 15.151
RMSE : 3.892
R2   : 0.972
```

These are real, calculated values from an actual training run — not
invented figures. If you regenerate the dataset (`python generate_dataset.py`)
or retrain the model, these numbers may change slightly since the sample
data includes randomness.

### Actual vs Predicted (sample from the test set)

```text
Study Hours | Actual Marks | Predicted Marks
---------------------------------------------
0.6         | 31.17        | 30.83
2.6         | 43.07        | 43.76
4.8         | 57.99        | 57.99
7.5         | 77.18        | 75.45
8.9         | 85.89        | 84.50
9.8         | 91.53        | 90.32
11.4        | 100.00       | 100.67
```

The full comparison table for every test-set row is generated at
`actual_vs_predicted.csv` each time you run `train_model.py`.

## Installation (Windows PowerShell)

1. Extract the project ZIP file and open the folder in VS Code.
2. Open a PowerShell terminal inside the project folder and run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell's execution policy blocks the activation script, you have two
options:

**Option A** — use the Command Prompt-style batch activator instead:

```powershell
venv\Scripts\activate.bat
```

**Option B** — run the project without activating the virtual environment,
by calling the venv's Python directly:

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe train_model.py
venv\Scripts\python.exe app.py
```

## Train the Model

The dataset is already included, but you can regenerate the sample dataset
at any time:

```powershell
python generate_dataset.py
```

Train the Linear Regression model:

```powershell
python train_model.py
```

This will:
- Load and inspect the dataset
- Clean the data (missing values, duplicates)
- Generate EDA plots into the `plots/` folder
- Train the Linear Regression model
- Print MAE, MSE, RMSE, and R² on the test set
- Print and save the Actual vs Predicted comparison table
- Save the trained model to `model/marks_prediction_model.pkl`

## Run the Application

```powershell
python app.py
```

Then open your browser and go to:

```text
http://127.0.0.1:5000
```

The Flask app loads the already-trained model from disk — it does **not**
retrain the model each time the server starts.

## Example Prediction

**Input:**

```text
Study Hours = 6
```

**Output (produced by actually running the trained model included in this project):**

```text
Predicted Marks: 65.75 / 100
Performance Category: Good
```

> Note: If you regenerate the dataset before retraining, your exact
> predicted value for 6 study hours may differ slightly, since the sample
> dataset includes random variation. Re-run `train_model.py` to see the
> value produced by your own trained model.

## Result Interpretation

| Marks Range | Category |
|---|---|
| 90 – 100 | Excellent |
| 75 – 89 | Very Good |
| 60 – 74 | Good |
| Below 60 | Needs Improvement |

**This is a machine learning based estimate and not a guaranteed
examination result.**

## Project Structure

```text
Study_Hours_Marks/
│
├── dataset/
│   └── student_marks.csv
│
├── model/
│   └── marks_prediction_model.pkl
│
├── plots/
│   ├── study_hours_vs_marks.png
│   ├── regression_line.png
│   └── marks_distribution.png
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
├── templates/
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   └── about.html
│
├── generate_dataset.py
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Future Enhancements

These are possible improvements for later — the current project deliberately
stays focused on **Study Hours → Marks** only:

- Adding more student-related features (attendance, previous marks, etc.)
- Using multiple regression with several input features
- Comparing different ML algorithms (Decision Tree, Random Forest, etc.)
- Adding a student login system
- Saving prediction history per user
- Adding a database (SQLite/PostgreSQL)
- Creating a student performance dashboard
- Deploying the application online (e.g., Render, Railway, PythonAnywhere)

---

## Viva Preparation — Questions and Answers

**1. What is Machine Learning?**
Machine Learning is a branch of artificial intelligence where a system
learns patterns from data and makes predictions without being explicitly
programmed with fixed rules.

**2. What is supervised learning?**
Supervised learning is a type of machine learning where the model learns
from labeled data — input values paired with known correct outputs — so it
can predict outputs for new inputs.

**3. What is regression?**
Regression is a type of supervised learning used when the target variable
is a continuous numerical value, rather than a category or class.

**4. Why is this project a regression problem?**
Because the target variable, `Marks`, is a continuous number (e.g., 65.75),
not a category like "Pass" or "Fail".

**5. What is Linear Regression?**
Linear Regression is an algorithm that models the relationship between an
independent variable and a dependent variable by fitting a straight line
of the form `y = mx + c`.

**6. Why did you select Linear Regression?**
Because both study hours and marks are numerical, the relationship between
them is expected to be approximately linear, and Linear Regression gives a
simple, easily interpretable model — ideal for demonstrating this concept.

**7. What is the independent variable?**
`Study_Hours` — the input feature the model uses to make its prediction.

**8. What is the dependent variable?**
`Marks` — the value that depends on (and is predicted from) study hours.

**9. What is the target variable?**
The target variable is `Marks`, the value the model is trained to predict.

**10. What is train-test splitting?**
It is the process of dividing the dataset into two parts: a training set
used to fit the model, and a testing set used to check how well the model
performs on data it hasn't seen before.

**11. Why use 80% training and 20% testing?**
This is a common, practical split that gives the model enough data to learn
from (80%) while still keeping a reasonably sized, unseen set (20%) to
fairly evaluate its performance.

**12. What is MAE?**
Mean Absolute Error — the average of the absolute differences between
actual and predicted marks. It tells you, on average, how many marks the
prediction is off by.

**13. What is MSE?**
Mean Squared Error — the average of the squared differences between actual
and predicted marks. Squaring makes larger errors count more heavily.

**14. What is RMSE?**
Root Mean Squared Error — the square root of MSE. It is expressed in the
same unit as marks, which makes it easier to interpret than MSE.

**15. What is R²?**
R² (coefficient of determination) measures how much of the variation in
marks is explained by study hours. In this project, R² = 0.972, meaning
study hours explain about 97.2% of the variation in marks in the test data.

**16. What is overfitting?**
Overfitting happens when a model learns the training data (including its
noise) too closely and as a result performs poorly on new, unseen data.

**17. What is underfitting?**
Underfitting happens when a model is too simple to capture the real pattern
in the data, leading to poor performance on both training and test data.

**18. Why use Pandas?**
Pandas is used to load, inspect, and manipulate the tabular CSV dataset —
for example, checking for missing values, duplicates, and computing summary
statistics.

**19. Why use Scikit-learn?**
Scikit-learn provides ready-to-use implementations of machine learning
algorithms like Linear Regression, along with tools for splitting data and
calculating evaluation metrics.

**20. Why use Flask?**
Flask is a lightweight Python web framework that makes it easy to build a
simple web interface around a trained Python ML model without needing a
separate backend language.

**21. How is the trained model saved?**
The trained Linear Regression model is serialized to disk using
`joblib.dump()` inside `train_model.py`, producing the file
`model/marks_prediction_model.pkl`.

**22. How does Flask use the trained model?**
When the Flask app (`app.py`) starts, it loads the saved model once using
`joblib.load()`. When a user submits study hours through the form, Flask
validates the input, passes it to the model's `predict()` method, and
displays the result — without ever retraining the model.

**23. What are the limitations of this project?**
The dataset is a synthetically generated sample rather than real student
records, only one feature (study hours) is used, and real-world marks are
influenced by many other factors not captured by this simple model.

**24. What are possible future improvements?**
Adding more input features (attendance, previous marks, etc.), comparing
multiple algorithms, adding a database to store prediction history, adding
user login, building a performance dashboard, and deploying the app online.

---

**Disclaimer:** This system provides a machine-learning-based estimate for
educational and demonstration purposes only, based solely on study hours.
It does not guarantee actual examination results.
