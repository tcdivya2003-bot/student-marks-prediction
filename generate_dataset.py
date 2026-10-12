"""
generate_dataset.py
--------------------
Generates a sample/educational dataset for the "Student Marks Prediction
Based on Study Hours" project.

IMPORTANT: This is a synthetically generated sample dataset created for
educational demonstration purposes. It does NOT come from real students.
It models a realistic, mostly-linear relationship between study hours and
marks, with a small amount of natural random variation added so the data
looks like real-world observations rather than a perfect straight line.

Run this once to create dataset/student_marks.csv.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 100  # number of sample records

# Study hours: reasonable daily study range for a student, 0.5 to 12 hours
study_hours = np.round(np.random.uniform(0.5, 12, N), 1)

# Marks = base linear relationship + small random noise, clipped to 0-100
# Roughly: every extra hour of study adds ~7-8 marks, with a base score
marks = 25 + 6.8 * study_hours + np.random.normal(0, 4, N)
marks = np.clip(marks, 0, 100)
marks = np.round(marks, 2)

df = pd.DataFrame({
    "Study_Hours": study_hours,
    "Marks": marks
})

# Sort by study hours for readability
df = df.sort_values(by="Study_Hours").reset_index(drop=True)

df.to_csv("dataset/student_marks.csv", index=False)
print("Sample dataset generated:", df.shape)
print(df.head(10))
