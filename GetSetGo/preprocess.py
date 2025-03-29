import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset df = pd.read_csv("data/Datasets.csv")
df = pd.read_csv(os.path.join("data", "Datasets.csv"))

# Display the first few rows
print("Original Dataset:")
print(df.head())

# Check for missing values
print("\nMissing Values Before Handling:")
print(df.isnull().sum())

# Handle missing values
df['GPA'] = df['GPA'].fillna(df['GPA'].mean())
df['AttendanceRate'] = df['AttendanceRate'].fillna(df['AttendanceRate'].median())

# Define Graduation Status based on Attendance and GPA
df['Graduation_Status'] = np.where((df['GPA'] >= 2.5) & (df['AttendanceRate'] >= 75), 1, 0)

# Remove any leading or trailing spaces from column names
df.columns = df.columns.str.strip()

# Drop unnecessary columns safely
columns_to_drop = ['Fid', 'Studentname']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Encode categorical variables (Gender, Major, Institution_Type) using Label Encoding
label_encoders = {}
categorical_columns = ['Gender', 'Major', 'Institution_Type']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Save encoders for future decoding

# Convert numerical columns to correct data types
df = df.astype({
    'Age': 'int32',
    'AttendanceRate': 'float32',
    'GPA': 'float32',
    'Year': 'int32'
})

# Save the cleaned dataset df.to_csv("data/cleaned_student_data.csv", index=False)
data_path = os.path.join("data", "cleaned_student_data.csv")
df.to_csv(data_path, index=False)

# Display the cleaned dataset
print("\nCleaned Dataset:")
print(df.head())

# Show Graduation Status distribution
print("\nGraduation Status Distribution:")
print(df['Graduation_Status'].value_counts())