import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the dataset
file_path = "data/raw/admission.csv"
df = pd.read_csv(file_path)

# Serial No. is just a row index so it can be dropped
df.drop("Serial No.", axis=1, inplace=True)

# print(df.columns) # Debugging since the last column contains a space at the end

# Define features and target variable
X = df.drop("Chance of Admit ", axis=1)
y = df["Chance of Admit "]

# Split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Save the processed datasets
output_dir = "data/processed/"
os.makedirs(output_dir, exist_ok=True)

X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

print("Data preparation completed. Processed files saved in data/processed/")

