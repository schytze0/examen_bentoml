import pandas as pd
import os
import joblib
import bentoml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the processed datasets
data_path = "data/processed/"
X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"))
y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"))

# Convert DFs to 1D array
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

print(f"Model Performance:\nR2 Score: {r2:.4f}")

# Save model to BentoML Model Store
# bentoml.sklearn.save_model("lr_model", model, metadata={"r2": r2})
