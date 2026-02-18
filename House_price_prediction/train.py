import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Load Dataset
df = pd.read_csv("C:\\Users\\nikhi\\OneDrive\\Desktop\\House_price_prediction\\dataset\\Housing.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# 2. Split Data
X = df.drop("price", axis=1)
y = df["price"]

categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 3. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

# 4. Create Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# 5. Train
pipeline.fit(X_train, y_train)

# 6. Evaluate

y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("\nModel R2 Score:", r2)


# 7. Save FULL Pipeline

if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(pipeline, "models/house_price_model.pkl")

print("\nModel saved successfully.")
