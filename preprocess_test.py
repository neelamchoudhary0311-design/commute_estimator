import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load the dataset created in Step 2
df = pd.read_csv("commute_data.csv")

# 2. Separate input features (X) and target output (y)
X = df.drop(columns=["fare"])
y = df["fare"]

# 3. Split into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Define which columns are numeric and which column is categorical
num_cols = ["distance_km", "hour_of_day", "is_peak", "traffic_index"]
cat_cols = ["mode"]

# 5. Build the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# 6. Test the preprocessor on training and testing features
X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)

print("Original X_train shape:", X_train.shape)
print("Prepared X_train shape:", X_train_prepared.shape)
print("Original X_test shape: ", X_test.shape)
print("Prepared X_test shape: ", X_test_prepared.shape)
print("Step 3 preprocessing check passed!")