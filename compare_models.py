import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load dataset and create train/test split
df = pd.read_csv("commute_data.csv")
X = df.drop(columns=["fare"])
y = df["fare"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Define preprocessing for numeric and categorical columns
num_cols = ["distance_km", "hour_of_day", "is_peak", "traffic_index"]
cat_cols = ["mode"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# 3. Define candidate models (linear baseline vs. ensemble models)
models = {
    "Ridge (Baseline)": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

# 4. Train each pipeline and evaluate on the test set
results = []
for name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "MAE (₹)": round(mae, 2),
        "RMSE (₹)": round(rmse, 2),
        "R2": round(r2, 4),
    })

results_df = pd.DataFrame(results).sort_values("MAE (₹)")
print("=== Model Comparison on Test Set ===")
print(results_df.to_string(index=False))
print("\nBest model so far:", results_df.iloc[0]["Model"])