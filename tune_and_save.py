import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load dataset and split into train/test sets
df = pd.read_csv("commute_data.csv")
X = df.drop(columns=["fare"])
y = df["fare"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Build preprocessing step
num_cols = ["distance_km", "hour_of_day", "is_peak", "traffic_index"]
cat_cols = ["mode"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# 3. Create pipeline with Gradient Boosting Regressor
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42)),
    ]
)

# 4. Define hyperparameter grid to search
param_grid = {
    "model__n_estimators": [100, 150],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 4],
}

# 5. Run 4-fold cross-validation grid search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=4,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# 6. Evaluate the best tuned pipeline on the test set
preds = best_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = root_mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("Best Hyperparameters:", grid_search.best_params_)
print(f"Tuned Test MAE : ₹{mae:.2f}")
print(f"Tuned Test RMSE: ₹{rmse:.2f}")
print(f"Tuned Test R2  : {r2:.4f}")

# 7. Save the complete trained pipeline to a .pkl file
model_filename = "commute_fare_model.pkl"
joblib.dump(best_model, model_filename)
print(f"\nSaved best model to '{model_filename}'!")

# 8. Quick sanity check: reload the .pkl file and predict one sample trip
loaded_model = joblib.load(model_filename)
sample_trip = pd.DataFrame([
    {
        "distance_km": 10.0,
        "hour_of_day": 9,
        "is_peak": 1,
        "traffic_index": 1.6,
        "mode": "Auto",
    }
])
sample_pred = float(loaded_model.predict(sample_trip)[0])
print(f"Sample reloaded prediction (10 km Auto at 9 AM): ₹{sample_pred:.2f}")