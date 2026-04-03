import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# Load data
data = pd.read_csv("cardekho_dataset.csv")
data.columns = data.columns.str.strip().str.lower()

data.rename(columns={
    "vehicle_age": "age",
    "fuel_type": "fuel",
    "transmission_type": "transmission",
    "selling_price": "price",
    "seller_type": "owner"
}, inplace=True)

data["engine"] = pd.to_numeric(data["engine"], errors="coerce")
data["max_power"] = pd.to_numeric(data["max_power"], errors="coerce")

data.dropna(inplace=True)

data = data[data["price"] < 10000000]  # your current filter

data = data[[
    "age", "mileage", "fuel", "transmission",
    "brand", "model", "engine", "max_power", "seats", "owner",
    "price"
]]

# One-hot encoding
data = pd.get_dummies(data, columns=["fuel", "transmission", "brand", "model", "owner"], drop_first=True)

X = data.drop("price", axis=1)
y = data["price"]

# ✅ LOG TRANSFORM
y_log = np.log1p(y)  # log(price + 1) to avoid log(0)

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train_log)

# ✅ Evaluate — convert predictions back to actual rupees
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)   # reverse the log transform
y_test = np.expm1(y_test_log)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("=" * 40)
print("       MODEL ACCURACY REPORT")
print("=" * 40)
print(f"  R² Score     : {r2:.4f}  ({r2*100:.2f}%)")
print(f"  MAE          : ₹ {mae:,.0f}")
print("=" * 40)

# ✅ CROSS VALIDATION (5-fold)
cv_scores = cross_val_score(model, X, y_log, cv=5, scoring="r2")
cv_mae = cross_val_score(model, X, y_log, cv=5, scoring="neg_mean_absolute_error")

print("\n  CROSS VALIDATION (5-Fold)")
print("=" * 40)
print(f"  R² per fold  : {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Avg R²       : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Avg MAE      : ₹ {-cv_mae.mean():,.0f}")
print("=" * 40)

# Feature importance
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1][:10]

print("\n  TOP 10 IMPORTANT FEATURES")
print("=" * 40)
for rank, i in enumerate(indices, 1):
    bar = "█" * int(importances[i] * 200)
    print(f"  {rank:>2}. {feature_names[i]:<30} {importances[i]:.4f}  {bar}")
print("=" * 40)

# Retrain on full data before saving
model.fit(X, y_log)
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("\n  Model saved successfully!")