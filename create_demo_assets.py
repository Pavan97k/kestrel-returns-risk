from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# -----------------------------
# 1. Create synthetic customers
# -----------------------------
rng = np.random.default_rng(42)

customers = pd.DataFrame({
    "customer_id": [f"DEMO_C{i:04d}" for i in range(1, 101)],
    "signup_date": pd.date_range("2024-01-01", periods=100, freq="7D"),
    "prior_orders": rng.integers(0, 8, 100),
    "prior_returns": rng.integers(0, 3, 100),
})

customers["prior_returns"] = np.minimum(
    customers["prior_returns"],
    customers["prior_orders"]
)

customers.to_csv(DATA_DIR / "customers.csv", index=False)


# -----------------------------
# 2. Create synthetic products
# -----------------------------
products = pd.DataFrame({
    "sku": [f"DEMO-SKU-{i:03d}" for i in range(1, 21)],
    "list_price": rng.integers(500, 5000, 20),
    "launch_date": pd.date_range("2023-01-01", periods=20, freq="30D"),
})

products.to_csv(DATA_DIR / "products.csv", index=False)


# -----------------------------
# 3. Create synthetic training data
# -----------------------------
n = 1000

customer_ids = rng.choice(customers["customer_id"], n)
skus = rng.choice(products["sku"], n)

train = pd.DataFrame({
    "customer_id": customer_ids,
    "sku": skus,
    "order_value": rng.integers(500, 5000, n),
    "quantity": rng.integers(1, 4, n),
    "delivery_pincode": rng.choice(
        [560001, 560002, 560003, 0],
        n,
        p=[0.35, 0.30, 0.25, 0.10]
    ),
    "payment_method": rng.choice(
        ["UPI", "CARD", "COD"],
        n
    ),
    "order_hour": rng.integers(8, 22, n),
    "order_dow": rng.integers(0, 7, n),
    "is_weekend": rng.integers(0, 2, n),
    "customer_return_rate": rng.random(n),
    "has_prior_orders": rng.integers(0, 2, n),
    "value_per_unit": rng.uniform(300, 5000, n),
    "price_ratio": rng.uniform(0.5, 1.5, n),
})

# Synthetic target.
# This is only for making a reproducible demo model.
risk_signal = (
    1.5 * train["customer_return_rate"]
    + 0.5 * (train["payment_method"] == "COD")
    + 0.4 * (train["delivery_pincode"] == 0)
    + 0.3 * train["is_weekend"]
    + rng.normal(0, 0.8, n)
)

probability = 1 / (1 + np.exp(-risk_signal + 1.5))
train["returned"] = rng.binomial(1, probability)

# -----------------------------
# 4. Train demo CatBoost model
# -----------------------------
X = train.drop(columns=["returned"])
y = train["returned"]

categorical_columns = [
    X.columns.get_loc("customer_id"),
    X.columns.get_loc("sku"),
    X.columns.get_loc("payment_method"),
]

model = CatBoostClassifier(
    iterations=100,
    depth=5,
    learning_rate=0.05,
    loss_function="Logloss",
    random_seed=42,
    verbose=False,
)

model.fit(
    X,
    y,
    cat_features=categorical_columns
)

model.save_model(BASE_DIR / "model.cbm")

print("Demo assets created successfully.")
print(f"Model: {BASE_DIR / 'model.cbm'}")
print(f"Customers: {DATA_DIR / 'customers.csv'}")
print(f"Products: {DATA_DIR / 'products.csv'}")