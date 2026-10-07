from pathlib import Path
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from catboost import CatBoostClassifier, Pool

BASE = Path(__file__).resolve().parent

# -----------------------------
# Load model and reference data
# -----------------------------
MODEL = CatBoostClassifier()
MODEL.load_model(str(BASE / "model.cbm"))

CUSTOMERS = pd.read_csv(BASE / "data" / "customers.csv")
PRODUCTS = pd.read_csv(BASE / "data" / "products.csv")

FEATURE_NAMES = MODEL.feature_names_

CAT_COLS = [
    c for c in FEATURE_NAMES
    if c in [
        "customer_id",
        "sku",
        "sales_channel",
        "payment_mode",
        "is_gift",
        "source",
        "city",
        "state",
        "shield_member",
        "family",
        "model_name",
    ]
]


# -----------------------------
# Feature engineering
# -----------------------------
def make_features(df):
    x = df.copy()

    if "dt" in x:
        x = x.drop(columns=["dt"])

    x["order_placed_at"] = pd.to_datetime(x["order_placed_at"])

    x["order_hour"] = x.order_placed_at.dt.hour
    x["order_dow"] = x.order_placed_at.dt.dayofweek
    x["order_month"] = x.order_placed_at.dt.month
    x["order_day"] = x.order_placed_at.dt.day
    x["is_weekend"] = (x.order_dow >= 5).astype(int)

    x["customer_return_rate"] = (
        x.customer_prior_returns
        / x.customer_prior_orders.replace(0, np.nan)
    )
    x["customer_return_rate"] = x["customer_return_rate"].fillna(0)

    x["has_prior_orders"] = (
        x.customer_prior_orders > 0
    ).astype(int)

    x["default_pincode"] = (
        x.delivery_pincode == 0
    ).astype(int)

    x["value_per_unit"] = (
        x.order_value_inr
        / x.qty.replace(0, np.nan)
    )

    x = x.merge(
        CUSTOMERS,
        on="customer_id",
        how="left"
    )

    x = x.merge(
        PRODUCTS,
        on="sku",
        how="left"
    )

    x["signup_date"] = pd.to_datetime(x.signup_date)

    x["customer_tenure_days"] = (
        x.order_placed_at - x.signup_date
    ).dt.days.clip(lower=0)

    x["product_age_days"] = (
        x.order_placed_at
        - pd.to_datetime(x.launch_date)
    ).dt.days.clip(lower=0)

    x["price_ratio"] = (
        x.order_value_inr
        / (x.list_price_inr * x.qty).replace(0, np.nan)
    )

    note = x.delivery_note.fillna("").str.lower()

    for kw in [
        "call",
        "address",
        "office",
        "security",
        "gate",
        "gift",
        "leave",
        "weekday",
        "delivery",
        "floor",
        "door",
        "building",
    ]:
        x["note_" + kw] = note.str.contains(
            kw,
            regex=False
        ).astype(int)

    drop = [
        "returned",
        "last_service_event_type",
        "pickup_scheduled_at",
        "order_id",
        "order_placed_at",
        "signup_date",
        "launch_date",
        "delivery_note",
    ]

    x = x.drop(
        columns=[c for c in drop if c in x.columns]
    )

    for c in CAT_COLS:
        if c in x.columns:
            x[c] = (
                x[c]
                .fillna("MISSING")
                .astype(str)
            )

    return x[FEATURE_NAMES]


# -----------------------------
# Explainability
# -----------------------------
def explain(row):

    pool = Pool(
        row,
        cat_features=CAT_COLS
    )

    shap = MODEL.get_feature_importance(
        pool,
        type="ShapValues"
    )[0, :-1]

    pairs = sorted(
        zip(FEATURE_NAMES, shap),
        key=lambda z: abs(z[1]),
        reverse=True
    )

    pretty = {
        "payment_mode": "payment method",
        "promised_delivery_days": "promised delivery time",
        "customer_return_rate": "prior customer return rate",
        "shield_member": "Shield membership",
        "customer_prior_returns": "prior returns",
        "discount_pct": "discount",
        "sales_channel": "sales channel",
        "family": "product family",
        "price_ratio": "paid/list-price ratio",
        "default_pincode": "missing-address flag",
    }

    out = []

    for feature, value in pairs:

        if (
            feature in pretty
            and abs(value) > 0.03
        ):

            direction = (
                "increases"
                if value > 0
                else "reduces"
            )

            out.append(
                f"{pretty[feature]} "
                f"{direction} return risk"
            )

        if len(out) == 3:
            break

    return out


# -----------------------------
# Business decision layer
# -----------------------------
RETURN_COST_INR = 1150
CALL_COST_INR = 45
CALL_PREVENTION_RATE = 0.35

# Approximate economic break-even:
# 45 / (1150 * 0.35) ≈ 11.2%
INTERVENTION_THRESHOLD = (
    CALL_COST_INR
    / (RETURN_COST_INR * CALL_PREVENTION_RATE)
)


def risk_decision(probability):

    if probability >= 0.20:

        return (
            "high",
            "Priority review / possible hold",
            "red"
        )

    elif probability >= INTERVENTION_THRESHOLD:

        return (
            "medium",
            "Confirmation call before dispatch",
            "orange"
        )

    else:

        return (
            "low",
            "Normal dispatch",
            "green"
        )


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(
    title="Kestrel Returns Risk",
    version="1.1"
)


# -----------------------------
# Health endpoint
# -----------------------------
@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": "loaded",
        "model_type": "CatBoostClassifier"
    }


# -----------------------------
# Main dashboard
# -----------------------------
@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return """
<!doctype html>

<html>

<head>

<title>Kestrel Returns Risk</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: #f4f6f8;
    margin: 0;
    padding: 35px;
}

.container {
    max-width: 1000px;
    margin: auto;
}

.header {
    background: #111827;
    color: white;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.header h1 {
    margin: 0 0 8px 0;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

textarea {
    width: 100%;
    height: 300px;
    padding: 12px;
    box-sizing: border-box;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    font-family: monospace;
}

button {
    margin-top: 15px;
    padding: 12px 25px;
    border: none;
    border-radius: 8px;
    background: #2563eb;
    color: white;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    background: #1d4ed8;
}

.result {
    display: none;
}

.metric {
    display: inline-block;
    width: 30%;
    margin-right: 2%;
    padding: 18px;
    background: #f9fafb;
    border-radius: 10px;
    vertical-align: top;
}

.metric:last-child {
    margin-right: 0;
}

.metric-title {
    font-size: 13px;
    color: #6b7280;
}

.metric-value {
    font-size: 24px;
    font-weight: bold;
    margin-top: 8px;
}

.reasons {
    margin-top: 15px;
}

.reason {
    padding: 10px;
    margin: 7px 0;
    background: #f3f4f6;
    border-radius: 7px;
}

#out {
    white-space: pre-wrap;
    font-family: monospace;
}

.note {
    color: #6b7280;
    font-size: 13px;
}

</style>

</head>


<body>

<div class="container">

<div class="header">

<h1>Kestrel Home — Returns Risk Intelligence</h1>

<p>
Pre-dispatch return-risk scoring and operational recommendation
</p>

</div>


<div class="card">

<h2>Order Input</h2>

<textarea id="x">{
  "order_id": "DEMO",
  "order_placed_at": "2026-07-01 10:00",
  "customer_id": "KC105196",
  "sku": "KH-IC-03",
  "sales_channel": "web",
  "payment_mode": "prepaid_upi",
  "discount_pct": 13,
  "qty": 1,
  "order_value_inr": 3521.76,
  "promised_delivery_days": 7,
  "delivery_pincode": 440378,
  "is_gift": "N",
  "customer_prior_orders": 2,
  "customer_prior_returns": 1,
  "delivery_note": "Office address, weekdays only",
  "last_service_event_type": "NONE",
  "pickup_scheduled_at": null,
  "source": "crm"
}</textarea>

<br>

<button onclick="go()">
Predict Return Risk
</button>

</div>


<div class="card result" id="result">

<h2>Prediction</h2>

<div>

<div class="metric">

<div class="metric-title">
Return Probability
</div>

<div class="metric-value"
id="probability">
-
</div>

</div>


<div class="metric">

<div class="metric-title">
Risk Band
</div>

<div class="metric-value"
id="band">
-
</div>

</div>


<div class="metric">

<div class="metric-title">
Recommended Action
</div>

<div class="metric-value"
id="action">
-
</div>

</div>

</div>


<div class="reasons">

<h3>Why?</h3>

<div id="reasons"></div>

</div>


<p class="note">
The score supports operational prioritization; it is not a guarantee
that an order will or will not be returned.
</p>

</div>


<div class="card">

<h3>API Response</h3>

<pre id="out"></pre>

</div>

</div>


<script>

async function go() {

    try {

        const response = await fetch(
            '/predict',
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: document.getElementById('x').value
            }
        );

        const data = await response.json();

        document.getElementById(
            'result'
        ).style.display = 'block';

        document.getElementById(
            'probability'
        ).textContent =
            (data.return_probability * 100).toFixed(1) + '%';

        document.getElementById(
            'band'
        ).textContent =
            data.risk_band.toUpperCase();

        document.getElementById(
            'action'
        ).textContent =
            data.recommended_action;

        document.getElementById(
            'reasons'
        ).innerHTML =
            data.reasons
            .map(
                r => '<div class="reason">✓ ' + r + '</div>'
            )
            .join('');

        document.getElementById(
            'out'
        ).textContent =
            JSON.stringify(
                data,
                null,
                2
            );

    }

    catch (e) {

        document.getElementById(
            'out'
        ).textContent = e;

    }

}

</script>

</body>

</html>
"""


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(record: dict):

    row = make_features(
        pd.DataFrame([record])
    )

    probability = float(
        MODEL.predict_proba(row)[0, 1]
    )

    band, action, _ = risk_decision(
        probability
    )

    estimated_return_exposure = (
        probability * RETURN_COST_INR
    )

    return {

        "return_probability": round(
            probability,
            4
        ),

        "risk_band": band,

        "recommended_action": action,

        "estimated_return_cost_inr": round(
            estimated_return_exposure,
            2
        ),

        "intervention_threshold": round(
            INTERVENTION_THRESHOLD,
            4
        ),

        "reasons": explain(row)

    }