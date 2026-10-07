from pathlib import Path
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from catboost import CatBoostClassifier, Pool

BASE = Path(__file__).resolve().parent
MODEL = CatBoostClassifier()
MODEL.load_model(str(BASE / "model.cbm"))
CUSTOMERS = pd.read_csv(BASE / "data" / "customers.csv")
PRODUCTS = pd.read_csv(BASE / "data" / "products.csv")

FEATURE_NAMES = MODEL.feature_names_
CAT_COLS = [c for c in FEATURE_NAMES if c in [
    "customer_id","sku","sales_channel","payment_mode","is_gift","source",
    "city","state","shield_member","family","model_name"
]]

def make_features(df):
    x=df.copy()
    if "dt" in x: x=x.drop(columns=["dt"])
    x["order_placed_at"]=pd.to_datetime(x["order_placed_at"])
    x["order_hour"]=x.order_placed_at.dt.hour
    x["order_dow"]=x.order_placed_at.dt.dayofweek
    x["order_month"]=x.order_placed_at.dt.month
    x["order_day"]=x.order_placed_at.dt.day
    x["is_weekend"]=(x.order_dow>=5).astype(int)
    x["customer_return_rate"]=x.customer_prior_returns/(x.customer_prior_orders.replace(0,np.nan))
    x["customer_return_rate"]=x["customer_return_rate"].fillna(0)
    x["has_prior_orders"]=(x.customer_prior_orders>0).astype(int)
    x["default_pincode"]=(x.delivery_pincode==0).astype(int)
    x["value_per_unit"]=x.order_value_inr/x.qty.replace(0,np.nan)
    x=x.merge(CUSTOMERS,on="customer_id",how="left")
    x=x.merge(PRODUCTS,on="sku",how="left")
    x["signup_date"]=pd.to_datetime(x.signup_date)
    x["customer_tenure_days"]=(x.order_placed_at-x.signup_date).dt.days.clip(lower=0)
    x["product_age_days"]=(x.order_placed_at-pd.to_datetime(x.launch_date)).dt.days.clip(lower=0)
    x["price_ratio"]=x.order_value_inr/(x.list_price_inr*x.qty).replace(0,np.nan)
    note=x.delivery_note.fillna("").str.lower()
    for kw in ["call","address","office","security","gate","gift","leave","weekday","delivery","floor","door","building"]:
        x["note_"+kw]=note.str.contains(kw,regex=False).astype(int)
    drop=["returned","last_service_event_type","pickup_scheduled_at","order_id","order_placed_at",
          "signup_date","launch_date","delivery_note"]
    x=x.drop(columns=[c for c in drop if c in x.columns])
    for c in CAT_COLS:
        if c in x.columns: x[c]=x[c].fillna("MISSING").astype(str)
    return x[FEATURE_NAMES]

def explain(row):
    pool=Pool(row,cat_features=CAT_COLS)
    shap=MODEL.get_feature_importance(pool,type="ShapValues")[0,:-1]
    pairs=sorted(zip(FEATURE_NAMES,shap),key=lambda z:abs(z[1]),reverse=True)
    pretty={
        "payment_mode":"payment method","promised_delivery_days":"promised delivery time",
        "customer_return_rate":"prior customer return rate","shield_member":"Shield membership",
        "customer_prior_returns":"prior returns","discount_pct":"discount","sales_channel":"sales channel",
        "family":"product family","price_ratio":"paid/list-price ratio","default_pincode":"missing-address flag"
    }
    out=[]
    for f,v in pairs:
        if f in pretty and abs(v)>0.03:
            direction="increases" if v>0 else "reduces"
            out.append(f"{pretty[f]} {direction} return risk")
        if len(out)==3: break
    return out

app=FastAPI(title="Kestrel Returns Risk")

@app.get("/",response_class=HTMLResponse)
def home():
    return """<!doctype html><html><head><title>Kestrel Returns Risk</title>
<style>body{font-family:Arial;max-width:900px;margin:40px auto;padding:0 20px}textarea{width:100%;height:320px}button{padding:10px 18px}pre{background:#f5f5f5;padding:16px;white-space:pre-wrap}</style>
</head><body><h1>Kestrel Returns Risk</h1>
<p>Paste one order JSON record and get a pre-dispatch risk score.</p>
<textarea id="x">{"order_id":"DEMO","order_placed_at":"2026-07-01 10:00","customer_id":"KC105196","sku":"KH-IC-03","sales_channel":"web","payment_mode":"prepaid_upi","discount_pct":13,"qty":1,"order_value_inr":3521.76,"promised_delivery_days":7,"delivery_pincode":440378,"is_gift":"N","customer_prior_orders":2,"customer_prior_returns":1,"delivery_note":"Office address, weekdays only","last_service_event_type":"NONE","pickup_scheduled_at":null,"source":"crm"}</textarea>
<br><br><button onclick="go()">Predict</button><pre id="out"></pre>
<script>async function go(){try{let r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:document.getElementById('x').value});document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2)}catch(e){document.getElementById('out').textContent=e}}</script>
</body></html>"""

@app.post("/predict")
def predict(record: dict):
    row=make_features(pd.DataFrame([record]))
    p=float(MODEL.predict_proba(row)[0,1])
    if p>=0.40: band="high"
    elif p>=0.20: band="medium"
    else: band="low"
    return {"return_probability":round(p,4),"risk_band":band,"reasons":explain(row)}
