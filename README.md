# Kestrel Home — Returns Risk

Pre-dispatch return-risk scoring service for Kestrel Home.

## What it does

The service accepts a single order record and returns:

- Return probability
- Risk band
- Recommended action
- Estimated return-cost exposure
- Human-readable reasons

The model is intended for risk ranking and intervention, not as a guarantee that an order will or will not be returned.

## Run locally

Python 3.11–3.13 recommended.

### 1. Create and activate a virtual environment

```bash
python -m venv .venv

Windows:
.venv\Scripts\activate

macOS/Linux:
source .venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Generate synthetic demo assets
python create_demo_assets.py

This creates a synthetic CatBoost model and synthetic customer/product reference data so the public repository can be run without exposing Kestrel's private operational data.
4. Start the service
uvicorn app:app --host 0.0.0.0 --port 8000

Open:
http://localhost:8000
Health check:
http://localhost:8000/health
No paid API key is required.
Endpoint
POST /predict
Send one order record using the expected input fields.
The response contains:
- return_probability
- risk_band
- recommended_action
- estimated_return_cost_inr
- intervention_threshold
- reasons
Modeling decisions
- Exact partner-feed re-import duplicates were removed from training.
- Time features, customer history, product attributes and simple delivery-note indicators were engineered.
- last_service_event_type and pickup_scheduled_at were excluded because they can contain post-outcome information.
- The primary validation metric is ROC-AUC.
- Validation uses forward chronological monthly holdouts.
- The intervention threshold is economically motivated using the confirmation-call cost and estimated return cost.
Validation
Validation results are provided in validation_results.json.
The model is evaluated primarily as a ranking system rather than using the client's requested 95% accuracy as the main metric.
Files
- app.py — FastAPI service and browser interface
- create_demo_assets.py — creates synthetic assets for public reproducibility
- requirements.txt — Python dependencies
- validation_results.json — validation evidence
- README.md — project documentation
Privacy
The public repository contains synthetic demo assets only.
Kestrel customer/order data, production model artifacts, predictions and internal operational documents are intentionally excluded from the public repository.


### Why we're changing your README

Your old README says:

> `model.cbm — trained CatBoost model`

and

> `data/customers.csv`, `data/products.csv`

But those **aren't actually in the GitHub repository** because they're ignored/private.

That could confuse the evaluator.

The new README instead tells them:

```text
clone repo
   ↓
pip install
   ↓
python create_demo_assets.py
   ↓
synthetic model/data created
   ↓
uvicorn
   ↓
working application

