# Kestrel Home Returns Risk

## Run locally

Python 3.11–3.13 recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

No paid API key is required.

## Endpoint

`POST /predict`

Send one raw order record using the columns from `test_unlabelled.csv` (excluding `returned`).

The response contains:
- `return_probability`
- `risk_band`
- `reasons`

## Modeling decisions

- Exact partner-feed re-import duplicates were removed from training.
- Time features, customer history, product attributes and simple delivery-note indicators were engineered.
- `last_service_event_type` and `pickup_scheduled_at` were excluded because they can contain post-outcome information.
- The primary validation metric is ROC-AUC.
- Validation uses forward chronological monthly holdouts.

## Files

- `predictions.csv` — required submission scores
- `model.cbm` — trained CatBoost model
- `app.py` — FastAPI service + browser screen
- `data/customers.csv`, `data/products.csv` — reference tables
- `validation_results.json` — validation evidence
- `memo_to_ritu.md` — one-page business memo
- `submission-form.md` — copy/paste submission answers
