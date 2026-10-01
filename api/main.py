from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="EV Logistics Forecaster API")

# Load the trained model into server memory
MODEL_PATH = "api/ev_model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class ScenarioRequest(BaseModel):
    scenario_type: str
    months_to_forecast: int = 6

@app.post("/predict")
def predict_ev_sales(req: ScenarioRequest):
    if not model:
        return {"error": "Model missing. Run train.py first."}
        
    # The historical data had 24 months (indices 0 to 23). 
    # We forecast the next N future months starting at index 24.
    future_indices = np.arange(24, 24 + req.months_to_forecast).reshape(-1, 1)
    baseline_forecast = model.predict(future_indices)
    
    # Apply Qualitative Supply Chain Modifiers
    if "Boost" in req.scenario_type:
        final_forecast = baseline_forecast * 1.15
    elif "Shortage" in req.scenario_type:
        final_forecast = baseline_forecast * 0.90
    else:
        final_forecast = baseline_forecast

    return {
        "months": [f"Month {i+1}" for i in range(req.months_to_forecast)],
        "forecast": [int(x) for x in final_forecast]
    }
