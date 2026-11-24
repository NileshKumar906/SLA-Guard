from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import pandas as pd,joblib
from sla_guard.config import ARTIFACTS
app=FastAPI(title="SLA-Guard API",description="Human-reviewed risk scoring research demo")
bundle=None
class Payload(BaseModel): features:dict
@app.get("/")
def root(): return {"status":"ready" if (ARTIFACTS/"model.joblib").exists() else "model_missing","human_review_required":True}
@app.post("/predict")
def predict(payload:Payload):
 global bundle
 if bundle is None:
  try: bundle=joblib.load(ARTIFACTS/"model.joblib")
  except: raise HTTPException(503,"Train model first")
 x=pd.DataFrame([{c:payload.features.get(c) for c in bundle["feature_columns"]}]);
 try: return {"probability_positive_class":float(bundle["model"].predict_proba(x)[0,1]),"threshold":bundle["threshold"],"model":bundle["model_name"],"human_review_required":True}
 except Exception as e: raise HTTPException(422,str(e))
