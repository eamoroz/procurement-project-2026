from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostRegressor
import json
import pandas as pd
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)

# --- модели ---
price_drop_model = CatBoostRegressor()
price_drop_model.load_model(
    os.path.join(BASE_DIR, "model/catboost_price_drop.cbm")
)

final_price_model = CatBoostRegressor()
final_price_model.load_model(
    os.path.join(BASE_DIR, "model/catboost_final_price.cbm")
)

# --- фичи ---
with open(os.path.join(BASE_DIR, "model/feature_columns.json")) as f:
    feature_columns = json.load(f)

with open(os.path.join(BASE_DIR, "model/final_feature_columns.json")) as f:
    final_feature_columns = json.load(f)

# --- вход ---
class InputData(BaseModel):
    customer_price_rub: float
    delivery_region: str
    trade_type: str
    electronic_trade_mode: Optional[str] = None


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])

        # --- ВАЖНО: как было у тебя ---
        full_df = pd.DataFrame([{col: None for col in feature_columns}])

        for col in df.columns:
            if col in full_df.columns:
                full_df.at[0, col] = df.at[0, col]

        # ❗ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ
        # только категориальные → строки
        for col in full_df.columns:
            if full_df[col].dtype == "object":
                full_df[col] = full_df[col].fillna("missing").astype(str)

        # --- предикт ---
        drop_pred = price_drop_model.predict(full_df)[0]
        drop_pred = max(drop_pred, 0)

        final_price = data.customer_price_rub * (1 - drop_pred)

        return {
            "predicted_drop_pct": float(drop_pred),
            "predicted_final_price": float(final_price)
        }

    except Exception as e:
        return {"error": str(e)}
