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
    trading_platform: Optional[str] = None
    delivery_city: Optional[str] = None


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])
        full_df = df.reindex(columns=feature_columns)

        # 🔥 получаем реальные категориальные фичи из модели
        cat_feature_indices = price_drop_model.get_cat_feature_indices()
        cat_cols = [feature_columns[i] for i in cat_feature_indices]

        for col in full_df.columns:
            if col in cat_cols:
                # категориальные → строки
                full_df[col] = full_df[col].astype(str)
                full_df[col] = full_df[col].fillna("unknown")
            else:
                # числовые → строго числа
                full_df[col] = pd.to_numeric(full_df[col], errors="coerce")
                full_df[col] = full_df[col].fillna(0)

        drop_pred = price_drop_model.predict(full_df)[0]
        drop_pred = max(drop_pred, 0)

        final_price = data.customer_price_rub * (1 - drop_pred)

        return {
            "predicted_drop_pct": float(drop_pred),
            "predicted_final_price": float(final_price)
        }

    except Exception as e:
        return {"error": str(e)}
