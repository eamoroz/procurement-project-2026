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

# --- загрузка моделей ---
price_drop_model = CatBoostRegressor()
price_drop_model.load_model(
    os.path.join(BASE_DIR, "model/catboost_price_drop.cbm")
)

# --- загрузка фичей ---
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

@app.post("/predict")
def predict(data: InputData):
    
    df = pd.DataFrame([data.dict()])
    
    # создаём полный DataFrame
    full_df = pd.DataFrame([{col: None for col in feature_columns}])
    
    # заполняем известные значения
    for col in df.columns:
        if col in full_df.columns:
            full_df.at[0, col] = df.at[0, col]
    
    # ВСЁ приводим к строкам (важно для CatBoost)
    for col in full_df.columns:
        full_df[col] = full_df[col].astype(str).fillna("missing")
    
    # --- предсказание ---
    drop_pred = price_drop_model.predict(full_df)[0]
    drop_pred = max(drop_pred, 0)
    
    final_price = data.customer_price_rub * (1 - drop_pred)
    
    return {
        "predicted_drop_pct": float(drop_pred),
        "predicted_final_price": float(final_price)
    }

    except Exception as e:
        return {"error": str(e)}
