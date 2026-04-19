from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostRegressor
import json
import pandas as pd
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# --- CORS (чтобы работал frontend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- базовая директория ---
BASE_DIR = os.path.dirname(__file__)

# --- загрузка моделей ---
price_drop_model = CatBoostRegressor()
price_drop_model.load_model(
    os.path.join(BASE_DIR, "model/catboost_price_drop.cbm")
)

final_price_model = CatBoostRegressor()
final_price_model.load_model(
    os.path.join(BASE_DIR, "model/catboost_final_price.cbm")
)

# --- загрузка фичей ---
with open(os.path.join(BASE_DIR, "model/feature_columns.json")) as f:
    feature_columns = json.load(f)

with open(os.path.join(BASE_DIR, "model/final_feature_columns.json")) as f:
    final_feature_columns = json.load(f)

# --- входные данные ---
class InputData(BaseModel):
    customer_price_rub: float
    delivery_region: str
    trade_type: str
    electronic_trade_mode: Optional[str] = None


# --- тестовый endpoint (чтобы проверить, что сервер жив) ---
@app.get("/")
def root():
    return {"status": "ok"}


# --- основной endpoint ---
@app.post("/predict")
def predict(data: InputData):
    
    try:
        # вход → DataFrame
        df = pd.DataFrame([data.dict()])

        # создаём DataFrame с нужными колонками
        full_df = pd.DataFrame(columns=feature_columns)
        full_df.loc[0] = None

        # заполняем входные значения
        for col in df.columns:
            if col in feature_columns:
                full_df.loc[0, col] = df.loc[0, col]

        # заполняем пропуски
        full_df = full_df.fillna("missing")

        # приводим к строкам (важно для CatBoost)
        full_df = full_df.astype(str)

        # гарантируем порядок колонок
        full_df = full_df[feature_columns]

        # --- предсказание ---
        drop_pred = price_drop_model.predict(full_df)[0]
        drop_pred = max(float(drop_pred), 0)

        final_price = data.customer_price_rub * (1 - drop_pred)

        return {
            "predicted_drop_pct": float(drop_pred),
            "predicted_final_price": float(final_price)
        }

    except Exception as e:
        return {"error": str(e)}
