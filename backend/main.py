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

# --- модель ---
model = CatBoostRegressor()
model.load_model(os.path.join(BASE_DIR, "model/catboost_price_drop.cbm"))

# --- фичи ---
with open(os.path.join(BASE_DIR, "model/feature_columns.json")) as f:
    feature_columns = json.load(f)


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
        input_dict = data.dict()

        # создаём полный df
        full_df = pd.DataFrame([{col: None for col in feature_columns}])

        # заполняем вход
        for col, value in input_dict.items():
            if col in full_df.columns:
                full_df.at[0, col] = value

        # 🔥 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ
        for col in full_df.columns:
            val = full_df.at[0, col]

            if val is None:
                full_df.at[0, col] = "missing"
            elif isinstance(val, (int, float)):
                # числа оставляем числами
                full_df.at[0, col] = val
            else:
                # всё остальное → строка
                full_df.at[0, col] = str(val)

        # 👉 отдельно приводим DataFrame
        full_df = full_df.apply(pd.to_numeric, errors="ignore")

        # предсказание
        pred = model.predict(full_df)[0]
        pred = max(pred, 0)

        final_price = data.customer_price_rub * (1 - pred)

        return {
            "predicted_drop_pct": float(pred),
            "predicted_final_price": float(final_price)
        }

    except Exception as e:
        return {"error": str(e)}
