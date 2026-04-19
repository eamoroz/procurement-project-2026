from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostRegressor
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
        # 👉 просто используем вход как есть
        df = pd.DataFrame([data.dict()])

        # 👉 никаких missing, никаких 100 колонок
        drop_pred = price_drop_model.predict(df)[0]
        drop_pred = max(drop_pred, 0)

        final_price = data.customer_price_rub * (1 - drop_pred)

        return {
            "predicted_drop_pct": float(drop_pred),
            "predicted_final_price": float(final_price)
        }

    except Exception as e:
        return {"error": str(e)}
