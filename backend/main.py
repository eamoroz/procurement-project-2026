from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostRegressor
import json
import pandas as pd

app = FastAPI()

# --- загрузка моделей ---
price_drop_model = CatBoostRegressor()
price_drop_model.load_model("backend/model/catboost_price_drop.cbm")

final_price_model = CatBoostRegressor()
final_price_model.load_model("backend/model/catboost_final_price.cbm")

# --- загрузка фичей ---
with open("backend/model/feature_columns.json") as f:
    feature_columns = json.load(f)

with open("backend/model/final_feature_columns.json") as f:
    final_feature_columns = json.load(f)

# --- входные данные ---
class InputData(BaseModel):
    customer_price_rub: float
    delivery_region: str
    trade_type: str
    electronic_trade_mode: str | None = None


# --- endpoint ---
@app.post("/predict")
def predict(data: InputData):
    
    # превращаем в DataFrame
    df = pd.DataFrame([data.dict()])
    
    # добавляем недостающие колонки
    for col in feature_columns:
        if col not in df.columns:
            df[col] = None
    
    df = df[feature_columns]
    
    # --- предсказание снижения ---
    drop_pred = price_drop_model.predict(df)[0]
    drop_pred = max(drop_pred, 0)
    
    # --- финальная цена через снижение ---
    final_price = data.customer_price_rub * (1 - drop_pred)
    
    return {
        "predicted_drop_pct": float(drop_pred),
        "predicted_final_price": float(final_price)
    }
