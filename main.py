from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostRegressor
import json
import pandas as pd
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import os
import traceback

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

    level: Optional[int] = 0
    industry_scope: Optional[str] = None
    trade_type: str
    trading_platform: Optional[str] = None
    electronic_trade_mode: Optional[str] = None
    national_regime: Optional[str] = None
    
    delivery_region: str
    delivery_city: Optional[str] = None

    publication_name: Optional[str] = None
    
    bid_security_rub: Optional[float] = 0
    bid_security_pct: Optional[float] = 0
    contract_security_rub: Optional[float] = 0
    contract_security_pct: Optional[float] = 0
    
    bank_treasury_support: Optional[int] = 0

    has_purchase_code: Optional[int] = 0
    
    num_participants: Optional[int] = 0
    
    publication_datetime: Optional[str] = None
    applications_deadline_datetime: Optional[str] = None
    applications_start_datetime: Optional[str] = None
    trading_end_datetime: Optional[str] = None

    is_electronic: Optional[int] = 0
    has_bid_security: Optional[int] = 0
    has_contract_security: Optional[int] = 0
    national_regime_flag: Optional[int] = 0


@app.post("/predict")
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])

        # --- обработка даты ---
        if data.publication_datetime:
            dt = pd.to_datetime(data.publication_datetime)
        
            df["publication_month"] = int(dt.month)
            df["publication_weekday"] = int(dt.weekday())
            df["publication_hour"] = int(dt.hour)
        else:
            df["publication_month"] = 0
            df["publication_weekday"] = 0
            df["publication_hour"] = 0

        # --- timestamp фичи ---
        if data.publication_datetime:
            df["publication_datetime_ts"] = pd.to_datetime(
                data.publication_datetime
            ).timestamp()
        else:
            df["publication_datetime_ts"] = 0

        if data.applications_deadline_datetime:
            df["applications_deadline_datetime_ts"] = pd.to_datetime(
                data.applications_deadline_datetime
            ).timestamp()
        else:
            df["applications_deadline_datetime_ts"] = 0

        if data.applications_start_datetime:
            df["applications_start_datetime_ts"] = pd.to_datetime(
                data.applications_start_datetime
            ).timestamp()
        else:
            df["applications_start_datetime_ts"] = 0

        if data.trading_end_datetime:
            df["trading_end_datetime_ts"] = pd.to_datetime(
                data.trading_end_datetime
            ).timestamp()
        else:
            df["trading_end_datetime_ts"] = 0
        
        full_df = df.reindex(columns=feature_columns)

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
        print(traceback.format_exc())
        return {"error": str(e)}
