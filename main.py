from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostRegressor, CatBoostClassifier
import json
import numpy as np
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

dumping_model = CatBoostClassifier()
dumping_model.load_model(
    os.path.join(BASE_DIR, "model/catboost_dumping_classifier.cbm")
)

# --- фичи ---
with open(os.path.join(BASE_DIR, "model/feature_columns.json")) as f:
    feature_columns = json.load(f)

with open(os.path.join(BASE_DIR, "model/cls_feature_columns.json")) as f:
    cls_feature_columns = json.load(f)


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
    
    bank_treasury_support: Optional[str] = None

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


def build_features(data: InputData, model, feature_cols):
    df = pd.DataFrame([data.dict()])

    # --- derived features ---

    df["has_bid_security"] = int(
        data.bid_security_rub > 0
        or data.bid_security_pct > 0
    )
    
    df["has_contract_security"] = int(
        data.contract_security_rub > 0
        or data.contract_security_pct > 0
    )
    
    df["customer_price_log1p"] = float(
        np.log1p(data.customer_price_rub)
    )

    # --- обработка publication_datetime ---
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
    
    datetime_columns = {
        "publication_datetime": "publication_datetime_ts",
        "applications_deadline_datetime": "applications_deadline_datetime_ts",
        "applications_start_datetime": "applications_start_datetime_ts",
        "trading_end_datetime": "trading_end_datetime_ts",
    }
    
    parsed_dates = {}
    
    for source_col, target_col in datetime_columns.items():
    
        value = getattr(data, source_col)
    
        if value:
    
            parsed_dt = pd.to_datetime(value)
    
            parsed_dates[source_col] = parsed_dt
    
            df[target_col] = parsed_dt.timestamp()
    
        else:
    
            parsed_dates[source_col] = None
    
            df[target_col] = 0
    
    
    # --- derived datetime features ---
    
    publication_dt = parsed_dates["publication_datetime"]
    
    deadline_dt = parsed_dates["applications_deadline_datetime"]
    
    start_dt = parsed_dates["applications_start_datetime"]
    
    trading_end_dt = parsed_dates["trading_end_datetime"]
    
    
    # days_to_deadline
    
    if publication_dt and deadline_dt:
    
        df["days_to_deadline"] = (
            (deadline_dt - publication_dt)
            .total_seconds() / 86400.0
        )
    
    else:
    
        df["days_to_deadline"] = 0
    
    
    # days_from_start_to_deadline
    
    if start_dt and deadline_dt:
    
        df["days_from_start_to_deadline"] = (
            (deadline_dt - start_dt)
            .total_seconds() / 86400.0
        )
    
    else:
    
        df["days_from_start_to_deadline"] = 0
    
    
    # days_to_trading_end
    
    if publication_dt and trading_end_dt:
    
        df["days_to_trading_end"] = (
            (trading_end_dt - publication_dt)
            .total_seconds() / 86400.0
        )
    
    else:
    
        df["days_to_trading_end"] = 0

    # --- приводим к нужным колонкам ---
    full_df = df.reindex(columns=feature_cols)

    # --- catboost categorical columns ---
    cat_feature_indices = model.get_cat_feature_indices()
    cat_cols = [feature_cols[i] for i in cat_feature_indices]

    # --- типизация ---
    for col in full_df.columns:

        if col in cat_cols:
            full_df[col] = full_df[col].astype(str)
            full_df[col] = full_df[col].fillna("unknown")

        else:
            full_df[col] = pd.to_numeric(
                full_df[col],
                errors="coerce"
            )

            full_df[col] = full_df[col].fillna(0)

    return full_df


@app.post("/predict")
def predict(data: InputData):

    try:
        full_df = build_features(data, price_drop_model, feature_columns)

        drop_pred = price_drop_model.predict(full_df)[0]

        drop_pred = max(drop_pred, 0)

        final_price = data.customer_price_rub * (1 - drop_pred)

        return {
            "predicted_drop_pct": float(drop_pred),
            "predicted_final_price": float(final_price)
        }

    except Exception as e:
        print(traceback.format_exc())

        return {
            "error": str(e)
        }

@app.post("/predict_dumping")
def predict_dumping(data: InputData):

    try:
        full_df = build_features(data, dumping_model, cls_feature_columns)

        dumping_proba = dumping_model.predict_proba(full_df)[0][1]

        dumping_pred = dumping_model.predict(full_df)[0]

        return {
            "dumping_probability": float(dumping_proba),
            "is_dumping_predicted": int(dumping_pred)
        }

    except Exception as e:

        print(traceback.format_exc())

        return {
            "error": str(e)
        }
