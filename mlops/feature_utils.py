import numpy as np
import pandas as pd

TARGET = "default.payment.next.month"
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
PAYMENT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
EXCLUDED_FROM_MODEL = ["ID", "SEX", "EDUCATION", "MARRIAGE", TARGET]

RAW_MODEL_FEATURES = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]


def validate_raw_features(data: pd.DataFrame) -> None:
    missing = sorted(set(RAW_MODEL_FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the exact features used during training and prediction."""
    validate_raw_features(data)
    output = data[RAW_MODEL_FEATURES].copy()
    safe_limit = output["LIMIT_BAL"].replace(0, np.nan)

    output["AVERAGE_BILL"] = output[BILL_COLS].mean(axis=1)
    output["LATEST_UTILISATION"] = output["BILL_AMT1"] / safe_limit
    output["AVERAGE_UTILISATION"] = output["AVERAGE_BILL"] / safe_limit

    for month in range(1, 7):
        output[f"UTILISATION_{month}"] = output[f"BILL_AMT{month}"] / safe_limit

    return output.replace([np.inf, -np.inf], np.nan).fillna(0)
