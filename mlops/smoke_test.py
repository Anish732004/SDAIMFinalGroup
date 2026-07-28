import sys
import pandas as pd
sys.path.append("mlops")
from feature_utils import RAW_MODEL_FEATURES, engineer_features

sample = pd.DataFrame([{column: 0 for column in RAW_MODEL_FEATURES}])
sample["LIMIT_BAL"] = 100000
sample["AGE"] = 35
result = engineer_features(sample)
assert not result.empty
assert "AVERAGE_BILL" in result.columns
assert "LATEST_UTILISATION" in result.columns
assert "AVERAGE_UTILISATION" in result.columns
print("Smoke test passed.")
