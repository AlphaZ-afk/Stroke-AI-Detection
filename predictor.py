import joblib
import pandas as pd
from preprocess import preprocess

try:
    data = joblib.load("model.pkl")
    if isinstance(data, dict):
        model = data.get("model")
        scaler = data.get("scaler")
        features = data.get("features")
    else:
        # Fallback for old model structure temporarily
        model = data
        scaler = None
        features = model.feature_names_in_
except Exception as e:
    model, scaler, features = None, None, None

def predict(data_dict):
    df = pd.DataFrame([data_dict])
    df = preprocess(df)

    # Reindex columns based on training features
    if features is not None:
        df = df.reindex(columns=features, fill_value=0)

    if scaler:
        df_scaled = scaler.transform(df)
    else:
        df_scaled = df

    prob = model.predict_proba(df_scaled)[0][1]
    
    return prob