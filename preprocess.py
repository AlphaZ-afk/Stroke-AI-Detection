import pandas as pd

def preprocess(df):
    df = df.copy()

    # Drop id
    if "id" in df.columns:
        df = df.drop("id", axis=1)

    # Smartly Handle Missing BMI
    if "bmi" in df.columns:
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    # Safely Encode Binary
    if "gender" in df.columns and df["gender"].dtype == "object":
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0, "Other": 0}).fillna(0)
    
    if "ever_married" in df.columns and df["ever_married"].dtype == "object":
        df["ever_married"] = df["ever_married"].map({"Yes": 1, "No": 0}).fillna(0)
        
    if "Residence_type" in df.columns and df["Residence_type"].dtype == "object":
        df["Residence_type"] = df["Residence_type"].map({"Urban": 1, "Rural": 0}).fillna(0)

    # One-hot encode correctly
    df = pd.get_dummies(df, columns=[c for c in ["work_type", "smoking_status"] if c in df.columns])

    # Convert object and booleans to float/int
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(float)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df