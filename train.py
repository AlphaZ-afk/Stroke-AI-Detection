import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from preprocess import preprocess

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = preprocess(df)

X = df.drop("stroke", axis=1)
y = df["stroke"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_scaled, y)

data = {
    "model": model,
    "scaler": scaler,
    "features": X.columns.tolist()
}
joblib.dump(data, "model.pkl")

print("Model trained with StandardScaler 💀🔥")