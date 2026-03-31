import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

HISTORY_FILE = "medical_history.csv"

def init_db():
    if not os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(columns=[
            "Date", "Risk Score", "Risk Level", "Glucose", "BMI", "Face Asymmetry", "Speech Score"
        ])
        df.to_csv(HISTORY_FILE, index=False)

def save_record(risk_score, risk_level, glucose, bmi, face_score, speech_score):
    init_db()
    df = pd.read_csv(HISTORY_FILE)
    
    new_record = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Risk Score": round(risk_score, 4),
        "Risk Level": risk_level,
        "Glucose": round(glucose, 2),
        "BMI": round(bmi, 2),
        "Face Asymmetry": round(face_score, 4),
        "Speech Score": round(speech_score, 2)
    }
    
    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

def get_history():
    init_db()
    df = pd.read_csv(HISTORY_FILE)
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def get_trend_indicator():
    df = get_history()
    if len(df) < 2:
        return "Stable ➖"
    
    # Get last 3-5 days
    recent = df.sort_values("Date").tail(5)
    
    if len(recent) >= 2:
        diff = recent.iloc[-1]["Risk Score"] - recent.iloc[-2]["Risk Score"]
        if diff > 0.05:
            return "Worsening 📉"
        elif diff < -0.05:
            return "Improving 📈"
        else:
            return "Stable ➖"
            
    return "Stable ➖"

def plot_risk_trend():
    df = get_history()
    if df.empty:
        return None
    fig = px.line(df, x="Date", y="Risk Score", title="Risk Score Trend Over Time",
                  markers=True, color_discrete_sequence=["#ff4b4b"])
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_metrics_comparison():
    df = get_history()
    if df.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Glucose"], mode='lines+markers', name='Glucose'))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BMI"], mode='lines+markers', name='BMI'))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Face Asymmetry"] * 100, mode='lines+markers', name='Face Asym. (Scaled)'))
    
    fig.update_layout(title="Health Metrics Comparison", template="plotly_dark", 
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def get_weekly_summary():
    df = get_history()
    if df.empty:
        return {"avg_risk": 0, "best_day": "N/A", "worst_day": "N/A"}
        
    recent = df.sort_values("Date").tail(7)
    avg_risk = recent["Risk Score"].mean()
    best_day = recent.loc[recent["Risk Score"].idxmin()]["Date"].strftime("%Y-%m-%d")
    worst_day = recent.loc[recent["Risk Score"].idxmax()]["Date"].strftime("%Y-%m-%d")
    
    return {
        "avg_risk": avg_risk,
        "best_day": best_day,
        "worst_day": worst_day
    }
