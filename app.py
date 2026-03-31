import streamlit as st
import numpy as np
import cv2
import time
from datetime import datetime

from predictor import predict
from face import analyze_face, draw_face_box
from speech import analyze_speech_live
from grok_ai import get_recommendation
import history

# ================= UI CONFIG =================
st.set_page_config(page_title="Stroke AI Pro", layout="wide", initial_sidebar_state="collapsed")

# ================= CUSTOM CSS (NEON SYNTHWAVE) =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
}
h1, h2, h3, h4, h5, h6, .st-emotion-cache-10trnc1 {
    font-family: 'Orbitron', sans-serif;
    color: #e879f9 !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    text-shadow: 0 0 15px rgba(232, 121, 249, 0.8);
}

/* Advanced Live Animated Deep Background */
.stApp, [data-testid="stAppViewContainer"] {
    background: linear-gradient(45deg, #050510, #15092b, #0d0620, #050510) !important;
    background-size: 400% 400% !important;
    animation: gradientBG 15s ease infinite !important;
    color: #e2d5f8;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Base Top Header Fix */
[data-testid="stHeader"] {
    background: transparent !important;
    background-color: transparent !important;
}

/* Sidebar Theming & Animations */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #050510, #15092b) !important;
    background-size: 200% 200% !important;
    animation: gradientBG 10s ease infinite !important;
    border-right: 2px solid #d946ef !important;
    box-shadow: inset -5px 0 20px rgba(217, 70, 239, 0.2), 5px 0 20px rgba(217, 70, 239, 0.3) !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
}

[data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
    color: #e2d5f8 !important;
}

/* Animated Neon Edge for Sidebar */
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    right: 0;
    top: 0;
    width: 3px;
    height: 100%;
    background: #d946ef;
    box-shadow: 0 0 10px #d946ef, 0 0 20px #8b5cf6;
    animation: linePulse 2s infinite alternate;
    z-index: 999;
}

@keyframes linePulse {
    0% { opacity: 0.3; box-shadow: 0 0 5px #d946ef; width: 1px; }
    100% { opacity: 1; box-shadow: 0 0 15px #d946ef, 0 0 30px #8b5cf6; width: 3px; }
}

/* Animated Scanning Grid */
.cyber-grid {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: -2;
    background: 
        linear-gradient(rgba(168, 85, 247, 0.15) 1px, transparent 1px),
        linear-gradient(90deg, rgba(168, 85, 247, 0.15) 1px, transparent 1px);
    background-size: 50px 50px;
    animation: grid-scan 10s linear infinite;
    perspective: 1000px;
}

@keyframes grid-scan {
    0% { transform: translateY(0); }
    100% { transform: translateY(50px); }
}

/* Pulsing Neon Orbs */
.neural-cores {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    z-index: -1;
    overflow: hidden;
    pointer-events: none;
    mix-blend-mode: screen;
}
.core {
    position: absolute;
    border-radius: 50%;
    filter: blur(60px);
    animation: pulse-core 8s infinite alternate ease-in-out;
}
.core:nth-child(1) { background: #d946ef; width: 300px; height: 300px; top: -50px; left: -50px; }
.core:nth-child(2) { background: #3b82f6; width: 400px; height: 400px; bottom: -100px; right: 10%; animation-delay: -3s; }
.core:nth-child(3) { background: #8b5cf6; width: 250px; height: 250px; top: 50%; left: 40%; animation-delay: -6s; }

@keyframes pulse-core {
    0% { transform: scale(0.8); opacity: 0.15; }
    100% { transform: scale(1.3); opacity: 0.5; }
}

/* Cybernetic Glass Cards with Active Scanner */
.glass-card {
    background: rgba(13, 10, 31, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(217, 70, 239, 0.3);
    border-top: 1px solid #d946ef;
    border-bottom: 1px solid #06b6d4;
    box-shadow: 0 0 30px rgba(139, 92, 246, 0.15), inset 0 0 20px rgba(139, 92, 246, 0.05);
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 25px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: -100%; width: 50%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(217, 70, 239, 0.15), transparent);
    animation: scanner 6s linear infinite;
    pointer-events: none;
}

@keyframes scanner {
    0% { left: -100%; }
    100% { left: 200%; }
}

.glass-card:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 0 40px rgba(217, 70, 239, 0.3), inset 0 0 30px rgba(6, 182, 212, 0.2);
    border: 1px solid #d946ef;
}

/* Neon Title */
.glow-text {
    text-align: center;
    color: #ffffff;
    font-weight: 900;
    font-size: 4rem;
    margin-bottom: 20px;
    text-transform: uppercase;
    text-shadow: 0 0 10px rgba(217, 70, 239, 0.5);
}

/* HUD Style Inputs */
div[data-baseweb="select"] > div, input[type="text"], input[type="number"] {
    background: rgba(15, 12, 41, 0.8) !important;
    border: 1px solid #6d28d9 !important;
    border-left: 4px solid #d946ef !important;
    color: #e2d5f8 !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
}
div[data-baseweb="select"] > div:hover, input[type="text"]:focus, input[type="number"]:focus {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.4) !important;
    background: rgba(30, 27, 75, 0.9) !important;
}

/* Sci-Fi Buttons */
.stButton>button {
    background: linear-gradient(90deg, #9333ea, #db2777);
    color: #ffffff;
    border-radius: 20px;
    padding: 12px 30px;
    font-weight: 700;
    font-size: 18px;
    border: 1px solid #f472b6;
    box-shadow: 0 0 15px rgba(219, 39, 119, 0.4), inset 0 0 10px rgba(255, 255, 255, 0.1);
    text-transform: uppercase;
    letter-spacing: 2px;
    transition: all 0.3s ease;
    font-family: 'Orbitron', sans-serif;
    position: relative;
    overflow: hidden;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #c026d3, #a21caf);
    color: #ffffff;
    box-shadow: 0 0 30px #d946ef, inset 0 0 20px rgba(255,255,255,0.3);
    transform: scale(1.05);
}

/* Warning Shimmer Animation */
.shimmer {
    background: rgba(30, 27, 75, 0.8);
    background-image: linear-gradient(to right, rgba(30, 27, 75, 0.8) 0%, rgba(217, 70, 239, 0.4) 20%, rgba(30, 27, 75, 0.8) 40%, rgba(30, 27, 75, 0.8) 100%);
    background-repeat: no-repeat;
    background-size: 800px 100%;
    animation: placeholderShimmer 1.5s infinite linear;
    border-radius: 8px;
    border-left: 5px solid #d946ef;
    height: 60px;
    margin: 20px 0;
}
@keyframes placeholderShimmer {
    0% { background-position: -468px 0; }
    100% { background-position: 468px 0; }
}

/* Circular HUD Meter Definition */
.circular-chart {
  display: block;
  margin: 10px auto;
  max-width: 80%;
  max-height: 250px;
  filter: drop-shadow(0 0 20px rgba(217, 70, 239, 0.6));
}
.circle-bg {
  fill: none;
  stroke: rgba(139, 92, 246, 0.2);
  stroke-width: 3.8;
}
.circle {
  fill: none;
  stroke-width: 2.8;
  stroke-linecap: round;
  animation: svgProgress 1.5s ease-out forwards;
}
@keyframes svgProgress {
  0% { stroke-dasharray: 0, 100; }
}
.percentage {
  fill: #fff;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.5em;
  text-anchor: middle;
}
.risk-label {
  font-family: 'Orbitron', sans-serif;
  font-size: 0.16em;
  text-anchor: middle;
  font-weight: 700;
  letter-spacing: 2px;
}
</style>
<div class="cyber-grid"></div>
<div class="neural-cores">
    <div class="core"></div>
    <div class="core"></div>
    <div class="core"></div>
</div>
""", unsafe_allow_html=True)

def get_circular_meter(percentage, risk_text, color_hex):
    # dasharray syntax: stroke-dasharray="percentage, 100"
    return f"""<svg viewBox="0 0 36 36" class="circular-chart">
  <path class="circle-bg"
    d="M18 2.0845
      a 15.9155 15.9155 0 0 1 0 31.831
      a 15.9155 15.9155 0 0 1 0 -31.831"
  />
  <path class="circle"
    stroke="{color_hex}"
    stroke-dasharray="{percentage}, 100"
    d="M18 2.0845
      a 15.9155 15.9155 0 0 1 0 31.831
      a 15.9155 15.9155 0 0 1 0 -31.831"
  />
  <text x="18" y="16.5" class="percentage">{percentage:.1f}%</text>
  <text x="18" y="21.5" class="risk-label" fill="{color_hex}">{risk_text.upper()}</text>
</svg>"""

st.markdown('<h1 class="glow-text">🧠 Stroke AI Pro</h1>', unsafe_allow_html=True)

lang = st.sidebar.radio("Language / भाषा", ["English", "Hindi"])
translations = {
    "English": {
        "dashboard": "🚀 Dashboard",
        "analytics": "📊 Analytics & History",
        "patient_info": "📋 Patient Information",
        "face_analysis": "📸 Face Analysis",
        "speech_analysis": "🎤 Speech Analysis",
        "analyze_btn": "⚡ Analyze Risk with AI"
    },
    "Hindi": {
        "dashboard": "🚀 डैशबोर्ड",
        "analytics": "📊 एनालिटिक्स और इतिहास",
        "patient_info": "📋 मरीज की जानकारी",
        "face_analysis": "📸 चेहरा विश्लेषण",
        "speech_analysis": "🎤 भाषण विश्लेषण",
        "analyze_btn": "⚡ एआई से जोखिम विश्लेषण"
    }
}
t = translations[lang]

if 'face_score' not in st.session_state:
    st.session_state.face_score = 0.0

if 'speech_score' not in st.session_state:
    st.session_state.speech_score = 0.0

trend = history.get_trend_indicator()
if trend == "Worsening 📉":
    st.warning("⚠️ Risk increasing compared to previous days. Please review actionable recommendations.")
elif trend == "Improving 📈":
    st.success("💪 Great progress! Your risk score is reducing. Keep it up!")

tab1, tab2 = st.tabs([t["dashboard"], t["analytics"]])

with tab1:
    st.markdown(f'<div class="glass-card"><h3>{t["patient_info"]}</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", 1, 100, 45)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
    
    with col2:
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence = st.selectbox("Residence", ["Urban", "Rural"])
        glucose = st.number_input("Glucose Level", 50.0, 300.0, 90.0)
        bmi = st.number_input("BMI", 10.0, 60.0, 22.0)
        smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    st.markdown("---")
    
    col_face, col_speech = st.columns(2)
    
    with col_face:
        st.markdown(f"### {t['face_analysis']}")
        img = st.camera_input("Capture Face")
        if img:
            bytes_data = np.asarray(bytearray(img.read()), dtype=np.uint8)
            frame = cv2.imdecode(bytes_data, 1)
            frame_with_box = draw_face_box(frame.copy())
            st.image(frame_with_box, channels="BGR", caption="Face Detected")
            st.session_state.face_score = analyze_face(frame)
        
        st.info(f"👉 Face Asymmetry Score: {st.session_state.face_score:.4f}")

    with col_speech:
        st.markdown(f"### {t['speech_analysis']}")
        duration = st.slider("Recording Duration (sec)", 3, 10, 5)
        if st.button("🎙️ Record Audio Use Device Mic"):
            with st.spinner("Recording... Speak naturally."):
                prog_bar = st.progress(0)
                for i in range(100):
                    time.sleep(duration / 100)
                    prog_bar.progress(i + 1)
                st.session_state.speech_score = analyze_speech_live(duration)
        
        st.info(f"👉 Speech Risk Score: {st.session_state.speech_score:.4f}")

    st.markdown("---")
    
    if st.button(t["analyze_btn"]):
        placeholder = st.empty()
        placeholder.markdown("""
        <div style="text-align: center;">
            <div class="shimmer"></div>
            <h3 style="color:#10b981; margin-top:10px;">⚡ AI is analyzing your vitals...</h3>
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(2) # Show animation
        
        data = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence,
            "avg_glucose_level": glucose,
            "bmi": bmi,
            "smoking_status": smoking
        }
        
        # Feature Validation Log output in UI
        st.markdown(f"***🔍 Validated Features:** BMI: **{bmi}** | Glucose: **{glucose}** | Face Asym: **{st.session_state.face_score:.1f}/100** | Speech Risk: **{st.session_state.speech_score:.1f}/100***")

        # Deterministic Risk Calculation = weighted_sum(face, speech, glucose, BMI, conditions)
        health_risk = 0.0
        if age > 50: health_risk += (age - 50) * 0.5
        if glucose > 140: health_risk += (glucose - 140) * 0.2
        if bmi > 25: health_risk += (bmi - 25) * 0.8
        if hypertension == 1: health_risk += 15.0
        if heart_disease == 1: health_risk += 20.0
        if smoking in ["smokes", "formerly smoked"]: health_risk += 10.0
        
        health_risk = min(health_risk, 100.0) 
        
        face_risk = st.session_state.face_score
        speech_risk = st.session_state.speech_score
        
        print(f"\\n[DEBUG PIPELINE] Health Risk: {health_risk:.1f}/100 | Face: {face_risk:.1f}/100 | Speech: {speech_risk:.1f}/100")
        
        # Weighted Fusion Logic
        w_health, w_face, w_speech = 0.4, 0.3, 0.3
        
        if face_risk == 0 and speech_risk == 0:
            w_health, w_face, w_speech = 1.0, 0.0, 0.0
        elif face_risk == 0:
            w_health, w_face, w_speech = 0.6, 0.0, 0.4
        elif speech_risk == 0:
            w_health, w_face, w_speech = 0.6, 0.4, 0.0
            
        final_risk = (health_risk * w_health) + (face_risk * w_face) + (speech_risk * w_speech)
        final_risk = min(final_risk, 100.0)
        
        # Base Confidence depends on multi-modal inputs provided
        confidence_base = 70.0 + (15.0 if face_risk > 0 else 0) + (15.0 if speech_risk > 0 else 0)
        
        # Threshold Execution
        if final_risk >= 50.0:
            risk = "HIGH"
            color = "#f43f5e" # Rose/Red neon
            box_glow = "0 0 30px rgba(244, 63, 94, 0.5)"
        elif final_risk >= 25.0:
            risk = "MODERATE"
            color = "#d946ef" # Magenta
            box_glow = "0 0 30px rgba(217, 70, 239, 0.5)"
        else:
            risk = "LOW"
            color = "#06b6d4" # Cyan
            box_glow = "0 0 30px rgba(6, 182, 212, 0.5)"
            
        placeholder.empty()
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            html_content = f"""<div class="glass-card" style="box-shadow: {box_glow}; text-align: center;">
<h4 style="color: #cbd5e1; margin-bottom: -10px;">RISK SCORE</h4>
{get_circular_meter(final_risk, risk, color)}
</div>"""
            st.markdown(html_content, unsafe_allow_html=True)
            st.progress(confidence_base / 100.0, text=f"Pipeline Confidence: {confidence_base:.1f}%")
            
        history.save_record(final_risk / 100.0, risk, glucose, bmi, face_risk / 100.0, speech_risk / 100.0)
        
        # Output exactly computed explicit values for LLM strictness
        computed_values = {
            "health_risk_percent": round(health_risk, 1),
            "face_risk_percent": round(face_risk, 1),
            "speech_risk_percent": round(speech_risk, 1),
            "overall_final_score": round(final_risk, 1)
        }
        
        # Display AI Report
        ai_response = get_recommendation(data, risk, confidence_base, computed_values)
        
        factor_contributions = ai_response.get('factor_contributions', 'N/A')
        score_interp = ai_response.get('score_interpretation', 'N/A')
        health_impl = ai_response.get('health_implications', 'N/A')
        recommendations = ai_response.get('recommendations', [])
        final_summary = ai_response.get('final_summary', 'Analysis Complete.')
        
        with col_res2:
            st.markdown(f"### 📊 Comprehensive AI Report")
            
            st.markdown(f"**🔍 Factor Contributions:**\n\n{factor_contributions}")
            st.markdown(f"**📈 Score Interpretation:**\n\n{score_interp}")
            st.markdown(f"**⚠️ Health Implications:**\n\n{health_impl}")
            
            st.markdown("#### ✅ Actionable Recommendations")
            for r in recommendations:
                st.markdown(f"- {r}")
                
            st.markdown("---")
            st.success(f"**📌 Summary:** {final_summary}")

with tab2:
    st.markdown("## 📈 Graphical Analytics")
    col_w1, col_w2, col_w3 = st.columns(3)
    summary = history.get_weekly_summary()
    
    col_w1.metric("Avg Weekly Risk", f"{summary['avg_risk']*100:.1f}%")
    col_w2.metric("Best Day", summary['best_day'])
    col_w3.metric("Worst Day", summary['worst_day'])
    
    st.markdown("---")
    fig_risk = history.plot_risk_trend()
    if fig_risk:
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("No history data available yet. Analyze a patient to see trends.")
        
    fig_metrics = history.plot_metrics_comparison()
    if fig_metrics:
        st.plotly_chart(fig_metrics, use_container_width=True)
