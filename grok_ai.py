import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os


def get_recommendation(user_data, risk_level, confidence, computed_values):

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0   # no randomness
    )

    template = """
You are "Stroke AI Pro", an advanced diagnostic neural network. Your absolute objective is to act as the primary physician predicting the patient's stroke chances based on the data provided. 
Speak directly to the patient in the first person ("I have computed your risk to be...", "My analysis of your facial telemetry indicates...").
Use ONLY the provided computed numerical values to form your diagnosis. DO NOT hallucinate conditions outside the data.

Data fed to your neural network:
- Patient Vitals: {user_data}
- Your Computed Health Risk Component: {health_risk}/100
- Your Computed Face Asymmetry Component: {face_risk}/100
- Your Computed Speech Risk Component: {speech_risk}/100
- Your Final Predicted Stroke Risk Score: {final_risk}/100
- Your Computed Risk Classification: {risk_level}

Requirements (Return EXACTLY a valid JSON object):
- "factor_contributions": Explain how you analyzed the data and calculated the {risk_level} risk. Explicitly detail how the Health, Face, and Speech components led you to predict a {final_risk}/100 score. Be authoritative and analytical.
- "score_interpretation": Interpret your {final_risk}/100 overall score prediction straightforwardly for the patient.
- "health_implications": Real-world implications of your predictive findings.
- "recommendations": Array of 4-6 strict actionable suggestions based EXACTLY on your generated risk prediction factors.
- "final_summary": A 2-3 line authoritative summary wrapping up your diagnosis and prediction.

Format:
{{
  "factor_contributions": "...",
  "score_interpretation": "...",
  "health_implications": "...",
  "recommendations": ["...", "...", "...", "..."],
  "final_summary": "..."
}}
"""

    prompt = PromptTemplate.from_template(template)

    try:
        chain = prompt | llm
        response = chain.invoke({
            "user_data": user_data,
            "risk_level": risk_level,
            "health_risk": computed_values.get("health_risk_percent", 0),
            "face_risk": computed_values.get("face_risk_percent", 0),
            "speech_risk": computed_values.get("speech_risk_percent", 0),
            "final_risk": computed_values.get("overall_final_score", 0)
        })

        content = response.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()

        return json.loads(content)

    except Exception as e:
        return {
            "factor_contributions": "Unable to calculate detailed factor contributions at this time.",
            "score_interpretation": "Score interpretation is currently unavailable.",
            "health_implications": "Unable to provide health implications due to system error.",
            "recommendations": [
                "Please consult a doctor",
                "Monitor your health vitals",
                "Maintain a balanced diet",
                "Exercise regularly"
            ],
            "final_summary": "AI Analysis system encountered an error. Please seek medical advice if you feel unwell."
        }