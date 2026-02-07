import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import uvicorn

app = FastAPI()

# Standard CORS setup to allow the HTML file to talk to Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load data
df = pd.read_csv('career_data.csv')

# 2. Setup Encoders
le_lang = LabelEncoder().fit(df['Native_Language'])
le_speed = LabelEncoder().fit(df['Learning_Speed'])
le_outcome = LabelEncoder().fit(df['Outcome_Status'])

# 3. Train Model (Using all relevant features)
# We convert text categories to numbers so the AI can process them
df['Lang_Enc'] = le_lang.transform(df['Native_Language'])
df['Speed_Enc'] = le_speed.transform(df['Learning_Speed'])

X = df[['Current_Skill_Level', 'Daily_Study_Hrs', 'Target_Salary', 'Lang_Enc', 'Speed_Enc']]
y = le_outcome.transform(df['Outcome_Status'])

model = RandomForestClassifier(n_estimators=100).fit(X, y)

@app.get("/predict")
def predict_outcome(
    skills: int,
    hours: int,
    salary: int,
    lang: str,
    speed: str,
    state: str = "Kerala",
    location_type: str = "Urban"
):

    try:
        l_enc = le_lang.transform([lang])[0]
        s_enc = le_speed.transform([speed])[0]
    except:
        l_enc, s_enc = 0, 0

    input_df = pd.DataFrame(
        [[skills, hours, salary, l_enc, s_enc]],
        columns=['Current_Skill_Level', 'Daily_Study_Hrs', 'Target_Salary', 'Lang_Enc', 'Speed_Enc']
    )

    probs = model.predict_proba(input_df)[0]
    prediction_enc = model.predict(input_df)[0]
    status = le_outcome.inverse_transform([prediction_enc])[0]

    roi = "High" if (salary / (101 - skills)) > 800 else "Standard"
    dropout_prob = "High" if (hours > 10 or (skills < 15 and hours < 2)) else "Low"

    # --------- LOCATION INTELLIGENCE ---------
    if location_type == "Rural":
        relocation_risk = "High"
        location_score = 60
        location_advice = "Remote or Govt-based careers recommended."
    elif location_type == "Semi-Urban":
        relocation_risk = "Medium"
        location_score = 80
        location_advice = "Hybrid career paths possible."
    else:
        relocation_risk = "Low"
        location_score = 100
        location_advice = "Full industry access available."

    explanations = {
        "en": f"AI predicts '{status}'. Burnout risk is {'Critical' if hours > 8 else 'Optimal'}.",
        "es": f"El IA predice '{status}'. Riesgo de agotamiento: {'Crítico' if hours > 8 else 'Óptimo'}.",
        "hi": f"AI '{status}' का अनुमान लगाता है। {'जोखिम अधिक है' if hours > 8 else 'सामान्य'}।",
        "fr": f"L'IA prévoit '{status}'. Risque de burnout : {'Critique' if hours > 8 else 'Optimal'}."
    }

    return {
        "status": status.replace("_", " "),
        "confidence": round(max(probs) * 100, 1),
        "feasibility": round((skills * 0.45) + (hours * 5), 1),
        "burnout": "Critical" if hours > 10 else "High" if hours > 7 else "Optimal",
        "roi": roi,
        "dropout": dropout_prob,
        "location_score": location_score,
        "relocation_risk": relocation_risk,
        "location_advice": location_advice,
        "adaptation": "Turbo" if speed == "Turbo" else "Deep Focus",
        "explanation": explanations.get(lang, explanations["en"])
    }


if __name__ == "__main__":
    # CHANGED TO PORT 8000 TO MATCH YOUR HTML
    uvicorn.run(app, host="0.0.0.0", port=8000)
