from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
import joblib
import pandas as pd
import os

app = FastAPI(
    title="API de predicción de muertes por demencia",
    description="Esta API permite predecir muertes por demencia en diferentes paises de Europa.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Montar la carpeta estática para servir archivos (opcional, ya que tienes ruta "/" personalizada)
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "best_xgb_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("No se encontraron 'best_xgb_model.pkl' o 'scaler.pkl'.")

best_xgb = joblib.load(MODEL_PATH)
if not hasattr(best_xgb, "gpu_id"):
    best_xgb.gpu_id = 0
if not hasattr(best_xgb, "predictor"):
    best_xgb.predictor = "cpu_predictor"

scaler = joblib.load(SCALER_PATH)

class PredictionInput(BaseModel):
    Country: str
    Year: int
    Annual_Mean_NO2_Concentration: float
    life_expectancy: float
    Avg_Percentage_Activity: float
    Ratio_annual_of_inequality_income: float
    percentage_of_overweight_BMI: float
    sex_Males: int
    Interaction_LifeExpectancy_Activity: float = None

    @validator("Interaction_LifeExpectancy_Activity", pre=True, always=True)
    def calculate_interaction(cls, v, values):
        if v is None:
            if "life_expectancy" in values and "Avg_Percentage_Activity" in values:
                return values["life_expectancy"] * values["Avg_Percentage_Activity"]
            else:
                raise ValueError("Faltan 'life_expectancy' y/o 'Avg_Percentage_Activity' para calcular la interacción.")
        return v

FEATURES_ORDER = [
    'Annual_Mean_NO2_Concentration',
    'life_expectancy',
    'Avg_Percentage_Activity',
    'Ratio annual of inequality income',
    'percentage_of_overweight_BMI',
    'sex_Males',
    'Interaction_LifeExpectancy_Activity',
    'Year'
]

@app.get("/", response_class=HTMLResponse)
def get_index():
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(content="<h1>Error: no se encontró index.html</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        data_dict = {
            'Annual_Mean_NO2_Concentration': [input_data.Annual_Mean_NO2_Concentration],
            'life_expectancy': [input_data.life_expectancy],
            'Avg_Percentage_Activity': [input_data.Avg_Percentage_Activity],
            'Ratio annual of inequality income': [input_data.Ratio_annual_of_inequality_income],
            'percentage_of_overweight_BMI': [input_data.percentage_of_overweight_BMI],
            'sex_Males': [input_data.sex_Males],
            'Interaction_LifeExpectancy_Activity': [input_data.Interaction_LifeExpectancy_Activity],
            'Year': [input_data.Year]
        }
        df_input = pd.DataFrame(data_dict, columns=FEATURES_ORDER)
        X_scaled = scaler.transform(df_input)
        prediction = best_xgb.predict(X_scaled)
        predicted_deaths = int(round(prediction[0]))
        return {
            "Country": input_data.Country,
            "Year": input_data.Year,
            "Predicted_Dementia_Deaths": predicted_deaths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

@app.get("/welcome")
def welcome():
    return {"message": "Bienvenido a la API de Predicción de Muertes por Demencia"}
#prueba
