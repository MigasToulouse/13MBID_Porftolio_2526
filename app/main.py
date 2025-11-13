from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import joblib
import os 
import json
from typing import Any, Dict

app = FastAPI(
    title="Modelo de Clasificación de Clientes Bancarios",
    description="API para predecir si un cliente bancario suscribirá un depósito a plazo fijo.",
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    """ Modelo de datos para la solicitud de predicción """
    # Atributos con la estructura ya preprocesada
    age: int
    job: str
    marital: str
    education: str
    housing: int
    loan: int
    contact: str
    month: str
    day_of_week: str
    duration: int
    campaign: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: int
    contacted_before: int
    contacts_diff: int
    y: int

    class Config:
        """ Ejemplo de datos de entrada para la predicción """
        json_schema_extra = {
            "example": {
                "age": 35,
                "job": "technician",
                "marital": "married",
                "education": "university.degree",
                "housing": 1,
                "loan": 0,
                "contact": "cellular",
                "month": "may",
                "day_of_week": "mon",
                "duration": 300,
                "campaign": 1,
                "contacted_before": 1,
                "contacts_diff": 1,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp_var_rate": 1.1,
                "cons_price_idx": 93.994,
                "cons_conf_idx": -36.4,
                "euribor3m": 4.857,
                "nr_employed": 5191,
                "y": 0
            }
        }

class PredictionResponse(BaseModel):
    """Estructura para la respuesta de predicción """
    prediction: int
    probability: Dict[float, Any]
    model_info: Dict[str, Any]

# Cargar el modelo y el preprocesador al iniciar la aplicación
MODEL_PATH = "models/decision_tree_model.pkl"


try:
    with open(MODEL_PATH, "rb") as model_file:
        model = joblib.load(model_file)
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {e}")

PREPROCESSOR_PATH = "models/preprocessor.pkl"
try:
    with open(PREPROCESSOR_PATH, "rb") as preprocessor_file:
        preprocessor = joblib.load(preprocessor_file)
except Exception as e:
    raise RuntimeError(f"Error al cargar el preprocesador: {e}")

@app.get("/")
def root():
    return {"message": "API de Clasificación de Clientes Bancarios está en funcionamiento."}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Realiza una predicción basada en los datos del cliente bancario proporcionados."""
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")
    
    # Convertir la solicitud a DataFrame
    try:
        input_data = pd.DataFrame([request.dict()])

        # Preprocesar los datos de entrada
        input_data = preprocessor.transform(input_data)
        # Realizar la predicción
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        class_labels = model.classes_
        probability_dict = {class_labels[i]: float(probability[i]) for i in range(len(class_labels))}
        model_info = {
            "model_type": type(model).__name__,
        }

        return PredictionResponse(
            prediction=prediction,
            probability=probability_dict,
            model_info=model_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")
