from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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
    age: int = Field (..., ge = 18, le = 100, description="Edad del cliente (entre 18 y 100 años)")
    job: str = Field (..., description="Tipo de trabajo del cliente")
    marital: str = Field (..., description="Estado civil del cliente")
    education: str = Field (..., description="Nivel educativo del cliente")
    housing: int = Field (..., description="Indicador de si el cliente tiene una hipoteca")
    loan: int = Field (..., description="Indicador de si el cliente tiene un préstamo personal")
    contact: str = Field (..., description="Tipo de contacto con el cliente")
    month: str = Field (..., description="Mes de contacto con el cliente")
    day_of_week: str = Field (..., description="Día de la semana del contacto con el cliente")
    duration: int = Field (..., description="Duración de la llamada en segundos")
    campaign: int = Field (..., description="Número de contactos realizados durante esta campaña")
    previous: int = Field (..., description="Número de contactos realizados en la campaña anterior")
    poutcome: str = Field (..., description="Resultado de la campaña anterior")
    emp_var_rate: float = Field (..., description="Tasa de variación del empleo")
    cons_price_idx: float = Field (..., description="Índice de precios al consumidor")
    cons_conf_idx: float = Field (..., description="Índice de confianza del consumidor")
    euribor3m: float = Field (..., description="Tasa Euribor a 3 meses")
    nr_employed: int = Field (..., description="Número de empleados")
    contacted_before: int = Field (..., description="Indicador de si el cliente fue contactado antes")
    contacts_diff: int = Field (..., description="Diferencia en el número de contactos entre campaña actual y anterior")
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
            }
        }

class PredictionResponse(BaseModel):
    """Estructura para la respuesta de predicción """
    prediction: int
    probability: Dict[float, Any]
    model_info: Dict[str, Any]

# Cargar el modelo y el preprocesador al iniciar la aplicación
MODEL_PATH = "models/decision_tree_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = joblib.load(model_file)
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {e}")

try:
    with open(PREPROCESSOR_PATH, "rb") as preprocessor_file:
        preprocessor = joblib.load(preprocessor_file)
except Exception as e:
    raise RuntimeError(f"Error al cargar el preprocesador: {e}")

@app.get("/")
def root():
    return {
        "message": "API de Clasificación de Clientes Bancarios está en funcionamiento.",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs",
        }
    }

@app.get("/health")
def health():
    """Verifica el estado de salud de la API y la carga del modelo."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_type": type(model).__name__ if model else None,
        }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Realiza una predicción basada en los datos del cliente bancario proporcionados."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="El modelo o el preprocesador no están cargados.")

    # Convertir la solicitud a DataFrame
    try:
        input_data = pd.DataFrame([request.dict()])

        # Convertir las columnas numéricas a float como en el entrenamiento
        int_columns = input_data.select_dtypes(include=['int']).columns
        for col in int_columns:
            input_data[col] = input_data[col].astype(float)

        # Preprocesar los datos de entrada
        input_data = preprocessor.transform(input_data)

        # Realizar la predicción
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        class_labels = model.classes_
        probability_dict = {class_labels[i]: float(probability[i]) for i in range(len(class_labels))}
        model_info = {
            "model_type": type(model).__name__,
            "preprocessor_type": type(preprocessor).__name__,
        }

        return PredictionResponse(
            prediction=prediction,
            probability=probability_dict,
            model_info=model_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la solicitud: {e}")
