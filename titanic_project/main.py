from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import numpy as np
from typing import Any

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger le modèle
model_path = 'titanic_pipeline.pkl'
try:
    pipeline = joblib.load(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

# Définir le modèle de données pour l'API
class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Cabin: str
    Embarked: str

NUMERICAL_VARIABLES = ["Age", "Fare"]
CATEGORICAL_VARIABLES = ['Sex', 'Cabin', 'Embarked']
CABIN = ['Cabin']

def make_prediction(pipeline, data):
    try:
        # Convertir les données en DataFrame
        df = pd.DataFrame([data.dict()])
        logger.info(f"DataFrame created: {df}")
        
        # Vérifier et traiter les variables numériques et catégoriques
        for var in NUMERICAL_VARIABLES:
            df[var] = pd.to_numeric(df[var], errors='coerce')
        
        for var in CATEGORICAL_VARIABLES:
            df[var] = df[var].astype('category')
        
        # Remplir les valeurs manquantes
        df = df.fillna('missing')
        logger.info(f"DataFrame after filling missing values: {df}")
        
        # Faire la prédiction
        class_preds = pipeline.predict(df)
        prob_preds = pipeline.predict_proba(df)[:, -1]
        
        # Convertir numpy.int64 en types natifs Python
        class_pred = class_preds[0].item() if isinstance(class_preds[0], np.generic) else class_preds[0]
        prob_pred = prob_preds[0].item() if isinstance(prob_preds[0], np.generic) else prob_preds[0]
        
        logger.info(f"Prediction made: class={class_pred}, probability={prob_pred}")
        
        return class_pred, prob_pred
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")

@app.get("/")
def read_root():
    return {"message": "Bienvenue dans le projet Titanic!"}

@app.post("/predict/")
def predict(passenger_data: PassengerData):
    try:
        logger.info(f"Received data: {passenger_data}")
        class_pred, prob_pred = make_prediction(pipeline, passenger_data)
        return {
            "prediction": class_pred,
            "probability": prob_pred
        }
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
