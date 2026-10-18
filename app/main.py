
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

# Select model version from environment variable
MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.1")

MODEL_FILES = {
    "v0.1": "model.joblib",
    "v0.2": "model_v02.joblib",
}

if MODEL_VERSION not in MODEL_FILES:
    raise ValueError(f"Unsupported model version: {MODEL_VERSION}")

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / MODEL_FILES[MODEL_VERSION]
)

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(lifespan=lifespan)


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(patient: PatientFeatures):
    data = pd.DataFrame([patient.model_dump()])
    prediction = model.predict(data)[0]
    return {"prediction": float(prediction)}
