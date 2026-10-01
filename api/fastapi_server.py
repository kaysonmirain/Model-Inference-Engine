"""FastAPI wrapper for the portable model artifact."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from api.model_runtime import load_model, predict

ROOT = Path(__file__).resolve().parents[1]
MODEL = load_model(ROOT / "models" / "travel_time_model.txt")

app = FastAPI(title="Model Inference Engine", version="0.1.0")


class PredictionRequest(BaseModel):
    route_distance_km: float = Field(ge=0)
    route_complexity: float = Field(ge=0)
    crowd_density: float = Field(ge=0)
    closure_active: float = Field(ge=0, le=1)
    average_speed_mps: float = Field(gt=0)
    reroute_pressure: float = Field(ge=0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict_trip_time(payload: PredictionRequest) -> dict[str, float]:
    prediction = predict(MODEL, payload.model_dump())
    return {"predicted_trip_seconds": round(prediction, 4)}
