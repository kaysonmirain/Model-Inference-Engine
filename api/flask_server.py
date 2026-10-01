"""Minimal Flask wrapper for the same portable model artifact."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request

from api.model_runtime import FEATURES, load_model, predict

ROOT = Path(__file__).resolve().parents[1]
MODEL = load_model(ROOT / "models" / "travel_time_model.txt")

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict_trip_time():
    payload = request.get_json(force=True)
    features = {feature: float(payload[feature]) for feature in FEATURES}
    return jsonify({"predicted_trip_seconds": round(predict(MODEL, features), 4)})
