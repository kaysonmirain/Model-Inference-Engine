"""Shared Python model runtime used by the optional API wrappers."""

from __future__ import annotations

from pathlib import Path

FEATURES = [
    "route_distance_km",
    "route_complexity",
    "crowd_density",
    "closure_active",
    "average_speed_mps",
    "reroute_pressure",
]


def load_model(path: str | Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                values[key] = float(value)
    missing = ["bias", "output_scale", *[f"weight.{feature}" for feature in FEATURES]]
    if any(key not in values for key in missing):
        raise ValueError("model artifact is missing required fields")
    return values


def predict(model: dict[str, float], features: dict[str, float]) -> float:
    missing = [feature for feature in FEATURES if feature not in features]
    if missing:
        raise ValueError(f"missing features: {', '.join(missing)}")
    score = model["bias"]
    for feature in FEATURES:
        score += model[f"weight.{feature}"] * float(features[feature])
    return max(0.0, score * model["output_scale"])
