#!/usr/bin/env python3
"""Train a deterministic crowd travel-time model for the C++ runtime."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

FEATURES = [
    "route_distance_km",
    "route_complexity",
    "crowd_density",
    "closure_active",
    "average_speed_mps",
    "reroute_pressure",
]


def make_training_data(rows: int, seed: int = 42) -> list[tuple[list[float], float]]:
    random.seed(seed)
    data = []
    for _ in range(rows):
        route_distance = random.uniform(0.2, 2.0)
        route_complexity = random.uniform(0.05, 1.0)
        crowd_density = random.uniform(0.05, 1.0)
        closure_active = float(random.random() < 0.22)
        average_speed = random.uniform(0.7, 1.8)
        reroute_pressure = random.uniform(0.0, 1.0) if closure_active else random.uniform(0.0, 0.25)
        trip_seconds = 12.0 + 95.0 * route_distance + 22.0 * route_complexity
        trip_seconds += 70.0 * crowd_density + 35.0 * closure_active
        trip_seconds -= 28.0 * average_speed
        trip_seconds += 45.0 * reroute_pressure + random.uniform(-3.0, 3.0)
        features = [
            route_distance,
            route_complexity,
            crowd_density,
            closure_active,
            average_speed,
            reroute_pressure,
        ]
        data.append((features, max(1.0, trip_seconds)))
    return data


def train(data: list[tuple[list[float], float]], epochs: int = 4000, learning_rate: float = 0.03):
    weights = [0.0] * len(FEATURES)
    bias = 0.0
    for _ in range(epochs):
        bias_gradient = 0.0
        gradients = [0.0] * len(weights)
        for features, target in data:
            prediction = bias + sum(w * x for w, x in zip(weights, features))
            error = prediction - target
            bias_gradient += error
            for index, value in enumerate(features):
                gradients[index] += error * value
        scale = 2.0 / len(data)
        bias -= learning_rate * scale * bias_gradient
        for index in range(len(weights)):
            weights[index] -= learning_rate * scale * gradients[index]
    return bias, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--output", default="models/generated_model.txt")
    args = parser.parse_args()

    data = make_training_data(args.rows)
    split = int(len(data) * 0.8)
    bias, weights = train(data[:split])
    errors = []
    for features, target in data[split:]:
        prediction = bias + sum(w * x for w, x in zip(weights, features))
        errors.append(abs(prediction - target))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("# Crowd travel-time model trained in Python for the C++ runtime\n")
        handle.write("output_scale=1.0\n")
        handle.write(f"bias={bias:.8f}\n")
        for name, weight in zip(FEATURES, weights):
            handle.write(f"weight.{name}={weight:.8f}\n")
    print(f"validation_mae_seconds={sum(errors) / len(errors):.4f} output={output}")


if __name__ == "__main__":
    main()
