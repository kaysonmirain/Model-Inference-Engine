#!/usr/bin/env python3
"""Reference implementation for checking predictions produced by the C++ runtime."""

import argparse
import csv
import math


def load_model(path: str):
    values = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                values[key] = float(value)
    return values


def predict(model: dict[str, float], row: dict[str, str]) -> float:
    score = model["bias"]
    for key, value in row.items():
        score += model[f"weight.{key}"] * float(value)
    return max(0.0, score * model["output_scale"])


def load_predictions(path: str) -> list[float]:
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for expected_id, row in enumerate(rows, 1):
        if int(row["row_id"]) != expected_id:
            raise ValueError(f"unexpected row_id at row {expected_id}")
    return [float(row["predicted_trip_seconds"]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/travel_time_model.txt")
    parser.add_argument("--input", default="data/sample_features.csv")
    parser.add_argument("--compare", help="C++ prediction CSV to verify")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    model = load_model(args.model)
    with open(args.input, encoding="utf-8") as handle:
        expected = [predict(model, row) for row in csv.DictReader(handle)]

    if not args.compare:
        for index, prediction in enumerate(expected, 1):
            print(index, prediction)
        return

    actual = load_predictions(args.compare)
    if len(actual) != len(expected):
        raise AssertionError(f"row count differs: Python={len(expected)} C++={len(actual)}")
    errors = [abs(left - right) for left, right in zip(expected, actual)]
    if any(not math.isfinite(value) for value in actual) or max(errors, default=0.0) > args.tolerance:
        raise AssertionError(
            f"prediction parity failed: max_absolute_error={max(errors, default=0.0):.12g}"
        )
    print(
        f"parity_check=passed rows={len(expected)} "
        f"max_absolute_error={max(errors, default=0.0):.12g}"
    )


if __name__ == "__main__":
    main()
