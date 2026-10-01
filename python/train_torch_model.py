#!/usr/bin/env python3
"""Optional PyTorch training path that exports the same portable model format."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_model import FEATURES, make_training_data


class TravelTimeRegressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(len(FEATURES), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--output", default="models/generated_torch_model.txt")
    args = parser.parse_args()

    torch.manual_seed(42)
    data = make_training_data(args.rows)
    features = torch.tensor([row[0] for row in data], dtype=torch.float32)
    targets = torch.tensor([[row[1]] for row in data], dtype=torch.float32)

    model = TravelTimeRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = torch.nn.MSELoss()

    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(features), targets)
        loss.backward()
        optimizer.step()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    layer = model.linear
    weights = layer.weight.detach().reshape(-1).tolist()
    bias = float(layer.bias.detach()[0])

    with output.open("w", encoding="utf-8") as handle:
        handle.write("# Crowd travel-time model trained with PyTorch for the C++ runtime\n")
        handle.write("output_scale=1.0\n")
        handle.write(f"bias={bias:.8f}\n")
        for name, weight in zip(FEATURES, weights):
            handle.write(f"weight.{name}={weight:.8f}\n")

    print(f"training_loss={float(loss):.4f} output={output}")


if __name__ == "__main__":
    main()
