# Model Inference Engine

A C++20 runtime for executing crowd travel-time models trained and exported from Python. Its focus is model portability, feature-contract validation, and inference throughput rather than another demand-forecasting application.

## How It Works

The Python training script creates deterministic simulation-style data, trains a linear model, and exports a compact text artifact. The C++ runtime validates the required features, loads the artifact, executes batch predictions, and reports throughput. Optional FastAPI and Flask wrappers expose the same artifact through REST endpoints.

The included model estimates agent trip time from route distance, path complexity, crowd density, dynamic closures, average speed, and rerouting pressure. These features mirror the operating signals produced by the Crowd Simulation Engine and summarized by the Telemetry Event Processor. An automated parity check verifies the C++ predictions against an independent Python reference within a configurable numerical tolerance.

## Project Layout

```text
python/train_model.py          Training and portable model export
python/train_torch_model.py    Optional PyTorch training path
python/reference_inference.py  Reference predictions
api/fastapi_server.py          REST inference service using FastAPI
api/flask_server.py            Minimal Flask inference endpoint
include/inference/model.hpp    Runtime API
src/model.cpp                  Model and CSV loading
src/main.cpp                   Batch benchmark CLI
models/travel_time_model.txt   Included deterministic model
data/sample_features.csv       Example feature rows
tests/test_model.cpp           Runtime tests
```

## Train a Model

```bash
python3 python/train_model.py --output models/generated_model.txt
```

Optional PyTorch training path:

```bash
python3 -m pip install torch
python3 python/train_torch_model.py --output models/generated_torch_model.txt
```

## Build and Run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/model-runtime --model models/travel_time_model.txt --repeat 100000
```

## Tests

```bash
ctest --test-dir build --output-on-failure
./build/model-runtime --model models/travel_time_model.txt --repeat 1
python3 python/reference_inference.py \
  --model models/travel_time_model.txt \
  --input data/sample_features.csv \
  --compare results/predictions.csv
```

## Optional REST API

```bash
python3 -m pip install -r requirements-api.txt
uvicorn api.fastapi_server:app --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"route_distance_km": 1.2, "route_complexity": 0.55, "crowd_density": 0.6, "closure_active": 1, "average_speed_mps": 1.1, "reroute_pressure": 0.5}'
```

Docker:

```bash
docker build -t model-inference-engine .
docker run --rm -p 8000:8000 model-inference-engine
```

## Limitations

The first version intentionally keeps the exported model format simple so the C++ runtime stays readable. Planned extensions include PyTorch-to-ONNX export, ONNX Runtime integration, tensor shape validation, quantized models, thread-pool execution, and model-version metadata.
