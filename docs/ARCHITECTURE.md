# Architecture Overview

## Layers

### Simulation Layer
- Runs WNTR and publishes node + link telemetry for the full network.
- Output: `sensor_telemetry` messages with raw features, leak labels, and leak metadata
  suitable for anomaly scoring and pinpointer training.

### Data Layer
- Subscribes to `sensor_telemetry` and handles node + link payloads.
- Validates telemetry, normalizes features, and writes labeled training records.
- Builds windowed datasets with pre/post leak context for model training.
- Streams normalized features to `live_features` for downstream consumers.

### AI Layer
- Inference consumes normalized features from the data layer only.
- Training reads windowed records from `data/training_data.jsonl`, including leak window
  labels and `leak_node_id` for pinpointer targets.

## Data Flow

1. Simulation publishes telemetry to Redis channel `sensor_telemetry`.
2. Data layer consumes telemetry, stores labeled training data, and publishes features.
3. AI inference consumes data layer features for live scoring.
4. AI training consumes stored labeled data with gold predictions to produce model artifacts.

## Channels and Storage

- `sensor_telemetry`: raw simulation output, used by the data layer.
- `live_features`: normalized features from data layer.
- `ai_predictions`: model scores emitted by AI inference.
- `data/training_data.jsonl`: windowed training set records.
- `data/training_data.meta.json`: windowing metadata and node coordinates.
- `data/models/`: model artifacts.
