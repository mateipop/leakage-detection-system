# leak-detect-ai-physics

## Description
Modular Pipeline Leakage Detection System. Integrates WNTR hydraulic physics with a real-time Data Layer for Z-score normalization and an AI Layer for anomaly detection. Built as a decoupled, event-driven architecture using Python and Redis to bridge simulation and inference.

## Architecture
The system is stratified into three isolated layers as defined in the Technical Architecture Document:
* **Simulation Layer**: WNTR Engine generates physics-based hydraulic states.
  It emits node + link telemetry plus leak metadata for pinpointer training.
* **Data Layer**: Ingests raw telemetry, performs validation, and calculates Z-score normalization.
* **AI Layer**: Interprets windowed features to detect leaks and pinpoint likely leak nodes.

## Prerequisites
* Python 3.12+
* Redis Server (`brew install redis`)
* WNTR Library (`pip install wntr`)

## How to Run
Open three separate terminals in VS Code:

1. **Start Infrastructure**:
   `redis-server`

2. **Start Data Layer**:
   `python3 -m leak_detect_ai_physics.data_layer.collector`

3. **Start Simulation**:
   `python3 -m leak_detect_ai_physics.simulation_layer.wntr_driver`

4. **Start AI Inference**:
   `python3 -m leak_detect_ai_physics.ai_layer.inference`

5. **Train Models**:
   `python3 -m leak_detect_ai_physics.ai_layer.trainer`

## Windowed Training

Generate windowed datasets (3-hour windows, 30-minute stride) with the dataset builder:

`uv run python -m leak_detect_ai_physics.data_layer.dataset_builder --clear --duration 86400 --timestep 300 --window-hours 3 --stride-steps 6`

Train the CNN anomaly + pinpointer models:

`uv run python -m leak_detect_ai_physics.ai_layer.trainer --dataset data/training_data.jsonl`

Artifacts under `data/` are gitignored (datasets and trained models).

## Message Broker Notes
Redis pub/sub is a good default for local development and low-throughput streams.
If you need durability, replay, or higher fan-out, consider NATS or Kafka.

## Architecture
See `docs/ARCHITECTURE.md` for the layer responsibilities and data flow.
