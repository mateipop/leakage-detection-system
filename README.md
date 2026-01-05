# leak-detect-ai-physics

## Description
Modular Pipeline Leakage Detection System. Integrates WNTR hydraulic physics with a real-time Data Layer for Z-score normalization and an AI Layer for anomaly detection. Built as a decoupled, event-driven architecture using Python and Redis to bridge simulation and inference.

## Architecture
The system is stratified into three isolated layers as defined in the Technical Architecture Document:
* **Simulation Layer**: WNTR Engine generates physics-based hydraulic states.
* **Data Layer**: Ingests raw telemetry, performs validation, and calculates Z-score normalization.
* **AI Layer**: Interprets normalized features to detect leaks and issue control commands.

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

5. **Train Baseline Model**:
   `python3 -m leak_detect_ai_physics.ai_layer.trainer`

The AI entrypoints are stubs right now and document the intended flow.

## Message Broker Notes
Redis pub/sub is a good default for local development and low-throughput streams.
If you need durability, replay, or higher fan-out, consider NATS or Kafka.

## Architecture
See `docs/ARCHITECTURE.md` for the layer responsibilities and data flow.
