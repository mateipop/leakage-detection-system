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
   `python3 -m data_layer.collector`

3. **Start Simulation**:
   `python3 simulation_layer/wntr_driver.py`
