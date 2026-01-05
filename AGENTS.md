# Repository Guidelines

## Project Structure & Module Organization
- `src/leak_detect_ai_physics/data_layer/`: Redis-backed ingestion and feature processing (`collector.py`, `processor.py`, `utils.py`).
- `src/leak_detect_ai_physics/simulation_layer/`: WNTR-based hydraulic simulation and telemetry publisher (`wntr_driver.py`).
- `src/leak_detect_ai_physics/ai_layer/`: Training and inference entrypoints (`trainer.py`, `inference.py`).
- `src/leak_detect_ai_physics/config.py`: Runtime configuration constants.
- `requirements.txt`: Python dependencies (install in a venv).
- `temp.*` / `dump.rdb`: local artifacts from simulations or Redis; keep them out of releases.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: create/activate a local virtualenv.
- `pip install -r requirements.txt`: install runtime dependencies.
- `redis-server`: start the local Redis broker on `localhost:6379`.
- `python3 -m leak_detect_ai_physics.data_layer.collector`: run the data layer subscriber and feature normalizer.
- `python3 -m leak_detect_ai_physics.simulation_layer.wntr_driver`: run the WNTR simulation and publish telemetry.
- `python3 -m leak_detect_ai_physics.ai_layer.inference`: run the AI inference stub.
- `python3 -m leak_detect_ai_physics.ai_layer.trainer`: run the AI training stub.

## Coding Style & Naming Conventions
- Python 3.12+; use 4-space indentation and PEP 8 naming.
- Modules and functions use `snake_case`; classes use `PascalCase`.
- Constants follow `UPPER_SNAKE_CASE` (see `src/leak_detect_ai_physics/config.py`).
- No formatter or linter is enforced yet; keep diffs small and readable.

## Testing Guidelines
- No automated test suite is present today.
- If you add tests, document how to run them and place them under a `tests/` directory
  (e.g., `tests/test_processor.py` for `src/leak_detect_ai_physics/data_layer/processor.py`).

## Commit & Pull Request Guidelines
- Commit history uses short, descriptive messages (sentence case, no prefixes).
  Example: `data layer start instructions`.
- PRs should include: a brief summary, how you validated locally (commands/output),
  and any relevant runtime notes (e.g., Redis running, channels used).

## Configuration & Runtime Notes
- Redis topics: `sensor_telemetry` (input), `live_features` (output).
- Default endpoints are hardcoded in code (`localhost:6379`); update constants
  in `src/leak_detect_ai_physics/config.py` if needed.
