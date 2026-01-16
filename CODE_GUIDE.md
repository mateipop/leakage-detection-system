# Water Network Leak Detection System - Code Guide

## Quick Reference for Demonstrations & Questions

---

## 1. Project Structure Overview

```
leak_detection/
├── orchestrator.py          # Main control loop, coordinates all components
├── config.py                # System configuration (thresholds, modes)
├── agents/
│   ├── base.py              # Agent base class, message bus (pub-sub)
│   ├── sensor_agent.py      # Distributed anomaly detection (Z-scores)
│   ├── coordinator_agent.py # Alert aggregation, investigation management
│   ├── localizer_agent.py   # Leak triangulation algorithm
│   └── multi_agent_system.py# Agent orchestration wrapper
├── simulation/
│   ├── network_simulator.py # WNTR/EPANET hydraulic simulation
│   ├── device_simulator.py  # Sensor noise, battery, failures
│   └── leak_injector.py     # Programmatic leak injection
├── data_layer/
│   ├── data_pipeline.py     # Data validation, filtering, Z-score computation
│   ├── redis_manager.py     # Time-series storage
│   └── telemetry_buffer.py  # Sliding window buffer
└── tui/
    └── dashboard.py         # Textual-based terminal UI
```

---

## 2. Key Algorithms & Where to Find Them

### 2.1 Anomaly Detection (Z-Score with Adaptive Baseline)

**File:** `agents/sensor_agent.py` → `AdaptiveThreshold` class (lines 20-45)

```python
# Exponential Moving Average for baseline
μ_{t+1} = μ_t + α(x_t - μ_t)

# Z-Score calculation
z = (x - μ) / σ
```

**Key Parameters:**
- `alpha = 0.05` - Learning rate for EMA
- `min_std = 0.02` - Minimum standard deviation floor
- Alert threshold: `|z| > 3.0` (line ~182)

**To Demonstrate:** Set a breakpoint in `SensorAgent.sense()` and watch the z-score computation as pressure drops after leak injection.

---

### 2.2 Leak Localization (Distance-Weighted Triangulation)

**File:** `agents/localizer_agent.py` → `_triangulate_with_distances()` (lines 165-230)

```python
Score(n) = Σ (w_i / (100 + d(n, s_i))) - penalty
```

Where:
- `w_i` = |z-score| × confidence (signal strength from sensor i)
- `d(n, s_i)` = Network distance (meters) from candidate node n to sensor i
- `100` = Smoothing constant to prevent division by zero
- `penalty` = Proximity to silent sensors (reduces score)

**Key Logic:**
1. Collect all alerting sensors and their weights
2. For each candidate node in the network (782 nodes):
   - Sum weighted inverse distances to alerting sensors
   - Subtract penalty for proximity to silent sensors
3. Rank by score, return top candidate

**To Demonstrate:** Run `analyze_false_positives.py` and observe the "Top candidates" output showing ranked nodes.

---

### 2.3 Multi-Agent Communication (Publish-Subscribe)

**File:** `agents/base.py` → `MessageBus` class (lines 40-90)

**Message Types:**
| Type | Sender | Receiver | Purpose |
|------|--------|----------|---------|
| `ANOMALY_ALERT` | Sensor | Coordinator | Report detected anomaly |
| `LOCALIZE_REQUEST` | Coordinator | Localizer | Request triangulation |
| `LOCALIZATION_RESULT` | Localizer | Coordinator | Return best candidate |
| `PEER_GOSSIP` | Sensor | Sensor | Neighbor verification |
| `MODE_CHANGE` | Coordinator | Sensor | Switch sampling rate |

**To Demonstrate:** Add logging in `MessageBus.publish()` to show message flow.

---

### 2.4 Coordinator Decision Logic

**File:** `agents/coordinator_agent.py` → `decide()` method (lines 115-200)

**Investigation Triggers:**
1. **Collaboration Rule:** ≥3 sensors alerting with avg confidence ≥0.6
2. **Persistence Rule:** 1 sensor with 5+ consecutive alerts and confidence ≥0.7

**Localization Trigger:** Investigation has ≥10 unique sensors

**Key Thresholds:**
```python
MIN_ALERTS_FOR_INVESTIGATION = 3
MIN_SENSORS_FOR_LOCALIZATION = 10
CONFIDENCE_THRESHOLD = 0.6
```

---

## 3. Data Flow Diagram

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   WNTR      │────▶│   Device     │────▶│    Data     │
│  Simulation │     │  Simulator   │     │  Pipeline   │
└─────────────┘     └──────────────┘     └─────────────┘
      │                   │                     │
      │ Hydraulics        │ + Noise             │ Filter/Z-Score
      ▼                   ▼                     ▼
┌─────────────────────────────────────────────────────┐
│                 Multi-Agent System                   │
│  ┌─────────┐   ┌─────────────┐   ┌──────────────┐  │
│  │ Sensor  │──▶│ Coordinator │──▶│  Localizer   │  │
│  │ Agents  │   │   Agent     │   │    Agent     │  │
│  │  (33)   │   │    (1)      │   │     (1)      │  │
│  └─────────┘   └─────────────┘   └──────────────┘  │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  Dashboard  │
                   │    (TUI)    │
                   └─────────────┘
```

---

## 4. Common Professor Questions & Answers

### Q: "How do you prevent the system from learning the leak as 'normal'?"

**A:** In `sensor_agent.py` line ~170:
```python
if self._learning_enabled and not self._pending_alert and abs(zscore) < 2.0:
    self._model.update(reading_val)
```
We only update the baseline when Z < 2.0. Once an anomaly is detected, the baseline freezes, preventing the leak from being normalized.

---

### Q: "What if a sensor is near the leak but doesn't alert?"

**A:** The localizer applies a **silent sensor penalty** (`localizer_agent.py` lines 207-215):
```python
for silent_s in silent_sensors:
    if s_dist < 400.0:
        penalty_score += 1.0 / (100.0 + s_dist)
```
Candidates near silent sensors get their score reduced.

---

### Q: "How do you handle multiple simultaneous leaks?"

**A:** The coordinator merges related alerts into a single investigation if >50% sensor overlap (`coordinator_agent.py` lines 168-180). Each distinct leak cluster becomes its own investigation.

---

### Q: "Why not use machine learning?"

**A:** 
1. **Interpretability:** Z-scores and distance formulas are explainable
2. **No training data needed:** Works out-of-the-box on any EPANET network
3. **Real-time:** No inference latency, pure math
4. **Future work:** LSTM baseline prediction mentioned in slides

---

### Q: "What's the detection accuracy?"

**A:** ~75% true positive rate within 25 network hops using only 4.2% sensor coverage (33 of 782 nodes). Run `analyze_false_positives.py` for detailed stats.

---

### Q: "How does the gossip protocol work?"

**A:** When a sensor detects an anomaly:
1. Sends `PEER_GOSSIP` to its 4 nearest neighbors
2. Neighbors check their own Z-score
3. If neighbor's |Z| > 1.5, they confirm
4. Confirmation increases confidence (not currently a hard gate)

**File:** `sensor_agent.py` → `_handle_peer_gossip()` (lines 341-358)

---

### Q: "Is there any 'cheating' with ground truth?"

**A:** No. The separation is strict:

| Component | Uses Ground Truth? | Purpose |
|-----------|-------------------|---------|
| Sensor Agents | ❌ No | Detect from sensor data only |
| Coordinator | ❌ No | Aggregate alerts |
| Localizer | ❌ No | Triangulate from alerting sensors |
| `evaluate_detection()` | ✅ Yes | **Scoring only** (not decisions) |
| Dashboard colors | ✅ Yes | User feedback (green/red) |

See `orchestrator.py` → `evaluate_detection()` (lines 320-365)

---

## 5. Key Files to Demo

### 5.1 Live System Demo
```bash
python main.py
```
Press `L` to inject leak, watch agents respond.

### 5.2 Headless Testing
```bash
python analyze_false_positives.py
```
Runs 20 automated leak tests, prints accuracy stats.

### 5.3 Show Specific Algorithm

**Z-Score Detection:**
```python
# In sensor_agent.py, add before line 182:
logger.info(f"Node {self.node_id}: value={reading_val:.2f}, mean={self._model.mean:.2f}, z={zscore:.2f}")
```

**Localization Scoring:**
```python
# In localizer_agent.py, add in _triangulate_with_distances():
logger.info(f"Candidate {node}: score={final_score:.4f}, confidence={confidence:.2%}")
```

---

## 6. Configuration Tuning

**File:** `config.py` and agent class constants

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| Z-Score Threshold | `sensor_agent.py:182` | 3.0 | Lower = more sensitive |
| Min Sensors for Localization | `coordinator_agent.py:35` | 10 | Higher = more accurate but slower |
| TP Distance Threshold | `orchestrator.py:346` | 25 hops | Defines "close enough" |
| Smoothing Constant | `localizer_agent.py:202` | 100 | Higher = less distance penalty |

---

## 7. Technology Stack Quick Reference

| Component | Library | Purpose |
|-----------|---------|---------|
| Hydraulics | WNTR + EPANET | Water network physics |
| Graph Distances | NetworkX | Shortest path (Dijkstra) |
| Math | NumPy | Statistics, linear algebra |
| Storage | Redis | Time-series telemetry |
| UI | Textual | Terminal dashboard |
| Config | dataclasses | Type-safe configuration |

---

## 8. Running the Tests

```bash
# Quick import check
python -c "from leak_detection.orchestrator import SystemOrchestrator; print('OK')"

# Full accuracy test
python analyze_false_positives.py

# Interactive demo
python main.py
```

---

## 9. Code Metrics

- **Total Python Files:** ~15
- **Lines of Code:** ~3,500
- **Agent Count:** 35 (33 sensors + 1 coordinator + 1 localizer)
- **Network Size:** 782 junctions, ~900 pipes
- **Sensor Coverage:** 4.2% (pressure), 10.5% (flow)

---

## 10. Quick Debugging

**No detections?**
- Check `MIN_SENSORS_FOR_LOCALIZATION` (needs 10+ sensors alerting)
- Leak might be far from sensors

**Too many false positives?**
- Increase `MIN_SENSORS_FOR_LOCALIZATION`
- Increase Z-score threshold from 3.0 to 3.5

**Dashboard not updating?**
- Ensure Redis is running: `redis-server`
- Check for exceptions in terminal

---

*Last Updated: January 2026*
