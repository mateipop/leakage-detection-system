#!/usr/bin/env python3
"""
Test detection pipeline specifically.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from leak_detection.config import SystemConfig
from leak_detection.orchestrator import SystemOrchestrator


def test_detection_pipeline():
    """Test the detection pipeline in detail."""
    print("Testing Detection Pipeline")
    print("=" * 60)

    config = SystemConfig()
    # Lower threshold for testing
    config.ai.anomaly_threshold = 0.5

    orchestrator = SystemOrchestrator(config)

    print(f"Network: {'MOCK' if orchestrator.network.is_mock else 'WNTR'}")
    print(f"Monitored nodes: {orchestrator.agent._fleet.monitored_nodes}")

    # Build up baseline
    print("\n[Phase 1] Building baseline (30 steps)...")
    for i in range(30):
        result = orchestrator.step()

    print(f"Baseline complete. System status: {result.status.system_status.name}")

    # Show some Z-scores before leak
    print("\nZ-scores before leak (sample):")
    for node_id, data in list(result.metrics.items())[:5]:
        z = data.get('pressure_zscore')
        print(f"  {node_id}: P={data['pressure']:.1f}, Z={z if z else 'N/A'}")

    # Inject leak at a node WITH a pressure sensor (for detection)
    pressure_sensors, _ = orchestrator.network.get_sensor_locations()
    import random
    leak_node = random.choice(pressure_sensors)
    leak_rate = random.uniform(3.0, 8.0)
    orchestrator.leak_injector.inject_leak(leak_node, leak_rate, orchestrator.sim_time)
    if not orchestrator.network.is_mock:
        orchestrator.network.run_simulation()
    print(f"\n[Phase 2] Leak injected at: {leak_node} (has pressure sensor, {leak_rate:.1f} L/s)")

    # Monitor detection
    print("\nRunning detection (60 steps)...")
    print("-" * 60)

    detected = False
    first_estimate = None
    confirmed_estimate = None

    for i in range(60):
        result = orchestrator.step()

        # Show detailed info every 10 steps or on events
        status = result.status.system_status.name
        mode = result.status.sampling_mode.name

        # Track first detection
        if result.estimated_location and first_estimate is None:
            first_estimate = result.estimated_location
            detected = True

        # Track confirmed detection
        if status == 'LEAK_CONFIRMED' and confirmed_estimate is None:
            confirmed_estimate = result.estimated_location

        # Find the leak node's metrics
        leak_metrics = result.metrics.get(leak_node, {})
        leak_z = leak_metrics.get('pressure_zscore', 'N/A')
        leak_p = leak_metrics.get('pressure', 0)

        z_str = 'N/A' if leak_z is None else f'{leak_z:+.2f}'

        if i % 10 == 0 or result.anomaly or status in ['ALERT', 'INVESTIGATING', 'LEAK_CONFIRMED']:
            print(f"Step {i:3d}: status={status:15s} mode={mode:8s} "
                  f"leak_node_P={leak_p:6.1f} Z={z_str}")

            if result.anomaly:
                print(f"         ANOMALY: {result.anomaly.node_id} conf={result.anomaly.confidence:.2f}")

            if result.estimated_location:
                print(f"         ESTIMATE: {result.estimated_location}")

    print("-" * 60)
    print(f"\nFinal status: {result.status.system_status.name}")
    print(f"Ground truth: {leak_node}")
    print(f"First estimate: {first_estimate}")
    print(f"Confirmed estimate: {confirmed_estimate}")
    print(f"Final estimate: {result.estimated_location}")

    # Determine success based on confirmed estimate or first estimate
    best_estimate = confirmed_estimate or first_estimate

    if detected and best_estimate:
        if best_estimate == leak_node:
            print("\n[SUCCESS] EXACT MATCH - AI correctly identified leak location!")
        elif best_estimate in orchestrator.network.get_node_neighbors(leak_node, depth=2):
            print(f"\n[NEAR MISS] AI identified nearby node (within 2 hops)")
        else:
            print(f"\n[PARTIAL] Detection made but location differs")
    else:
        print("\n[FAIL] No detection")


if __name__ == "__main__":
    test_detection_pipeline()
