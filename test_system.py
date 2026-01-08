#!/usr/bin/env python3
"""
Quick test of the leak detection system without TUI.
Demonstrates the full detection pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from leak_detection.config import SystemConfig
from leak_detection.orchestrator import SystemOrchestrator


def test_leak_detection():
    """Test the full leak detection pipeline."""
    print("=" * 60)
    print("LEAK DETECTION SYSTEM - FUNCTIONAL TEST")
    print("=" * 60)

    # Initialize
    print("\n[1] Initializing system...")
    config = SystemConfig()
    orchestrator = SystemOrchestrator(config)

    print(f"    Network type: {'MOCK' if orchestrator.network.is_mock else 'WNTR'}")
    print(f"    Monitored nodes: {len(orchestrator.agent._fleet.monitored_nodes)}")

    # Run baseline (no leak)
    print("\n[2] Running baseline simulation (no leak)...")
    for i in range(20):  # ~100 minutes simulated
        result = orchestrator.step()

    print(f"    Simulation time: {result.sim_time:.0f}s")
    print(f"    System status: {result.status.system_status.name}")
    print(f"    Samples processed: {result.status.samples_processed}")

    # Inject leak
    print("\n[3] Injecting leak...")
    leak_node = orchestrator.inject_random_leak()
    print(f"    Leak injected at: {leak_node}")

    # Run detection
    print("\n[4] Running leak detection...")
    detection_time = None
    estimated_location = None

    for i in range(100):  # More simulation steps for detection
        result = orchestrator.step()

        # Debug: show anomaly scores periodically
        if i % 20 == 0 and result.anomaly:
            print(f"    t={result.sim_time:.0f}s: Top anomaly at {result.anomaly.node_id} "
                  f"(conf={result.anomaly.confidence:.2f})")

        if result.status.system_status.name in ['ALERT', 'INVESTIGATING', 'LEAK_CONFIRMED']:
            if detection_time is None:
                detection_time = result.sim_time
                print(f"    Alert triggered at t={detection_time:.0f}s")

        if result.estimated_location and estimated_location is None:
            estimated_location = result.estimated_location
            print(f"    AI estimated location: {estimated_location}")

        if result.status.system_status.name == 'LEAK_CONFIRMED':
            print(f"    Leak confirmed at t={result.sim_time:.0f}s")
            break

    # Get final estimate even if not confirmed
    if estimated_location is None:
        estimated_location = orchestrator.agent.get_estimated_leak_location()

    # Results
    print("\n[5] Detection Results:")
    print("-" * 40)
    print(f"    Ground Truth:      {leak_node}")
    print(f"    AI Estimate:       {estimated_location}")

    if estimated_location == leak_node:
        print("    Result:            [OK] EXACT MATCH!")
    elif estimated_location and estimated_location in orchestrator.network.get_node_neighbors(leak_node, depth=2):
        print("    Result:            [~] NEAR MISS (neighbor node)")
    elif estimated_location is None:
        print("    Result:            [--] No detection")
    else:
        print("    Result:            [X] Location mismatch")

    if detection_time:
        delay = detection_time - (result.sim_time - 50 * config.simulation.hydraulic_timestep_seconds)
        print(f"    Detection delay:   ~{delay:.0f}s")

    # Show summary
    print("\n[6] Full Summary:")
    print(orchestrator.get_detection_summary())

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_leak_detection()
